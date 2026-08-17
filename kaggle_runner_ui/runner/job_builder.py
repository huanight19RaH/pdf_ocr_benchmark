import concurrent.futures
import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from .kaggle_api import cancel_kernel, run_kaggle
from .project_config import ROOT, resolve_tool_path

WORK_DIR = ROOT / "work"
OUTPUTS_DIR = ROOT / "outputs"
TEMPLATES_DIR = ROOT / "templates"


def job_slug(name: str) -> str:
    return "ocr-" + "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


def kernel_id(account: Dict[str, Any], job: Dict[str, Any]) -> str:
    return f"{account['username']}/{job_slug(job['name'])}"


def prepare_project_jobs(project: Dict[str, Any], accounts: Dict[str, Any]) -> List[Dict[str, Any]]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for job in project.get("jobs", []):
        account = accounts.get(job["account_id"])
        if not account:
            continue
        slug = job_slug(job["name"])
        job_dir = WORK_DIR / project["name"] / slug
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True)
        code_file = f"{slug}.py"
        context = build_context(project, job, account, slug, code_file)
        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)
        metadata = env.get_template("kernel-metadata.json.j2").render(**context)
        script = env.get_template("kernel_script.py.j2").render(**context)
        (job_dir / "kernel-metadata.json").write_text(metadata, encoding="utf-8")
        (job_dir / code_file).write_text(script, encoding="utf-8")
        prepared.append({
            "job": job,
            "account": account,
            "job_dir": job_dir,
            "kernel_id": kernel_id(account, job),
            "slug": slug,
        })
    return prepared


def build_context(
    project: Dict[str, Any],
    job: Dict[str, Any],
    account: Dict[str, Any],
    slug: str,
    code_file: str,
) -> Dict[str, Any]:
    engines = " ".join(job.get("engines", []))
    config_path = project.get("config_path", "configs/kaggle_doclaynet_science.yaml")
    limit = str(job.get("limit", project.get("limit", 20)))
    epochs = str(job.get("epochs", 10))
    job_type = job.get("job_type", "benchmark")
    finetune_model = job.get("finetune_model", "paddleocr")

    setup_commands = project.get("commands", {}).get("setup") or ["python -m pip install -q -r requirements.txt"]

    # Generate custom run / finetune commands if job_type == 'finetune'
    if job_type == "finetune":
        finetune_commands = [
            f"python -m ocr_benchmark.finetune --config {config_path} --models {finetune_model} --limit {limit} --epochs {epochs} --output-dir /kaggle/working/ocr_finetune_outputs --execute"
        ]
        run_commands = [
            f"python -m ocr_benchmark.prefetch_models --config {config_path} --engines {engines} --output-dir /kaggle/working/prefetch_{slug}",
            f"python -m ocr_benchmark.benchmark --config {config_path} --engines {engines} --limit {limit} --output-dir /kaggle/working/results_{slug}",
        ]
    else:
        finetune_commands = []
        run_commands = project.get("commands", {}).get("run") or [
            f"python -m ocr_benchmark.prefetch_models --engines {engines} --output-dir /kaggle/working/prefetch_{slug}",
            f"python -m ocr_benchmark.benchmark --config {config_path} --engines {engines} --limit {limit} --output-dir /kaggle/working/results_{slug}",
        ]

    variables = {
        "engines": engines,
        "config_path": config_path,
        "limit": limit,
        "epochs": epochs,
        "result_dir": f"/kaggle/working/results_{slug}",
        "prefetch_dir": f"/kaggle/working/prefetch_{slug}",
    }
    rendered_run_commands = [render_command(command, variables) for command in run_commands]
    install_commands = [f"python -m pip install -q -r {path}" for path in job.get("install_files", [])]

    return {
        "username": account["username"],
        "slug": slug,
        "code_file": code_file,
        "machine_shape": job.get("machine_shape", project.get("machine_shape", "NvidiaTeslaT4")),
        "repo_url": project.get("repo_url", "https://github.com/huanight19RaH/pdf_ocr_benchmark.git"),
        "setup_commands": [command_to_list(cmd) for cmd in setup_commands],
        "install_commands": [command_to_list(cmd) for cmd in install_commands],
        "finetune_commands": [command_to_list(cmd) for cmd in finetune_commands],
        "run_commands": [command_to_list(cmd) for cmd in rendered_run_commands],
    }


def render_command(command: str, variables: Dict[str, Any]) -> str:
    for key, value in variables.items():
        command = command.replace("{{ " + key + " }}", str(value)).replace("{{" + key + "}}", str(value))
    return command


def command_to_list(command: str) -> str:
    return json.dumps(shlex.split(command))


def push_job(account: Dict[str, Any], job_dir: Path) -> Any:
    return run_kaggle(account, ["kernels", "push", "-p", str(job_dir)], timeout=300)


def status_job(account: Dict[str, Any], job: Dict[str, Any]) -> Any:
    return run_kaggle(account, ["kernels", "status", kernel_id(account, job)], timeout=120)


def stop_job(account: Dict[str, Any], job: Dict[str, Any]) -> Any:
    k_id = kernel_id(account, job)
    return cancel_kernel(account, k_id)


def download_job_output(project_name: str, account: Dict[str, Any], job: Dict[str, Any]) -> Tuple[Path, Any]:
    out_dir = OUTPUTS_DIR / project_name / job["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_kaggle(account, ["kernels", "output", kernel_id(account, job), "-p", str(out_dir), "--force"], timeout=600)
    return out_dir, res


def run_parallel_tasks(tasks: List[Tuple[Callable, Tuple]], max_workers: int = 4) -> List[Any]:
    """Executes multiple Kaggle operations concurrently across threads."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(fn, *args): (fn, args) for fn, args in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
    return results
