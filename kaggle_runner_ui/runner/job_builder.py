import json
import shlex
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .kaggle_api import run_kaggle
from .project_config import ROOT, resolve_tool_path


WORK_DIR = ROOT / "work"
OUTPUTS_DIR = ROOT / "outputs"
TEMPLATES_DIR = ROOT / "templates"


def job_slug(name):
    return "ocr-" + "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


def kernel_id(account, job):
    return f"{account['username']}/{job_slug(job['name'])}"


def prepare_project_jobs(project, accounts):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for job in project.get("jobs", []):
        account = accounts[job["account_id"]]
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
        prepared.append({"job": job, "account": account, "job_dir": job_dir, "kernel_id": kernel_id(account, job)})
    return prepared


def build_context(project, job, account, slug, code_file):
    engines = " ".join(job.get("engines", []))
    config_path = project.get("config_path", "")
    limit = str(job.get("limit", project.get("limit", 20)))
    setup_commands = project.get("commands", {}).get("setup") or ["python -m pip install -q -r requirements.txt"]
    run_commands = project.get("commands", {}).get("run") or []
    variables = {
        "engines": engines,
        "config_path": config_path,
        "limit": limit,
        "result_dir": f"/kaggle/working/results_{slug}",
        "prefetch_dir": f"/kaggle/working/prefetch_{slug}",
    }
    rendered_run_commands = [render_command(command, variables) for command in run_commands]
    install_commands = [f"python -m pip install -q -r {path}" for path in job.get("install_files", [])]
    return {
        "username": account["username"],
        "slug": slug,
        "code_file": code_file,
        "machine_shape": project.get("machine_shape", "NvidiaTeslaT4"),
        "repo_url": project["repo_url"],
        "setup_commands": [command_to_list(command) for command in setup_commands],
        "install_commands": [command_to_list(command) for command in install_commands],
        "run_commands": [command_to_list(command) for command in rendered_run_commands],
    }


def render_command(command, variables):
    for key, value in variables.items():
        command = command.replace("{{ " + key + " }}", str(value)).replace("{{" + key + "}}", str(value))
    return command


def command_to_list(command):
    return json.dumps(shlex.split(command))


def push_job(account, job_dir):
    return run_kaggle(account, ["kernels", "push", "-p", str(job_dir)], timeout=300)


def status_job(account, job):
    return run_kaggle(account, ["kernels", "status", kernel_id(account, job)], timeout=120)


def download_job_output(project_name, account, job):
    out_dir = OUTPUTS_DIR / project_name / job["name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, run_kaggle(account, ["kernels", "output", kernel_id(account, job), "-p", str(out_dir)], timeout=600)

