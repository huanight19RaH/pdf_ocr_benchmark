import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create, submit, monitor, and download Kaggle OCR benchmark jobs.")
    parser.add_argument("--config", default="configs/kaggle_accounts.example.yaml")
    parser.add_argument("--work-dir", default="kaggle_remote_jobs")
    parser.add_argument("--action", choices=["prepare", "push", "status", "output", "delete", "all"], default="prepare")
    parser.add_argument("--job", help="Run the action for one job name only, for example paddleocr-ft or paddleocr-vl.")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-minutes", type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.job:
        matched = [job for job in config["jobs"] if job["name"] == args.job]
        if not matched and args.job in {"finetune-paddleocr", "paddleocr-ft"}:
            matched = [job for job in config["jobs"] if job["name"] in {"finetune-paddleocr", "paddleocr-ft"}]
        config["jobs"] = matched
        if not config["jobs"]:
            raise KeyError(f"No job named '{args.job}' in {args.config}")
    if args.action == "delete" and not args.job:
        raise ValueError("--action delete requires --job so you do not delete every kernel by accident.")
    work_dir = Path(args.work_dir)
    job_dirs = prepare_jobs(config, work_dir)

    if args.action == "prepare":
        print(f"Prepared {len(job_dirs)} jobs in {work_dir}")
        return
    if args.action in {"push", "all"}:
        for job, job_dir in job_dirs:
            result = run_kaggle(job, ["kernels", "push", "-p", str(job_dir)], check=False)
            print(f"\n[{job['name']}: push {kernel_id(job)}]\n{result.stdout}{result.stderr}", flush=True)
            if result.returncode != 0:
                raise RuntimeError(f"Push failed for {job['name']} ({kernel_id(job)})")
    if args.action in {"status", "all"}:
        wait_for_jobs(config, job_dirs, args.poll_seconds, args.max_wait_minutes, wait=args.action == "all")
    if args.action in {"output", "all"}:
        for job, job_dir in job_dirs:
            out_dir = work_dir / "outputs" / job["name"]
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = run_kaggle(job, ["kernels", "output", kernel_id(job), "-p", str(out_dir), "--force"], check=False)
            print(f"\n[{job['name']}: output {kernel_id(job)}]\n{result.stdout}{result.stderr}", flush=True)
    if args.action == "delete":
        for job, job_dir in job_dirs:
            result = run_kaggle(job, ["kernels", "delete", "-y", kernel_id(job)], check=False)
            print(f"\n[{job['name']}: delete {kernel_id(job)}]\n{result.stdout}{result.stderr}", flush=True)
            if result.returncode != 0:
                raise RuntimeError(f"Delete failed for {job['name']} ({kernel_id(job)})")


def prepare_jobs(config, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    job_dirs = []
    for job in config["jobs"]:
        slug = job_slug(job["name"])
        job_dir = work_dir / slug
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True)
        code_file = f"{slug}.py"
        metadata = {
            "id": f"{job['username']}/{slug}",
            "title": slug,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": config.get("machine_shape", "NvidiaTeslaT4"),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (job_dir / code_file).write_text(render_job_script(config, job), encoding="utf-8")
        job_dirs.append((job, job_dir))
    return job_dirs


def render_job_script(config, job):
    repo_url = config["repo_url"]
    limit = int(job.get("limit", config.get("limit", 20)))
    epochs = int(job.get("epochs", 10))
    engines = " ".join(job.get("engines", []))
    install_files = job.get("install_files", [])
    install_lines = "\n".join(
        f"run(['python', '-m', 'pip', 'install', '-q', '-r', '{req}'])" for req in install_files
    )
    slug = job_slug(job["name"])
    is_finetune = (
        job.get("job_type") == "finetune"
        or "paddleocr_ft" in job.get("engines", [])
        or "paddleocr-ft" in job["name"]
        or "finetune" in job["name"]
        or bool(job.get("is_finetune", False))
    )
    finetune_model = job.get("finetune_model", "paddleocr")

    finetune_block = ""
    zip_targets = f"results_{slug} prefetch_{slug} job_debug_{slug}.log"
    if is_finetune:
        zip_targets += " ocr_finetune_outputs"
        finetune_block = f"""write_log('=== Running Finetune Pipeline: {finetune_model} ({epochs} epochs) ===')
run([
    'python', '-m', 'ocr_benchmark.finetune',
    '--config', 'configs/kaggle_doclaynet_science.yaml',
    '--models', '{finetune_model}',
    '--limit', '{limit}',
    '--epochs', '{epochs}',
    '--output-dir', str(FINETUNE_DIR),
    '--execute',
])"""

    return f"""import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

LOG_PATH = Path('/kaggle/working/job_debug_{slug}.log')
RESULT_DIR = Path('/kaggle/working/results_{slug}')
PREFETCH_DIR = Path('/kaggle/working/prefetch_{slug}')
FINETUNE_DIR = Path('/kaggle/working/ocr_finetune_outputs')


def run(cmd):
    line = '+ ' + ' '.join(cmd)
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\\n')
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed with code {{proc.returncode}}: {{line}}')


def write_log(message):
    print(message, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(message + '\\n')


try:
    os.chdir('/kaggle/working')
    repo_dir = Path('/kaggle/working/ocr_benchmark')
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run(['git', 'clone', '--depth', '1', '--filter=blob:none', '{repo_url}', str(repo_dir)])
    shutil.rmtree(repo_dir / '.git', ignore_errors=True)
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir / 'src'))
    os.environ['PYTHONPATH'] = str(repo_dir / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    write_log('PYTHONPATH=' + os.environ['PYTHONPATH'])

    run(['python', '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
{indent_lines(install_lines, 4)}
{indent_lines(finetune_block, 4)}
    run([
        'python', '-m', 'ocr_benchmark.prefetch_models',
        '--config', 'configs/kaggle_doclaynet_science.yaml',
        '--engines', *'{engines}'.split(),
        '--output-dir', str(PREFETCH_DIR),
    ])

    run([
        'python', '-m', 'ocr_benchmark.benchmark',
        '--config', 'configs/kaggle_doclaynet_science.yaml',
        '--engines', *'{engines}'.split(),
        '--limit', '{limit}',
        '--output-dir', str(RESULT_DIR),
    ])
except Exception:
    write_log('FAILED WITH TRACEBACK:')
    write_log(traceback.format_exc())
    raise
finally:
    os.chdir('/kaggle/working')
    if Path('/kaggle/working/ocr_benchmark').exists():
        shutil.rmtree('/kaggle/working/ocr_benchmark', ignore_errors=True)
    if Path('/kaggle/working/PaddleOCR').exists():
        shutil.rmtree('/kaggle/working/PaddleOCR', ignore_errors=True)
    rec_data_dir = Path('/kaggle/working/ocr_finetune_outputs/paddleocr/paddleocr_rec_dataset/images')
    if rec_data_dir.exists():
        shutil.rmtree(rec_data_dir, ignore_errors=True)
    run([
        'bash', '-lc',
        'cd /kaggle/working && zip -qr {slug}.zip '
        '{zip_targets} || true'
    ])
"""


def indent_lines(text, spaces):
    if not text:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def wait_for_jobs(config, job_dirs, poll_seconds, max_wait_minutes, wait):
    deadline = time.time() + max_wait_minutes * 60
    pending = {job["name"]: job for job, _ in job_dirs}
    while pending:
        finished = []
        for name, job in pending.items():
            result = run_kaggle(job, ["kernels", "status", kernel_id(job)], check=False)
            status_text = (result.stdout + "\n" + result.stderr).lower()
            print(f"\n[{name}]\n{result.stdout}{result.stderr}", flush=True)
            if any(word in status_text for word in ["complete", "error", "failed", "cancelled"]):
                finished.append(name)
        for name in finished:
            pending.pop(name, None)
        if not wait or not pending:
            break
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for jobs: {sorted(pending)}")
        time.sleep(poll_seconds)


def run_kaggle(job, args, check=True):
    env = os.environ.copy()
    token_dir = Path(job["token_dir"]).expanduser().resolve()
    env["KAGGLE_CONFIG_DIR"] = str(token_dir)
    access_token_file = token_dir / "access_token"
    legacy_token_file = token_dir / "kaggle.json"
    if access_token_file.exists():
        env["KAGGLE_API_TOKEN"] = access_token_file.read_text(encoding="utf-8").strip()
    elif legacy_token_file.exists():
        data = json.loads(legacy_token_file.read_text(encoding="utf-8"))
        env["KAGGLE_API_TOKEN"] = data["key"]
        env["KAGGLE_KEY"] = data["key"]
        env["KAGGLE_USERNAME"] = data.get("username", job.get("username", ""))
    return subprocess.run([sys.executable, "-m", "kaggle", *args], check=check, env=env, capture_output=True, text=True)


def kernel_id(job):
    return f"{job['username']}/{job_slug(job['name'])}"


def job_slug(name):
    return "ocr-" + "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


if __name__ == "__main__":
    main()
