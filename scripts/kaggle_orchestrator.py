import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create, submit, monitor, and download Kaggle OCR benchmark jobs.")
    parser.add_argument("--config", default="configs/kaggle_accounts.example.yaml")
    parser.add_argument("--work-dir", default="kaggle_remote_jobs")
    parser.add_argument("--action", choices=["prepare", "push", "status", "output", "all"], default="prepare")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-minutes", type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work_dir = Path(args.work_dir)
    job_dirs = prepare_jobs(config, work_dir)

    if args.action == "prepare":
        print(f"Prepared {len(job_dirs)} jobs in {work_dir}")
        return
    if args.action in {"push", "all"}:
        for job, job_dir in job_dirs:
            run_kaggle(job, ["kernels", "push", "-p", str(job_dir)])
    if args.action in {"status", "all"}:
        wait_for_jobs(config, job_dirs, args.poll_seconds, args.max_wait_minutes, wait=args.action == "all")
    if args.action in {"output", "all"}:
        for job, job_dir in job_dirs:
            out_dir = work_dir / "outputs" / job["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            run_kaggle(job, ["kernels", "output", kernel_id(job), "-p", str(out_dir)])


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
            "title": f"OCR Benchmark {job['name']}",
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
    engines = " ".join(job["engines"])
    install_files = job.get("install_files", [])
    install_lines = "\n".join(
        f"run(['python', '-m', 'pip', 'install', '-q', '-r', '{req}'])" for req in install_files
    )
    return f"""import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


os.chdir('/kaggle/working')
repo_dir = Path('/kaggle/working/ocr_benchmark')
if repo_dir.exists():
    shutil.rmtree(repo_dir)
run(['git', 'clone', '--depth', '1', '{repo_url}', str(repo_dir)])
os.chdir(repo_dir)
sys.path.insert(0, str(repo_dir / 'src'))

run(['python', '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
{install_lines}

run([
    'python', '-m', 'ocr_benchmark.prefetch_models',
    '--engines', *'{engines}'.split(),
    '--output-dir', '/kaggle/working/prefetch_{job_slug(job["name"])}',
])

run([
    'python', '-m', 'ocr_benchmark.benchmark',
    '--config', 'configs/kaggle_doclaynet_science.yaml',
    '--engines', *'{engines}'.split(),
    '--limit', '{limit}',
    '--output-dir', '/kaggle/working/results_{job_slug(job["name"])}',
])

run([
    'bash', '-lc',
    'cd /kaggle/working && zip -qr results_{job_slug(job["name"])}.zip '
    'results_{job_slug(job["name"])} prefetch_{job_slug(job["name"])}'
])
"""


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
    env["KAGGLE_CONFIG_DIR"] = str(Path(job["token_dir"]).expanduser().resolve())
    return subprocess.run(["kaggle", *args], check=check, env=env, capture_output=True, text=True)


def kernel_id(job):
    return f"{job['username']}/{job_slug(job['name'])}"


def job_slug(name):
    return "ocr-" + "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


if __name__ == "__main__":
    main()

