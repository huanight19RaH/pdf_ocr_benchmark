import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def main():
    copy_if_exists(REPO_ROOT / "configs" / "kaggle_accounts.yaml", ROOT / "data" / "projects.yaml", mode="projects")
    source_tokens = REPO_ROOT / ".kaggle_tokens"
    target_tokens = ROOT / "data" / "tokens"
    target_tokens.mkdir(parents=True, exist_ok=True)
    for account_dir in ["account1", "account2", "account3"]:
        src_dir = source_tokens / account_dir
        dst_dir = target_tokens / account_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["kaggle.json", "access_token"]:
            if (src_dir / filename).exists():
                shutil.copy2(src_dir / filename, dst_dir / filename)
                print(f"copied token {account_dir}/{filename}")
    write_accounts_from_current_config()
    print("Done. Run: cd kaggle_runner_ui && python -m streamlit run app.py")


def copy_if_exists(src: Path, dst: Path, mode: str):
    if not src.exists():
        return
    if mode == "projects":
        import yaml

        old = yaml.safe_load(src.read_text(encoding="utf-8"))
        project = {
            "name": "pdf_ocr_benchmark",
            "repo_url": old["repo_url"],
            "local_path": str(REPO_ROOT).replace("\\", "/"),
            "config_path": "configs/kaggle_doclaynet_science.yaml",
            "limit": old.get("limit", 20),
            "machine_shape": old.get("machine_shape", "NvidiaTeslaT4"),
            "commands": {
                "setup": ["python -m pip install -q -r requirements.txt"],
                "run": [
                    "python -m ocr_benchmark.prefetch_models --engines {{ engines }} --output-dir {{ prefetch_dir }}",
                    "python -m ocr_benchmark.benchmark --config {{ config_path }} --engines {{ engines }} --limit {{ limit }} --output-dir {{ result_dir }}",
                ],
            },
            "jobs": [
                {
                    "name": job["name"],
                    "account_id": account_id_for_token_dir(job["token_dir"]),
                    "engines": job["engines"],
                    "install_files": job.get("install_files", []),
                }
                for job in old.get("jobs", [])
            ],
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(yaml.safe_dump({"projects": [project]}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {dst}")


def write_accounts_from_current_config():
    import yaml

    src = REPO_ROOT / "configs" / "kaggle_accounts.yaml"
    if not src.exists():
        return
    old = yaml.safe_load(src.read_text(encoding="utf-8"))
    seen = {}
    for job in old.get("jobs", []):
        account_id = account_id_for_token_dir(job["token_dir"])
        seen[account_id] = {
            "id": account_id,
            "username": job["username"],
            "token_dir": f"data/tokens/{account_id}",
        }
    target = ROOT / "data" / "accounts.yaml"
    target.write_text(yaml.safe_dump({"accounts": list(seen.values())}, sort_keys=False), encoding="utf-8")
    print(f"wrote {target}")


def account_id_for_token_dir(token_dir: str) -> str:
    name = Path(token_dir).name
    return name if name else token_dir.replace("/", "_").replace("\\", "_")


if __name__ == "__main__":
    main()

