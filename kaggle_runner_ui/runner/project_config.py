from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ACCOUNTS_PATH = DATA_DIR / "accounts.yaml"
PROJECTS_PATH = DATA_DIR / "projects.yaml"
ACCOUNTS_EXAMPLE_PATH = DATA_DIR / "accounts.example.yaml"
PROJECTS_EXAMPLE_PATH = DATA_DIR / "projects.example.yaml"


def ensure_data_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ACCOUNTS_PATH.exists():
        ACCOUNTS_PATH.write_text(ACCOUNTS_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    if not PROJECTS_PATH.exists():
        PROJECTS_PATH.write_text(PROJECTS_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def load_accounts():
    ensure_data_files()
    return yaml.safe_load(ACCOUNTS_PATH.read_text(encoding="utf-8")) or {"accounts": []}


def save_accounts(config):
    ensure_data_files()
    ACCOUNTS_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_projects():
    ensure_data_files()
    return yaml.safe_load(PROJECTS_PATH.read_text(encoding="utf-8")) or {"projects": []}


def save_projects(config):
    ensure_data_files()
    PROJECTS_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def accounts_by_id():
    return {account["id"]: account for account in load_accounts().get("accounts", [])}


def get_project(name):
    for project in load_projects().get("projects", []):
        if project.get("name") == name:
            return project
    return None


def token_file_paths(token_dir):
    token_dir = resolve_tool_path(token_dir)
    return token_dir / "kaggle.json", token_dir / "access_token"


def resolve_tool_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

