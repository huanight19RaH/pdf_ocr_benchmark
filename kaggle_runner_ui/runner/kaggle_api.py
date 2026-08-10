import json
import os
import subprocess
import sys
from pathlib import Path


def read_token(token_dir):
    token_dir = Path(token_dir)
    access_token = token_dir / "access_token"
    kaggle_json = token_dir / "kaggle.json"
    if access_token.exists():
        token = access_token.read_text(encoding="utf-8").strip()
        return {"token": token, "username": None, "source": str(access_token)}
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        return {"token": data.get("key", ""), "username": data.get("username"), "source": str(kaggle_json)}
    return {"token": "", "username": None, "source": ""}


def write_kaggle_json(token_dir, username, key):
    token_dir = Path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}, indent=2), encoding="utf-8")
    return path


def write_access_token(token_dir, token):
    token_dir = Path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "access_token"
    path.write_text(token.strip(), encoding="utf-8")
    return path


def kaggle_env(account):
    env = os.environ.copy()
    token_info = read_token(account["token_dir"])
    env["KAGGLE_CONFIG_DIR"] = str(Path(account["token_dir"]).resolve())
    if token_info["token"]:
        env["KAGGLE_API_TOKEN"] = token_info["token"]
    return env


def run_kaggle(account, args, timeout=120):
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=kaggle_env(account),
    )


def validate_account(account):
    token_info = read_token(account["token_dir"])
    if not token_info["token"]:
        return {"ok": False, "message": "No token found", "token_username": token_info["username"], "key_len": 0}
    result = run_kaggle(account, ["kernels", "list", "--mine", "--page-size", "1"], timeout=60)
    return {
        "ok": result.returncode == 0,
        "message": (result.stdout + result.stderr).strip(),
        "token_username": token_info["username"],
        "key_len": len(token_info["token"]),
    }

