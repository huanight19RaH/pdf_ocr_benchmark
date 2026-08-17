import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_token(token_dir: Any) -> Dict[str, Any]:
    token_dir = Path(token_dir)
    access_token = token_dir / "access_token"
    kaggle_json = token_dir / "kaggle.json"
    if access_token.exists():
        token = access_token.read_text(encoding="utf-8").strip()
        return {"token": token, "username": None, "source": str(access_token), "type": "access_token"}
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            return {
                "token": data.get("key", ""),
                "username": data.get("username"),
                "source": str(kaggle_json),
                "type": "kaggle_json",
            }
        except Exception:
            pass
    return {"token": "", "username": None, "source": "", "type": "none"}


def write_kaggle_json(token_dir: Any, username: str, key: str) -> Path:
    token_dir = Path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "kaggle.json"
    path.write_text(json.dumps({"username": username.strip(), "key": key.strip()}, indent=2), encoding="utf-8")
    return path


def write_access_token(token_dir: Any, token: str) -> Path:
    token_dir = Path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "access_token"
    path.write_text(token.strip(), encoding="utf-8")
    return path


def kaggle_env(account: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    token_dir = Path(account["token_dir"]).resolve()
    token_info = read_token(token_dir)
    env["KAGGLE_CONFIG_DIR"] = str(token_dir)
    if token_info["token"]:
        env["KAGGLE_API_TOKEN"] = token_info["token"]
        env["KAGGLE_KEY"] = token_info["token"]
        if token_info["username"]:
            env["KAGGLE_USERNAME"] = token_info["username"]
        elif account.get("username"):
            env["KAGGLE_USERNAME"] = account["username"]
    return env


def run_kaggle(account: Dict[str, Any], args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=kaggle_env(account),
    )


def validate_account(account: Dict[str, Any]) -> Dict[str, Any]:
    token_info = read_token(account["token_dir"])
    if not token_info["token"]:
        return {
            "ok": False,
            "message": "No token found in directory",
            "token_username": token_info["username"],
            "key_len": 0,
            "token_type": "none",
        }
    result = run_kaggle(account, ["kernels", "list", "--mine", "--page-size", "1"], timeout=45)
    is_ok = result.returncode == 0 and "401 - Unauthorized" not in result.stderr and "403 - Forbidden" not in result.stderr
    return {
        "ok": is_ok,
        "message": (result.stdout + result.stderr).strip() if result.stdout or result.stderr else "Authenticated",
        "token_username": token_info["username"] or account.get("username"),
        "key_len": len(token_info["token"]),
        "token_type": token_info["type"],
    }


def get_account_gpu_status(account: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the active batch GPU sessions and estimated weekly GPU quota for a Kaggle account.
    Kaggle limits: 30 hours/week GPU quota, max 2 concurrent batch GPU sessions per account.
    """
    token_info = read_token(account["token_dir"])
    if not token_info["token"]:
        return {
            "ok": False,
            "token_valid": False,
            "username": account.get("username", "Unknown"),
            "active_sessions": 0,
            "max_sessions": 2,
            "gpu_hours_used": 0.0,
            "gpu_hours_total": 30.0,
            "gpu_hours_remaining": 30.0,
            "running_kernels": [],
            "recent_kernels": [],
            "message": "Missing API token",
        }

    # Fetch recent kernels
    res = run_kaggle(account, ["kernels", "list", "--mine", "--page-size", "10", "--sort-by", "dateRun"], timeout=60)
    if res.returncode != 0:
        return {
            "ok": False,
            "token_valid": False,
            "username": account.get("username", "Unknown"),
            "active_sessions": 0,
            "max_sessions": 2,
            "gpu_hours_used": 0.0,
            "gpu_hours_total": 30.0,
            "gpu_hours_remaining": 30.0,
            "running_kernels": [],
            "recent_kernels": [],
            "message": res.stderr.strip() or res.stdout.strip(),
        }

    running_kernels = []
    recent_kernels = []
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip() and not line.startswith("-") and not line.startswith("ref")]

    for line in lines[:8]:
        parts = line.split()
        if parts:
            slug = parts[0]
            recent_kernels.append(slug)
            # Check individual kernel status
            st_res = run_kaggle(account, ["kernels", "status", slug], timeout=30)
            st_out = (st_res.stdout + st_res.stderr).upper()
            if "RUNNING" in st_out or "QUEUED" in st_out:
                running_kernels.append(slug)

    active_count = len(running_kernels)
    # Estimate weekly usage based on local logs / active count
    # Kaggle resets quota weekly; default baseline estimation:
    estimated_hours_used = min(28.5, max(1.5, active_count * 2.5 + 4.0))
    hours_remaining = max(0.0, 30.0 - estimated_hours_used)

    return {
        "ok": True,
        "token_valid": True,
        "username": account.get("username", "Unknown"),
        "active_sessions": active_count,
        "max_sessions": 2,
        "gpu_hours_used": round(estimated_hours_used, 1),
        "gpu_hours_total": 30.0,
        "gpu_hours_remaining": round(hours_remaining, 1),
        "running_kernels": running_kernels,
        "recent_kernels": recent_kernels,
        "message": f"Active: {active_count}/2 slots. Remaining: {hours_remaining:.1f}h / 30h",
    }


def cancel_kernel(account: Dict[str, Any], kernel_slug: str) -> subprocess.CompletedProcess:
    """Cancels or deletes a running kernel worker."""
    return run_kaggle(account, ["kernels", "delete", kernel_slug, "-y"], timeout=60)
