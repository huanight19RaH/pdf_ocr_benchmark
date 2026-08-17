import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
_UI_DIR = _ROOT / "kaggle_runner_ui"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_UI_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_DIR))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kaggle_runner_ui.runner.kaggle_api import (
    KaggleApiResult,
    KaggleClientPool,
    SimpleTTLCache,
    cancel_kernel,
    clear_api_cache,
    download_kernel_output,
    get_account_gpu_status,
    gpu_status_cache,
    job_status_cache,
    push_kernel,
    query_kernel_status,
    read_token,
    run_kaggle,
    validate_account,
    validate_cache,
    write_access_token,
    write_kaggle_json,
)
from kaggle_runner_ui.runner.log_reader import (
    combine_summaries,
    find_job_files,
    read_csv,
    read_jsonl,
    read_text_tail,
)
from kaggle_runner_ui.server import app


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_caches():
    clear_api_cache()
    yield
    clear_api_cache()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# -----------------------------------------------------------------------------
# 1. Token Reader & Writer Tests
# -----------------------------------------------------------------------------
def test_read_token_none_or_empty():
    assert read_token(None) == {"token": "", "username": None, "source": "", "type": "none"}
    assert read_token("") == {"token": "", "username": None, "source": "", "type": "none"}


def test_read_token_non_existent(temp_dir):
    missing_path = temp_dir / "non_existent_token_dir"
    res = read_token(missing_path)
    assert res["token"] == ""
    assert res["type"] == "none"


def test_read_token_access_token_file(temp_dir):
    token_file = temp_dir / "access_token"
    token_file.write_text("   my_secret_token_123 \n", encoding="utf-8")
    res = read_token(temp_dir)
    assert res["token"] == "my_secret_token_123"
    assert res["type"] == "access_token"
    assert res["username"] is None


def test_read_token_empty_access_token(temp_dir):
    token_file = temp_dir / "access_token"
    token_file.write_text("   \n\t ", encoding="utf-8")
    res = read_token(temp_dir)
    assert res["token"] == ""
    assert res["type"] == "none"


def test_read_token_kaggle_json(temp_dir):
    k_json = temp_dir / "kaggle.json"
    k_json.write_text(json.dumps({"username": "test_user", "key": "kaggle_key_xyz"}), encoding="utf-8")
    res = read_token(temp_dir)
    assert res["token"] == "kaggle_key_xyz"
    assert res["username"] == "test_user"
    assert res["type"] == "kaggle_json"


def test_read_token_corrupted_kaggle_json(temp_dir):
    k_json = temp_dir / "kaggle.json"
    k_json.write_text("{corrupted json not valid : [}", encoding="utf-8")
    res = read_token(temp_dir)
    assert res["token"] == ""
    assert res["type"] == "none"


def test_write_kaggle_json_and_read(temp_dir):
    target = temp_dir / "account_1"
    path = write_kaggle_json(target, "rah_user", "key_abc_456")
    assert path.exists()
    token_info = read_token(target)
    assert token_info["token"] == "key_abc_456"
    assert token_info["username"] == "rah_user"
    assert token_info["type"] == "kaggle_json"


def test_write_access_token_and_read(temp_dir):
    target = temp_dir / "account_2"
    path = write_access_token(target, "raw_token_xyz")
    assert path.exists()
    token_info = read_token(target)
    assert token_info["token"] == "raw_token_xyz"
    assert token_info["type"] == "access_token"


# -----------------------------------------------------------------------------
# 2. TTL Cache & Concurrency Stress Tests
# -----------------------------------------------------------------------------
def test_ttl_cache_basic():
    cache = SimpleTTLCache(default_ttl=1.0)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"
    assert cache.get("k2", "def") == "def"

    cache.delete("k1")
    assert cache.get("k1") is None


def test_ttl_cache_expiration():
    cache = SimpleTTLCache(default_ttl=0.1)
    cache.set("k1", "v1", ttl=0.05)
    assert cache.get("k1") == "v1"
    time.sleep(0.08)
    assert cache.get("k1") is None


def test_ttl_cache_invalidate_prefix():
    cache = SimpleTTLCache(default_ttl=60.0)
    cache.set("acc:user1:status", "active")
    cache.set("acc:user1:quota", 30.0)
    cache.set("acc:user2:status", "idle")

    removed = cache.invalidate_prefix("acc:user1:")
    assert removed == 2
    assert cache.get("acc:user1:status") is None
    assert cache.get("acc:user1:quota") is None
    assert cache.get("acc:user2:status") == "idle"


def test_ttl_cache_get_or_set():
    cache = SimpleTTLCache(default_ttl=60.0)
    counter = [0]

    def producer():
        counter[0] += 1
        return f"val_{counter[0]}"

    res1 = cache.get_or_set("item", producer)
    assert res1 == "val_1"
    res2 = cache.get_or_set("item", producer)
    assert res2 == "val_1"
    assert counter[0] == 1


def test_ttl_cache_high_concurrency():
    cache = SimpleTTLCache(default_ttl=5.0)
    errors = []

    def worker(worker_id):
        try:
            for i in range(100):
                key = f"key_{i % 10}"
                cache.set(key, f"val_{worker_id}_{i}")
                val = cache.get(key)
                if i % 15 == 0:
                    cache.delete(key)
                if i % 30 == 0:
                    cache.invalidate_prefix("key_")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_clear_api_cache_scenarios():
    validate_cache.set("validate:acc1", {"ok": True})
    gpu_status_cache.set("gpu_status:acc1", {"active": 1})
    job_status_cache.set("kernel_status:acc1:job1", KaggleApiResult())
    validate_cache.set("validate:acc2", {"ok": True})

    # Clear only acc1
    clear_api_cache("acc1")
    assert validate_cache.get("validate:acc1") is None
    assert gpu_status_cache.get("gpu_status:acc1") is None
    assert job_status_cache.get("kernel_status:acc1:job1") is None
    assert validate_cache.get("validate:acc2") is not None

    # Clear all
    clear_api_cache()
    assert validate_cache.get("validate:acc2") is None


# -----------------------------------------------------------------------------
# 3. Account Validation & GPU Status Tests
# -----------------------------------------------------------------------------
def test_validate_account_no_token():
    account = {"id": "acc_empty", "token_dir": "non/existent/path"}
    res = validate_account(account)
    assert res["ok"] is False
    assert "No token found" in res["message"]


def test_validate_account_mocked_success(temp_dir):
    token_file = temp_dir / "access_token"
    token_file.write_text("valid_token", encoding="utf-8")
    account = {"id": "acc_test", "username": "rah_user", "token_dir": str(temp_dir)}

    with patch("kaggle_runner_ui.runner.kaggle_api.client_pool.get_client", return_value=None):
        with patch("kaggle_runner_ui.runner.kaggle_api.run_kaggle") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Authenticated as rah_user", stderr="")
            res = validate_account(account, force_refresh=True)
            assert res["ok"] is True
            assert "Authenticated as rah_user" in res["message"]

            # Check cached return
            cached_res = validate_account(account, force_refresh=False)
            assert cached_res["ok"] is True
            assert mock_run.call_count == 1


def test_validate_account_unauthorized(temp_dir):
    token_file = temp_dir / "access_token"
    token_file.write_text("invalid_token", encoding="utf-8")
    account = {"id": "acc_bad", "username": "bad_user", "token_dir": str(temp_dir)}

    with patch("kaggle_runner_ui.runner.kaggle_api.client_pool.get_client", return_value=None):
        with patch("kaggle_runner_ui.runner.kaggle_api.run_kaggle") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="401 - Unauthorized")
            res = validate_account(account, force_refresh=True)
            assert res["ok"] is False
            assert "401 - Unauthorized" in res["message"]


def test_get_account_gpu_status_no_token():
    account = {"id": "acc_no_token", "username": "user1", "token_dir": ""}
    status = get_account_gpu_status(account)
    assert status["ok"] is False
    assert status["token_valid"] is False
    assert status["active_sessions"] == 0
    assert status["gpu_hours_remaining"] == 30.0


def test_get_account_gpu_status_mocked(temp_dir):
    write_access_token(temp_dir, "tok123")
    account = {"id": "acc_gpu", "username": "gpu_user", "token_dir": str(temp_dir)}

    with patch("kaggle_runner_ui.runner.kaggle_api.client_pool.get_client", return_value=None):
        with patch("kaggle_runner_ui.runner.kaggle_api.run_kaggle") as mock_run:
            # 1st call for kernels list, subsequent for kernel status
            def side_effect(acc, args, timeout=60):
                if "list" in args:
                    return MagicMock(returncode=0, stdout="gpu_user/ocr-bench-1  title1\ngpu_user/ocr-bench-2  title2\n", stderr="")
                elif "status" in args:
                    slug = args[-1]
                    if "ocr-bench-1" in slug:
                        return MagicMock(returncode=0, stdout="Kernel status is running", stderr="")
                    return MagicMock(returncode=0, stdout="Kernel status is complete", stderr="")
                return MagicMock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = side_effect

            res = get_account_gpu_status(account, force_refresh=True)
            assert res["ok"] is True
            assert res["token_valid"] is True
            assert res["active_sessions"] == 1
            assert any("ocr-bench-1" in k for k in res["running_kernels"])
            assert len(res["recent_kernels"]) == 2
            assert res["gpu_hours_remaining"] < 30.0


# -----------------------------------------------------------------------------
# 4. Log Reader Edge Case Tests
# -----------------------------------------------------------------------------
def test_read_text_tail_edge_cases(temp_dir):
    assert read_text_tail(None) == ""
    assert read_text_tail(temp_dir / "non_existent.log") == ""

    # Empty file
    empty_file = temp_dir / "empty.log"
    empty_file.write_text("", encoding="utf-8")
    assert read_text_tail(empty_file) == ""

    # Normal file with lines
    log_file = temp_dir / "sample.log"
    lines = [f"Line {i}" for i in range(100)]
    log_file.write_text("\n".join(lines), encoding="utf-8")

    tail_lines = read_text_tail(log_file, max_lines=5)
    assert tail_lines.splitlines() == [f"Line {i}" for i in range(95, 100)]

    tail_chars = read_text_tail(log_file, max_chars=20)
    assert len(tail_chars) == 20


def test_read_text_tail_large_binary(temp_dir):
    bin_file = temp_dir / "large.bin"
    # Write 100KB with some utf-8 and non-utf8 bytes
    content = b"log header\n" + (b"random log line with data\n" * 4000)
    bin_file.write_bytes(content)

    tail = read_text_tail(bin_file, max_chars=500, max_lines=10)
    assert len(tail) > 0
    assert len(tail.splitlines()) <= 10


def test_read_csv_edge_cases(temp_dir):
    assert read_csv(None).empty
    assert read_csv(temp_dir / "missing.csv").empty

    empty_csv = temp_dir / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    assert read_csv(empty_csv).empty

    corrupted_csv = temp_dir / "corrupted.csv"
    corrupted_csv.write_text("a,b,c\n1,2\n3,4,5,6,7,8\n", encoding="utf-8")
    # Should safely read or return df without uncaught crash
    df = read_csv(corrupted_csv)
    assert isinstance(df, pd.DataFrame)


def test_read_jsonl_edge_cases(temp_dir):
    assert read_jsonl(None).empty
    assert read_jsonl(temp_dir / "missing.jsonl").empty

    jsonl_file = temp_dir / "test.jsonl"
    jsonl_file.write_text(
        '{"step": 1, "loss": 0.5}\n{corrupted line}\n{"step": 2, "loss": 0.3}\n',
        encoding="utf-8",
    )
    df = read_jsonl(jsonl_file)
    assert not df.empty
    assert len(df) == 2
    assert list(df["step"]) == [1, 2]


def test_find_job_files_edge_cases(temp_dir):
    assert find_job_files(None) == {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}
    assert find_job_files(temp_dir / "non_existent") == {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}

    # Corrupted zip
    bad_zip = temp_dir / "results_bad.zip"
    bad_zip.write_bytes(b"not a valid zip file binary")
    files = find_job_files(temp_dir)
    assert bad_zip in files["zips"]


def test_combine_summaries_edge_cases(temp_dir):
    assert combine_summaries(None).empty
    assert combine_summaries(temp_dir / "missing_dir").empty

    # Create dummy structure
    job1_dir = temp_dir / "job_paddle"
    job1_dir.mkdir()
    (job1_dir / "summary.csv").write_text("model,cer,wer\npaddle,0.05,0.12\n", encoding="utf-8")

    job2_dir = temp_dir / "job_surya"
    job2_dir.mkdir()
    (job2_dir / "summary.csv").write_text("model,cer,wer\nsurya,0.03,0.08\n", encoding="utf-8")

    combined = combine_summaries(temp_dir)
    assert not combined.empty
    assert len(combined) == 2
    assert "job" in combined.columns
    assert set(combined["job"]) == {"job_paddle", "job_surya"}


# -----------------------------------------------------------------------------
# 5. FastAPI Server Integration Tests
# -----------------------------------------------------------------------------
def test_server_accounts_endpoint():
    client = TestClient(app)
    response = client.get("/api/accounts")
    assert response.status_code == 200
    data = response.json()
    assert "accounts" in data
    assert isinstance(data["accounts"], list)


def test_server_validate_nonexistent_account():
    client = TestClient(app)
    response = client.post("/api/accounts/non_existent_account_9999/validate")
    assert response.status_code == 404


def test_server_logs_endpoint():
    client = TestClient(app)
    response = client.get("/api/logs/non_existent_project")
    assert response.status_code == 200
    data = response.json()
    assert data["files"] == []
    assert "No log files available" in data["content"]


def test_server_analytics_endpoint():
    client = TestClient(app)
    response = client.get("/api/analytics/non_existent_project")
    assert response.status_code == 200
    data = response.json()
    assert data["has_data"] is False
    assert data["summary"] == []
