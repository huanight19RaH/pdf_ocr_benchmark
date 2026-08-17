from pathlib import Path
import sys

from scripts.kaggle_orchestrator import prepare_jobs, render_job_script


def test_render_job_script_for_finetune(tmp_path):
    config = {
        "repo_url": "https://github.com/huanight19RaH/pdf_ocr_benchmark.git",
        "limit": 20,
        "machine_shape": "NvidiaTeslaT4",
    }
    job = {
        "name": "finetune-paddleocr",
        "username": "TThanh13",
        "token_dir": ".kaggle_tokens/account2",
        "is_finetune": True,
        "engines": ["paddleocr_ft"],
        "install_files": ["requirements-kaggle-paddleocr.txt"],
        "epochs": 10,
    }

    script = render_job_script(config, job)
    assert "ocr_benchmark.finetune" in script
    assert "'--epochs', '10'" in script
    assert "ocr_benchmark.benchmark" in script
    assert "paddleocr_ft" in script
    assert "ocr_finetune_outputs" in script


def test_prepare_jobs_creates_metadata_and_script(tmp_path):
    config = {
        "repo_url": "https://github.com/huanight19RaH/pdf_ocr_benchmark.git",
        "limit": 10,
        "machine_shape": "NvidiaTeslaT4",
        "jobs": [
            {
                "name": "finetune-paddleocr",
                "username": "TThanh13",
                "token_dir": ".kaggle_tokens/account2",
                "is_finetune": True,
                "engines": ["paddleocr_ft"],
                "install_files": ["requirements-kaggle-paddleocr.txt"],
            }
        ],
    }
    job_dirs = prepare_jobs(config, tmp_path)
    assert len(job_dirs) == 1
    job, job_dir = job_dirs[0]
    assert (job_dir / "kernel-metadata.json").exists()
    assert (job_dir / "ocr-finetune-paddleocr.py").exists()
