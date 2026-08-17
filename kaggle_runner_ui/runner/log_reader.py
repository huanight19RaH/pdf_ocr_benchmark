import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def find_job_files(output_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Finds and discovers all job log files, summary CSVs, error logs, and result zips.
    Safely auto-extracts zip files and handles corrupted or missing archives.
    """
    if not output_dir:
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}

    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}

    try:
        zips = sorted(output_dir.rglob("results_*.zip"))
        for zip_path in zips:
            extract_dir = zip_path.with_suffix("")
            if not extract_dir.exists():
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(extract_dir)
                except (zipfile.BadZipFile, OSError, Exception):
                    pass

        return {
            "logs": sorted(output_dir.rglob("job_debug_*.log")),
            "summaries": sorted(output_dir.rglob("summary.csv")),
            "errors": sorted(output_dir.rglob("errors.csv")),
            "prefetch": sorted(output_dir.rglob("prefetch_status.jsonl")),
            "zips": sorted(output_dir.rglob("results_*.zip")),
        }
    except Exception:
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}


def read_text_tail(path: Union[str, Path], max_chars: int = 20000, max_lines: Optional[int] = None) -> str:
    """
    Reads the tail of a text or log file safely and efficiently.
    Supports both max_chars and max_lines constraints.
    Handles large files, non-existent files, empty files, and binary content safely.
    """
    if not path:
        return ""

    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""

        file_size = p.stat().st_size
        if file_size == 0:
            return ""

        # Efficient tail reading for large files
        bytes_to_read = max(max_chars * 4, 65536)
        if max_lines and max_lines > 0:
            bytes_to_read = max(bytes_to_read, max_lines * 512)

        if file_size > bytes_to_read:
            with open(p, "rb") as f:
                f.seek(file_size - bytes_to_read)
                raw = f.read()
                # Decode ignoring errors
                text = raw.decode("utf-8", errors="ignore")
                # Drop possible partial line at the start if seek was midway
                first_nl = text.find("\n")
                if first_nl != -1 and first_nl < 100:
                    text = text[first_nl + 1:]
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")

        if max_lines is not None and max_lines > 0:
            lines = text.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            text = "\n".join(lines)

        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            text = text[-max_chars:]

        return text
    except Exception:
        return ""


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Safely reads a CSV file into a pandas DataFrame.
    Returns an empty DataFrame if file is missing, empty, or corrupted.
    """
    if not path:
        return pd.DataFrame()

    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def read_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    """
    Safely reads a JSONL file into a pandas DataFrame.
    Skips invalid / corrupted lines and returns empty DataFrame if file is missing or empty.
    """
    if not path:
        return pd.DataFrame()

    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return pd.DataFrame()

        rows = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    rows.append(json.loads(line_str))
                except Exception:
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def combine_summaries(project_output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Combines all summary.csv files under a project directory into a unified DataFrame.
    """
    if not project_output_dir:
        return pd.DataFrame()

    try:
        project_dir = Path(project_output_dir)
        if not project_dir.exists() or not project_dir.is_dir():
            return pd.DataFrame()

        rows = []
        for summary_path in project_dir.rglob("summary.csv"):
            try:
                df = read_csv(summary_path)
                if not df.empty:
                    # Determine job name from relative path or parent dir
                    rel = summary_path.relative_to(project_dir)
                    parts = rel.parts
                    if len(parts) >= 2:
                        job_name = parts[0]
                    else:
                        job_name = summary_path.parent.name
                    df.insert(0, "job", job_name)
                    rows.append(df)
            except Exception:
                pass
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
