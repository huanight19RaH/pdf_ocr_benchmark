from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from runner.job_builder import (
    OUTPUTS_DIR,
    kernel_id,
    prepare_project_jobs,
    push_job,
    status_job,
    download_job_output,
)
from runner.kaggle_api import validate_account, write_access_token, write_kaggle_json
from runner.log_reader import combine_summaries, find_job_files, read_csv, read_jsonl, read_text_tail
from runner.project_config import (
    ROOT,
    accounts_by_id,
    load_accounts,
    load_projects,
    resolve_tool_path,
    save_accounts,
    save_projects,
)


st.set_page_config(page_title="Kaggle Runner UI", layout="wide")
st.title("Kaggle Runner UI")


def rerun():
    st.rerun()


accounts_cfg = load_accounts()
projects_cfg = load_projects()
accounts = accounts_by_id()
project_names = [p["name"] for p in projects_cfg.get("projects", [])]

tabs = st.tabs(["Accounts", "Projects", "Jobs", "Run", "Logs", "Results", "Git"])

with tabs[0]:
    st.subheader("Accounts")
    st.caption("Tokens stay local in this tool folder and are ignored by Git.")
    edited_accounts = st.data_editor(
        accounts_cfg.get("accounts", []),
        num_rows="dynamic",
        use_container_width=True,
        key="accounts_editor",
    )
    if st.button("Save accounts"):
        save_accounts({"accounts": edited_accounts})
        rerun()

    st.divider()
    selected_account_id = st.selectbox("Account", [a["id"] for a in accounts_cfg.get("accounts", [])])
    account = accounts.get(selected_account_id)
    if account:
        token_dir = resolve_tool_path(account["token_dir"])
        st.write("Token folder:", str(token_dir))
        col1, col2, col3 = st.columns(3)
        with col1:
            uploaded = st.file_uploader("Import kaggle.json", type=["json"], key="kaggle_json_upload")
            if uploaded and st.button("Save uploaded token"):
                token_dir.mkdir(parents=True, exist_ok=True)
                (token_dir / "kaggle.json").write_bytes(uploaded.getvalue())
                st.success("Saved kaggle.json")
        with col2:
            username = st.text_input("Username", value=account.get("username", ""))
            key = st.text_input("API key", type="password")
            if st.button("Save username + key"):
                write_kaggle_json(token_dir, username, key)
                st.success("Saved kaggle.json")
        with col3:
            access_token = st.text_area("Raw access_token", height=100)
            if st.button("Save access_token"):
                write_access_token(token_dir, access_token)
                st.success("Saved access_token")

        if st.button("Validate selected account"):
            result = validate_account({**account, "token_dir": str(token_dir)})
            st.json(result)

with tabs[1]:
    st.subheader("Projects")
    project_yaml = st.text_area(
        "projects.yaml",
        value=yaml.safe_dump(projects_cfg, sort_keys=False, allow_unicode=True),
        height=520,
    )
    if st.button("Save projects"):
        save_projects(yaml.safe_load(project_yaml))
        rerun()

with tabs[2]:
    st.subheader("Jobs")
    selected_project_name = st.selectbox("Project", project_names, key="jobs_project")
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == selected_project_name), None)
    if project:
        rows = []
        for job in project.get("jobs", []):
            account = accounts.get(job["account_id"], {})
            rows.append(
                {
                    "job": job["name"],
                    "account_id": job["account_id"],
                    "username": account.get("username"),
                    "kernel": kernel_id(account, job) if account else "",
                    "engines": " ".join(job.get("engines", [])),
                    "install_files": ", ".join(job.get("install_files", [])),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[3]:
    st.subheader("Run")
    selected_project_name = st.selectbox("Project", project_names, key="run_project")
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == selected_project_name), None)
    if project:
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Prepare"):
            prepared = prepare_project_jobs(project, accounts)
            st.success(f"Prepared {len(prepared)} jobs")
            for item in prepared:
                st.write(item["kernel_id"], str(item["job_dir"]))
        if col2.button("Push"):
            prepared = prepare_project_jobs(project, accounts)
            for item in prepared:
                st.write(f"Pushing {item['kernel_id']}")
                result = push_job(item["account"], item["job_dir"])
                st.code(result.stdout + result.stderr)
        if col3.button("Status"):
            for job in project.get("jobs", []):
                account = accounts[job["account_id"]]
                result = status_job(account, job)
                st.write(kernel_id(account, job))
                st.code(result.stdout + result.stderr)
        if col4.button("Download"):
            for job in project.get("jobs", []):
                account = accounts[job["account_id"]]
                out_dir, result = download_job_output(project["name"], account, job)
                st.write(kernel_id(account, job), str(out_dir))
                st.code(result.stdout + result.stderr)

with tabs[4]:
    st.subheader("Logs")
    selected_project_name = st.selectbox("Project", project_names, key="logs_project")
    project_output_dir = OUTPUTS_DIR / selected_project_name
    files = find_job_files(project_output_dir) if project_output_dir.exists() else {}
    for kind, paths in files.items():
        with st.expander(f"{kind} ({len(paths)})", expanded=kind == "logs"):
            for path in paths:
                st.write(str(path.relative_to(ROOT)))
                if kind == "logs":
                    st.code(read_text_tail(path))
                elif kind in {"summaries", "errors"}:
                    st.dataframe(read_csv(path), use_container_width=True)
                elif kind == "prefetch":
                    st.dataframe(read_jsonl(path), use_container_width=True)

with tabs[5]:
    st.subheader("Results")
    selected_project_name = st.selectbox("Project", project_names, key="results_project")
    combined = combine_summaries(OUTPUTS_DIR / selected_project_name)
    if combined.empty:
        st.info("No summary.csv files found yet.")
    else:
        st.dataframe(combined, use_container_width=True)
        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button("Download combined CSV", csv, file_name=f"{selected_project_name}_combined_summary.csv")

with tabs[6]:
    st.subheader("Git")
    import subprocess

    local_project = None
    if project_names:
        selected_project_name = st.selectbox("Project", project_names, key="git_project")
        local_project = next((p for p in projects_cfg.get("projects", []) if p["name"] == selected_project_name), None)
    if local_project:
        local_path = Path(local_project.get("local_path", "."))
        st.write("Local path:", str(local_path))
        if st.button("Git status"):
            result = subprocess.run(["git", "status", "--short"], cwd=local_path, capture_output=True, text=True)
            st.code(result.stdout + result.stderr)
        commit_message = st.text_input("Commit message", value="update Kaggle benchmark workflow")
        col1, col2 = st.columns(2)
        if col1.button("Commit"):
            subprocess.run(["git", "add", "."], cwd=local_path)
            result = subprocess.run(["git", "commit", "-m", commit_message], cwd=local_path, capture_output=True, text=True)
            st.code(result.stdout + result.stderr)
        if col2.button("Push"):
            result = subprocess.run(["git", "push"], cwd=local_path, capture_output=True, text=True)
            st.code(result.stdout + result.stderr)

