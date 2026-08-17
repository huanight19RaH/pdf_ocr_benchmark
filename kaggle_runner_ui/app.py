import os
import shutil
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from runner.ai_assistant import KaggleAIAssistant
from runner.job_builder import (
    OUTPUTS_DIR,
    WORK_DIR,
    download_job_output,
    kernel_id,
    prepare_project_jobs,
    push_job,
    run_parallel_tasks,
    status_job,
    stop_job,
)
from runner.kaggle_api import (
    get_account_gpu_status,
    read_token,
    run_kaggle,
    validate_account,
    write_access_token,
    write_kaggle_json,
)
from runner.log_reader import combine_summaries, find_job_files, read_csv, read_jsonl, read_text_tail
from runner.project_config import (
    ROOT,
    accounts_by_id,
    add_account,
    add_job_to_project,
    delete_account,
    delete_job_from_project,
    load_accounts,
    load_projects,
    resolve_tool_path,
    save_accounts,
    save_projects,
)

# Set page configuration
st.set_page_config(
    page_title="Kaggle Multi-Account & Multi-Thread OCR Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Aesthetic Dashboard
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }
    
    .account-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-active { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-running { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .badge-error { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "👋 **Xin chào! Tôi là Antigravity AI Assistant.**\n\n"
                "Tôi có thể hỗ trợ bạn điều khiển đa tài khoản Kaggle, kiểm tra dung lượng quota GPU, "
                "kích hoạt các luồng benchmark/finetune song song và chẩn đoán lỗi trong log."
            ),
        }
    ]

if "assistant" not in st.session_state:
    st.session_state.assistant = KaggleAIAssistant()

# Load Configs
accounts_cfg = load_accounts()
projects_cfg = load_projects()
accounts = accounts_by_id()
project_names = [p["name"] for p in projects_cfg.get("projects", [])]
default_project_name = project_names[0] if project_names else "pdf_ocr_benchmark"

# Sidebar Navigation
with st.sidebar:
    st.title("⚡ Kaggle Hub")
    st.caption("Multi-Account & Multi-Thread Orchestrator")
    st.divider()

    selected_project_name = st.selectbox("📂 Active Project", project_names, index=0 if project_names else None)
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == selected_project_name), None)

    st.divider()
    st.markdown("### 📊 System Overview")
    total_accounts = len(accounts_cfg.get("accounts", []))
    total_jobs = len(project.get("jobs", [])) if project else 0
    st.metric("Tài khoản Kaggle", f"{total_accounts} accounts")
    st.metric("Tổng số Luồng (Jobs)", f"{total_jobs} threads")

    st.divider()
    if st.button("🔄 Đồng bộ dữ liệu Local", use_container_width=True):
        st.session_state.assistant.refresh()
        st.rerun()

# Header Toolbar
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;">🚀 Kaggle Multi-Account Control Hub</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Quản lý tài khoản, kiểm soát GPU Quota, điều phối đa luồng & Trợ lý AI</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Tabs
tabs = st.tabs([
    "🚀 Dashboard & Quotas",
    "🧵 Multi-Thread Manager",
    "🤖 AI Assistant & Chat",
    "📊 Benchmark Analytics",
    "📜 Live Logs & Debug",
    "⚙️ Accounts & Git",
])

# ---------------------------------------------------------
# TAB 1: DASHBOARD & GPU QUOTAS
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("⚡ Tài khoản Kaggle & Quota GPU")
    st.caption("Theo dõi thời lượng GPU hàng tuần (30h limit) và số lượng batch GPU sessions đang chạy (tối đa 2 slots/account).")

    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 5])
    if col_btn1.button("🔍 Quét Quota tất cả Tài khoản", type="primary", use_container_width=True):
        st.session_state.quota_scanned = True

    acc_list = accounts_cfg.get("accounts", [])
    if not acc_list:
        st.warning("Chưa có tài khoản nào được thêm. Hãy mở tab 'Accounts & Git' để thêm tài khoản.")
    else:
        cols = st.columns(len(acc_list) if len(acc_list) <= 3 else 3)
        for idx, acc in enumerate(acc_list):
            with cols[idx % 3]:
                status = get_account_gpu_status(acc)
                valid = status.get("token_valid", False)
                active_s = status.get("active_sessions", 0)
                used_h = status.get("gpu_hours_used", 0.0)
                total_h = status.get("gpu_hours_total", 30.0)
                remain_h = status.get("gpu_hours_remaining", 30.0)
                pct = min(100.0, (used_h / total_h) * 100.0)

                badge_class = "badge-active" if valid else "badge-error"
                badge_text = "🟢 Token Hoạt động" if valid else "🔴 Token Lỗi / Thiếu"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <h3 style="margin:0; font-size: 1.1rem; font-weight: 600;">{acc['id']}</h3>
                            <span class="account-badge {badge_class}">{badge_text}</span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">
                            Username: <code style="color: #38bdf8;">{acc.get('username', 'N/A')}</code>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                                <span>Active Batch GPU:</span>
                                <b>{active_s} / 2 slots</b>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                                <span>GPU Quota còn lại:</span>
                                <b>{remain_h:.1f}h / {total_h:.0f}h</b>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(pct / 100.0)
                if status.get("running_kernels"):
                    st.caption(f"Đang chạy: {', '.join(status['running_kernels'])}")
                st.divider()

    st.markdown("### ⚡ Thao tác nhanh cho Project")
    if project:
        col_act1, col_act2, col_act3, col_act4 = st.columns(4)
        if col_act1.button("▶️ Chạy toàn bộ Luồng (Parallel)", use_container_width=True):
            with st.spinner("Đang chuẩn bị và đẩy tất cả các luồng lên Kaggle song song..."):
                prepared = prepare_project_jobs(project, accounts)
                push_tasks = [(push_job, (item["account"], item["job_dir"])) for item in prepared]
                results = run_parallel_tasks(push_tasks, max_workers=len(prepared) or 1)
                st.success(f"Đã kích hoạt {len(prepared)} luồng chạy song song trên các tài khoản!")
                st.rerun()

        if col_act2.button("🔄 Quét trạng thái các Luồng", use_container_width=True):
            st.rerun()

        if col_act3.button("📥 Tải toàn bộ Artifacts & Báo cáo", use_container_width=True):
            with st.spinner("Đang tải kết quả từ Kaggle..."):
                for job in project.get("jobs", []):
                    account = accounts.get(job["account_id"])
                    if account:
                        download_job_output(project["name"], account, job)
                st.success("Đã tải xong toàn bộ kết quả!")
                st.rerun()

        if col_act4.button("🧹 Xóa Cache & Dọn dẹp Work Dir", use_container_width=True):
            if WORK_DIR.exists():
                shutil.rmtree(WORK_DIR, ignore_errors=True)
            st.success("Đã dọn dẹp thư mục làm việc tạm thời.")

# ---------------------------------------------------------
# TAB 2: MULTI-THREAD MANAGER
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("🧵 Quản lý Đa luồng (Multi-Thread Jobs)")
    st.caption("Cấu hình, thêm mới, xóa và điều khiển từng luồng thực thi trên từng tài khoản Kaggle riêng biệt.")

    if not project:
        st.warning("Chưa có project nào được chọn.")
    else:
        # Action Toolbar
        col_t1, col_t2, col_t3, _ = st.columns([1.5, 1.5, 1.5, 4.5])
        if col_t1.button("➕ Thêm Luồng Mới", use_container_width=True):
            st.session_state.show_add_thread = not st.session_state.get("show_add_thread", False)

        # Expandable Add Thread Form
        if st.session_state.get("show_add_thread", False):
            with st.expander("📝 Cấu hình Luồng Mới (New Thread)", expanded=True):
                with st.form("add_job_form"):
                    f_name = st.text_input("Tên Luồng (Job Name)", placeholder="finetune-paddleocr, surya-exp, docling-test")
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        f_account = st.selectbox("Gán Tài khoản (Account)", [a["id"] for a in accounts_cfg.get("accounts", [])])
                    with col_f2:
                        f_type = st.selectbox("Loại Luồng (Job Type)", ["benchmark", "finetune"])
                    with col_f3:
                        f_gpu = st.selectbox("Phần cứng tăng tốc", ["NvidiaTeslaT4", "NvidiaTeslaP100", "TPU"])

                    col_f4, col_f5 = st.columns(2)
                    with col_f4:
                        f_engines = st.multiselect(
                            "OCR Engines",
                            ["paddleocr", "docling", "surya", "paddleocr_vl", "paddleocr_ft"],
                            default=["paddleocr"],
                        )
                    with col_f5:
                        f_reqs = st.text_input("Dependency Files", value="requirements-kaggle-paddleocr.txt")

                    f_epochs = 10
                    f_ft_model = "paddleocr"
                    if f_type == "finetune":
                        col_ft1, col_ft2 = st.columns(2)
                        with col_ft1:
                            f_ft_model = st.selectbox("Finetune Model Target", ["paddleocr"])
                        with col_ft2:
                            f_epochs = st.number_input("Số Epochs", min_value=1, max_value=100, value=10)

                    if st.form_submit_button("Lưu Luồng"):
                        if not f_name:
                            st.error("Tên luồng không được để trống.")
                        else:
                            new_job = {
                                "name": f_name.strip(),
                                "account_id": f_account,
                                "job_type": f_type,
                                "machine_shape": f_gpu,
                                "engines": f_engines,
                                "install_files": [f.strip() for f in f_reqs.split(",") if f.strip()],
                            }
                            if f_type == "finetune":
                                new_job["finetune_model"] = f_ft_model
                                new_job["epochs"] = f_epochs

                            ok = add_job_to_project(project["name"], new_job)
                            if ok:
                                st.success(f"Đã thêm luồng `{f_name}` thành công!")
                                st.session_state.show_add_thread = False
                                st.rerun()
                            else:
                                st.error("Luồng với tên này đã tồn tại trong project.")

        # Jobs Table / Control Matrix
        jobs = project.get("jobs", [])
        if not jobs:
            st.info("Chưa có luồng nào được cấu hình trong project này.")
        else:
            st.markdown("### 📋 Danh sách các Luồng đang quản lý")
            for job in jobs:
                acc = accounts.get(job["account_id"], {})
                k_id = kernel_id(acc, job) if acc else "N/A"

                with st.container():
                    col_j1, col_j2, col_j3, col_j4, col_j5, col_j6 = st.columns([2.2, 1.8, 1.8, 1.2, 1.2, 1.2])
                    with col_j1:
                        st.markdown(f"**`{job['name']}`**")
                        st.caption(f"Engines: {', '.join(job.get('engines', []))}")
                    with col_j2:
                        st.markdown(f"Acc: `{job['account_id']}` ({acc.get('username', 'N/A')})")
                        st.caption(f"Kernel: `{k_id}`")
                    with col_j3:
                        res = status_job(acc, job) if acc else None
                        st_text = (res.stdout + res.stderr).strip() if res else "Unknown"
                        if "COMPLETE" in st_text.upper():
                            st.markdown("🟢 `COMPLETE`")
                        elif "RUNNING" in st_text.upper():
                            st.markdown("🔵 `RUNNING`")
                        elif "QUEUED" in st_text.upper():
                            st.markdown("🟡 `QUEUED`")
                        elif "ERROR" in st_text.upper() or "FAILED" in st_text.upper():
                            st.markdown("🔴 `ERROR`")
                        else:
                            st.markdown("⚪ `IDLE`")
                    with col_j4:
                        if st.button("▶️ Chạy", key=f"run_{job['name']}", use_container_width=True):
                            prepared = prepare_project_jobs(project, accounts)
                            item = next((it for it in prepared if it["job"]["name"] == job["name"]), None)
                            if item:
                                push_res = push_job(item["account"], item["job_dir"])
                                st.code(push_res.stdout + push_res.stderr)
                                st.rerun()
                    with col_j5:
                        if st.button("📥 Tải", key=f"dl_{job['name']}", use_container_width=True):
                            if acc:
                                out_dir, dl_res = download_job_output(project["name"], acc, job)
                                st.success(f"Tải thành công về `{out_dir.name}`")
                                st.rerun()
                    with col_j6:
                        if st.button("🗑️ Xóa", key=f"del_{job['name']}", use_container_width=True):
                            delete_job_from_project(project["name"], job["name"])
                            st.success(f"Đã xóa luồng `{job['name']}`")
                            st.rerun()
                    st.divider()

# ---------------------------------------------------------
# TAB 3: AI ASSISTANT & CHATBOT
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("🤖 Antigravity AI Companion")
    st.caption("Giao tiếp bằng ngôn ngữ tự nhiên để điều khiển các tài khoản, kiểm tra tiến độ, bắt lỗi và xuất báo cáo.")

    # Suggested Prompts
    st.markdown("**💡 Gợi ý câu lệnh mẫu:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    if col_p1.button("🔍 Quota GPU các tài khoản", use_container_width=True):
        st.session_state.pending_prompt = "Kiểm tra quota GPU của các tài khoản"
    if col_p2.button("📊 Trạng thái các luồng", use_container_width=True):
        st.session_state.pending_prompt = "Kiểm tra trạng thái các luồng"
    if col_p3.button("⚡ Chạy tất cả các luồng", use_container_width=True):
        st.session_state.pending_prompt = "Chạy tất cả các luồng"
    if col_p4.button("🔍 Phân tích lỗi log", use_container_width=True):
        st.session_state.pending_prompt = "Phân tích lỗi gần nhất trong log"

    st.divider()

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    user_input = st.chat_input("Nhập yêu cầu điều khiển hoặc câu hỏi cho AI Assistant...")
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Antigravity đang xử lý yêu cầu..."):
                response_data = st.session_state.assistant.process_message(
                    user_input,
                    project_name=project["name"] if project else None,
                )
                reply_text = response_data.get("response", "")
                st.markdown(reply_text)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

                # Handle action triggers
                if response_data.get("action_trigger") == "run_jobs" and project:
                    target_names = response_data.get("target_job_names", [])
                    if st.button("🚀 Thực thi ngay các luồng này", key="trigger_run_btn"):
                        prepared = prepare_project_jobs(project, accounts)
                        filtered = [it for it in prepared if it["job"]["name"] in target_names]
                        push_tasks = [(push_job, (it["account"], it["job_dir"])) for it in filtered]
                        run_parallel_tasks(push_tasks, max_workers=len(filtered) or 1)
                        st.success(f"Đã kích hoạt {len(filtered)} luồng thành công!")
                        st.rerun()

# ---------------------------------------------------------
# TAB 4: BENCHMARK ANALYTICS
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("📊 Phân tích Kết quả Benchmark & Leaderboard")
    st.caption("So sánh đối soát trực tiếp giữa các mô hình OCR và hiệu quả của việc Finetune trên tài liệu khoa học DocLayNet.")

    if project:
        results_dir = OUTPUTS_DIR / project["name"]
        combined = combine_summaries(results_dir) if results_dir.exists() else None

        if combined is None or combined.empty:
            st.info("Chưa tìm thấy file `summary.csv` tổng hợp nào. Hãy chuyển sang tab 'Run' hoặc 'Dashboard' và nhấn 'Download All'.")
        else:
            st.dataframe(combined, use_container_width=True)

            # Interactive Plotly Charts
            col_ch1, col_ch2 = st.columns(2)
            if "cer" in combined.columns and "wer" in combined.columns and "engine" in combined.columns:
                with col_ch1:
                    st.markdown("#### 🎯 So sánh Tỷ lệ Lỗi (CER & WER - Càng thấp càng tốt)")
                    fig_err = px.bar(
                        combined,
                        x="engine",
                        y=["cer", "wer"],
                        barmode="group",
                        title="Character Error Rate (CER) & Word Error Rate (WER)",
                        color_discrete_sequence=["#38bdf8", "#818cf8"],
                    )
                    fig_err.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_err, use_container_width=True)

            if "latency_s" in combined.columns and "chars_per_second" in combined.columns and "engine" in combined.columns:
                with col_ch2:
                    st.markdown("#### ⚡ So sánh Tốc độ Xử lý (Chars/s & Latency)")
                    fig_speed = px.bar(
                        combined,
                        x="engine",
                        y="chars_per_second",
                        color="engine",
                        title="Thông lượng nhận diện (Ký tự / Giây - Càng cao càng tốt)",
                    )
                    fig_speed.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_speed, use_container_width=True)

            # Download CSV Button
            csv_data = combined.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Tải Bảng Tổng Hợp Kết Quả (CSV)",
                csv_data,
                file_name=f"{project['name']}_benchmark_summary.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------
# TAB 5: LIVE LOGS & DEBUG
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("📜 Nhật ký Thực thi (Live Logs & Debug)")
    st.caption("Xem log chi tiết thời gian thực từ các máy chủ Kaggle GPU và phát hiện nguyên nhân lỗi tự động.")

    if project:
        project_output_dir = OUTPUTS_DIR / project["name"]
        files = find_job_files(project_output_dir) if project_output_dir.exists() else {}

        log_paths = files.get("logs", [])
        if not log_paths:
            st.info("Chưa có file log tải về local. Nhấn 'Download' ở Dashboard để đồng bộ.")
        else:
            col_l1, col_l2 = st.columns([3, 1])
            with col_l1:
                selected_log = st.selectbox("Chọn File Log để kiểm tra", log_paths, format_func=lambda p: p.name)
            with col_l2:
                tail_n = st.selectbox("Số dòng cuối", [50, 100, 200, 500], index=1)

            if selected_log:
                log_text = read_text_tail(selected_log, max_lines=tail_n)
                search_term = st.text_input("🔍 Tìm kiếm từ khóa trong log", placeholder="Traceback, Error, Epoch, Loss...")

                if search_term:
                    matched_lines = [l for l in log_text.splitlines() if search_term.lower() in l.lower()]
                    st.markdown(f"**Tìm thấy {len(matched_lines)} dòng khớp:**")
                    st.code("\n".join(matched_lines) if matched_lines else "Không có dòng nào khớp từ khóa.")
                else:
                    st.code(log_text, language="text")

# ---------------------------------------------------------
# TAB 6: ACCOUNTS & GIT
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("⚙️ Quản lý Tài khoản & Git Sync")
    st.caption("Thêm, sửa, xóa token Kaggle và đồng bộ Git repository.")

    col_acc_edit, col_acc_token = st.columns(2)
    with col_acc_edit:
        st.markdown("#### ➕ Thêm / Quản lý Tài khoản Kaggle")
        with st.form("new_acc_form"):
            new_id = st.text_input("Account ID", placeholder="account4, my_kaggle_2")
            new_user = st.text_input("Kaggle Username", placeholder="rah1111, hungnguyen")
            if st.form_submit_button("Thêm Tài khoản"):
                if new_id and new_user:
                    add_account(new_id.strip(), new_user.strip())
                    st.success(f"Đã thêm tài khoản `{new_id}`!")
                    st.rerun()
                else:
                    st.error("Vui lòng điền đủ ID và Username.")

        st.divider()
        st.markdown("#### 🗑️ Xóa Tài khoản")
        del_acc_id = st.selectbox("Chọn tài khoản muốn xóa", [a["id"] for a in accounts_cfg.get("accounts", [])])
        if st.button("Xóa tài khoản này", type="secondary"):
            delete_account(del_acc_id)
            st.success(f"Đã xóa tài khoản `{del_acc_id}`.")
            st.rerun()

    with col_acc_token:
        st.markdown("#### 🔑 Cập nhật API Token cho Tài khoản")
        sel_acc_id = st.selectbox("Chọn tài khoản cập nhật", [a["id"] for a in accounts_cfg.get("accounts", [])], key="token_sel_acc")
        sel_acc = accounts.get(sel_acc_id)
        if sel_acc:
            t_dir = resolve_tool_path(sel_acc["token_dir"])
            st.caption(f"Thư mục lưu token: `{t_dir}`")

            up_file = st.file_uploader("Upload kaggle.json", type=["json"], key="up_kaggle_json")
            if up_file and st.button("Lưu file kaggle.json vừa upload"):
                t_dir.mkdir(parents=True, exist_ok=True)
                (t_dir / "kaggle.json").write_bytes(up_file.getvalue())
                st.success("Đã lưu kaggle.json!")
                st.rerun()

            p_key = st.text_input("Hoặc Dán API Key / Token", type="password")
            if st.button("Lưu API Token trực tiếp") and p_key:
                write_kaggle_json(t_dir, sel_acc.get("username", "user"), p_key)
                st.success("Đã cập nhật Token!")
                st.rerun()

            if st.button("🔍 Kiểm tra tính hợp lệ của Token này"):
                v_res = validate_account({**sel_acc, "token_dir": str(t_dir)})
                st.json(v_res)

    st.divider()
    st.markdown("### 🐙 Git Synchronization")
    import subprocess
    if st.button("Git Status"):
        g_res = subprocess.run(["git", "status", "--short"], cwd=str(ROOT.parent), capture_output=True, text=True)
        st.code(g_res.stdout + g_res.stderr)
