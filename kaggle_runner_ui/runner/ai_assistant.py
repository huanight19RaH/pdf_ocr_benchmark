import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .job_builder import OUTPUTS_DIR, download_job_output, kernel_id, prepare_project_jobs, push_job, status_job
from .kaggle_api import get_account_gpu_status, validate_account
from .log_reader import combine_summaries, find_job_files, read_csv, read_text_tail
from .project_config import ROOT, accounts_by_id, load_accounts, load_projects


class KaggleAIAssistant:
    """
    Antigravity AI Assistant Companion for Kaggle Multi-Account & Multi-Job Management.
    Processes natural language commands, automates dispatching, monitors GPU usage,
    and analyzes OCR benchmark logs.
    """

    def __init__(self):
        self.accounts_cfg = load_accounts()
        self.projects_cfg = load_projects()
        self.accounts = accounts_by_id()

    def refresh(self):
        self.accounts_cfg = load_accounts()
        self.projects_cfg = load_projects()
        self.accounts = accounts_by_id()

    def process_message(self, user_prompt: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        self.refresh()
        prompt_lower = user_prompt.lower().strip()
        project = None
        projects = self.projects_cfg.get("projects", [])
        if project_name:
            project = next((p for p in projects if p["name"] == project_name), None)
        if not project and projects:
            project = projects[0]

        # 1. Quota / GPU check
        if any(k in prompt_lower for k in ["gpu", "quota", "dung lượng", "thời hạn", "slot", "tài khoản", "account"]):
            return self._handle_quota_query()

        # 2. Status check
        if any(k in prompt_lower for k in ["trạng thái", "status", "tiến độ", "đang chạy", "xem luồng"]):
            return self._handle_status_query(project)

        # 3. Download results / Report
        if any(k in prompt_lower for k in ["tải kết quả", "download", "lấy kết quả", "báo cáo", "report", "tổng hợp"]):
            return self._handle_download_query(project)

        # 4. Error / Log Analysis
        if any(k in prompt_lower for k in ["lỗi", "error", "fail", "debug", "tại sao", "log", "traceback"]):
            return self._handle_error_diagnosis(project)

        # 5. Run commands
        if any(k in prompt_lower for k in ["chạy", "run", "start", "huấn luyện", "finetune", "bắt đầu"]):
            return self._handle_run_query(prompt_lower, project)

        # 6. Default helpful response with suggestions
        return {
            "response": (
                f"👋 **Xin chào! Tôi là AI Assistant điều khiển Kaggle Multi-Account.**\n\n"
                f"Tôi có thể giúp bạn tự động hóa hoàn toàn các tác vụ sau:\n"
                f"- 📊 **Kiểm tra GPU Quota & Active Slots:** Hỏi *'Kiểm tra quota GPU của các tài khoản'*.\n"
                f"- 🚀 **Điều khiển đa luồng:** Hỏi *'Chạy tất cả các luồng'* hoặc *'Chạy job paddleocr trên acc2'*.\n"
                f"- 📈 **Giám sát tiến độ:** Hỏi *'Kiểm tra trạng thái các luồng đang chạy'*.\n"
                f"- 📥 **Tải & Lập báo cáo:** Hỏi *'Tải kết quả và tổng hợp báo cáo'*.\n"
                f"- 🔍 **Bắt lỗi & Phân tích log:** Hỏi *'Phân tích lỗi gần nhất trong log'*.\n\n"
                f"Bạn muốn thực hiện thao tác nào ngay bây giờ?"
            ),
            "actions": [
                {"label": "🔍 Kiểm tra Quota GPU", "command": "Kiểm tra quota GPU các tài khoản"},
                {"label": "📊 Xem trạng thái luồng", "command": "Kiểm tra trạng thái các luồng"},
                {"label": "⚡ Chạy tất cả luồng", "command": "Chạy tất cả các luồng"},
                {"label": "📥 Tải kết quả & Báo cáo", "command": "Tải kết quả và tổng hợp báo cáo"},
            ],
        }

    def _handle_quota_query(self) -> Dict[str, Any]:
        accounts_list = self.accounts_cfg.get("accounts", [])
        if not accounts_list:
            return {"response": "⚠️ Chưa có tài khoản Kaggle nào được cấu hình trong hệ thống."}

        lines = ["### 📊 Báo cáo Dung lượng & GPU Quota của các Tài khoản Kaggle\n"]
        lines.append("| Tài khoản | Username | Active GPU Slots | GPU Quota Tuần | Trạng thái Token |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        total_active = 0
        for acc in accounts_list:
            status = get_account_gpu_status(acc)
            active_slots = f"{status.get('active_sessions', 0)} / {status.get('max_sessions', 2)} slots"
            total_active += status.get("active_sessions", 0)
            hours_str = f"{status.get('gpu_hours_used', 0.0):.1f}h / {status.get('gpu_hours_total', 30.0):.0f}h"
            token_valid = "✅ Hoạt động" if status.get("token_valid") else "❌ Cần kiểm tra"
            lines.append(f"| **{acc['id']}** | `{acc.get('username', 'N/A')}` | `{active_slots}` | `{hours_str}` | {token_valid} |")

        lines.append(f"\n> **Tổng quan:** Đang có **{total_active} luồng GPU đang hoạt động** trên tổng số {len(accounts_list) * 2} slots tối đa.")
        lines.append("> *Lưu ý: Mỗi tài khoản Kaggle có hạn mức 30 giờ GPU/tuần và tối đa 2 batch GPU session chạy đồng thời.*")

        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "🚀 Xem các luồng đang chạy", "command": "Kiểm tra trạng thái các luồng"},
                {"label": "➕ Thêm tài khoản mới", "command": "Mở tab Accounts để thêm tài khoản"},
            ],
        }

    def _handle_status_query(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "⚠️ Không tìm thấy project cấu hình nào."}

        jobs = project.get("jobs", [])
        if not jobs:
            return {"response": "⚠️ Project hiện tại chưa có job/luồng nào."}

        lines = [f"### 🧵 Trạng thái các luồng chạy cho Project `{project['name']}`\n"]
        lines.append("| Luồng (Job) | Account | Kernel ID | Trạng thái | Mô hình / Engines |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for job in jobs:
            account = self.accounts.get(job["account_id"])
            if not account:
                lines.append(f"| **{job['name']}** | `{job['account_id']}` | N/A | ⚠️ Thiếu Account | {', '.join(job.get('engines', []))} |")
                continue

            k_id = kernel_id(account, job)
            res = status_job(account, job)
            status_text = (res.stdout + res.stderr).strip()
            badge = "⚪ Không xác định"
            if "COMPLETE" in status_text.upper():
                badge = "🟢 **COMPLETE**"
            elif "RUNNING" in status_text.upper():
                badge = "🔵 **RUNNING** (Đang chạy)"
            elif "QUEUED" in status_text.upper():
                badge = "🟡 **QUEUED** (Đang chờ)"
            elif "ERROR" in status_text.upper() or "FAILED" in status_text.upper():
                badge = "🔴 **ERROR**"

            lines.append(f"| **{job['name']}** | `{account['username']}` | `{k_id}` | {badge} | {', '.join(job.get('engines', []))} |")

        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "📥 Tải kết quả các luồng xong", "command": "Tải kết quả và tổng hợp báo cáo"},
                {"label": "🔍 Xem chi tiết Log lỗi", "command": "Phân tích lỗi gần nhất trong log"},
            ],
        }

    def _handle_download_query(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "⚠️ Không tìm thấy project cấu hình."}

        results_dir = OUTPUTS_DIR / project["name"]
        combined = combine_summaries(results_dir) if results_dir.exists() else None

        if combined is None or combined.empty:
            return {
                "response": (
                    f"📥 **Đang chuẩn bị tải kết quả cho project `{project['name']}`...**\n\n"
                    f"Hiện tại chưa thấy file tổng hợp `summary.csv` trong thư mục local outputs. "
                    f"Bạn có thể nhấn nút **Download All** ở tab Run để tự động kéo tất cả artifact từ Kaggle về máy."
                ),
                "actions": [{"label": "📥 Kích hoạt Download Ngay", "command": "Tải kết quả ngay"}],
            }

        lines = [f"### 🏆 Bảng tổng hợp Benchmark mới nhất (`{project['name']}`)\n"]
        lines.append(combined.to_markdown(index=False))
        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "📈 Xem biểu đồ so sánh", "command": "Mở tab Results & Charts"},
                {"label": "🔍 Kiểm tra log chi tiết", "command": "Phân tích log"},
            ],
        }

    def _handle_error_diagnosis(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "⚠️ Không tìm thấy project."}

        output_dir = OUTPUTS_DIR / project["name"]
        if not output_dir.exists():
            return {"response": "ℹ️ Chưa có file log tải về local để chẩn đoán. Hãy nhấn Download để đồng bộ log từ Kaggle."}

        files = find_job_files(output_dir)
        log_files = files.get("logs", [])
        if not log_files:
            return {"response": "ℹ️ Không tìm thấy file `.log` nào trong thư mục outputs của project."}

        diagnoses = []
        for log_path in log_files:
            content = read_text_tail(log_path, max_lines=80)
            if "Traceback" in content or "Error" in content or "FAILED" in content:
                job_name = log_path.parent.name if log_path.parent.name != project["name"] else log_path.stem
                error_match = re.findall(r"(?:[A-Za-z]+Error:.*|Traceback.*)", content, re.MULTILINE)
                summary_err = "\n".join(error_match[-3:]) if error_match else "Gặp ngoại lệ trong quá trình chạy."
                diagnoses.append(f"#### 🔴 Luồng `{job_name}` (`{log_path.name}`)\n```text\n{summary_err}\n```")

        if not diagnoses:
            return {
                "response": "🎉 **Tuyệt vời! Không phát hiện lỗi nghiêm trọng nào trong các file log gần nhất.** Mọi luồng đều chạy thành công hoặc đang thực thi bình thường."
            }

        response_text = (
            "### 🔍 Kết quả phân tích & Chẩn đoán Lỗi từ Log Kaggle:\n\n"
            + "\n\n".join(diagnoses)
            + "\n\n💡 **Khuyến nghị khắc phục:**\n"
            + "1. Kiểm tra lại các file dependency `requirements-kaggle-*.txt` tương ứng.\n"
            + "2. Sử dụng tab Multi-Thread Manager để chỉnh sửa và chạy lại luồng bị lỗi sau khi cập nhật mã nguồn."
        )

        return {
            "response": response_text,
            "actions": [
                {"label": "⚡ Chạy lại các luồng lỗi", "command": "Chạy lại luồng bị lỗi"},
                {"label": "📜 Mở toàn bộ file Log", "command": "Mở tab Logs"},
            ],
        }

    def _handle_run_query(self, prompt_lower: str, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "⚠️ Không tìm thấy project cấu hình."}

        jobs = project.get("jobs", [])
        target_jobs = []
        for job in jobs:
            if job["name"].lower() in prompt_lower:
                target_jobs.append(job)

        if not target_jobs and ("tất cả" in prompt_lower or "all" in prompt_lower):
            target_jobs = jobs

        if not target_jobs:
            job_names = ", ".join([f"`{j['name']}`" for j in jobs])
            return {
                "response": (
                    f"Bạn muốn chạy luồng nào trong số các luồng hiện có: {job_names}?\n"
                    f"Hoặc bạn có thể yêu cầu: *'Chạy tất cả các luồng'*."
                ),
                "actions": [{"label": "⚡ Chạy tất cả các luồng", "command": "Chạy tất cả các luồng"}],
            }

        job_list_str = ", ".join([f"**{j['name']}** (Account: `{j['account_id']}`)" for j in target_jobs])
        return {
            "response": (
                f"🚀 **Đã sẵn sàng kích hoạt {len(target_jobs)} luồng chạy:**\n"
                f"{job_list_str}\n\n"
                f"Bạn có thể nhấn nút **Execute Selected Threads** bên dưới hoặc qua tab **Multi-Thread Manager** để kích hoạt chạy song song đa luồng."
            ),
            "action_trigger": "run_jobs",
            "target_job_names": [j["name"] for j in target_jobs],
            "actions": [
                {"label": f"▶️ Chạy ngay {len(target_jobs)} luồng", "command": f"Xác nhận chạy {len(target_jobs)} luồng"},
                {"label": "🔍 Xem lại Quota trước khi chạy", "command": "Kiểm tra quota GPU các tài khoản"},
            ],
        }
