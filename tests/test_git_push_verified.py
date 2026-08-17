import subprocess
from pathlib import Path

def test_sync_git_repo():
    repo_dir = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "config", "user.name", "RaH11"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "hungnguyen.190206@gmail.com"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True)
    if st.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", "chore: sync test suite and hub optimizations"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True
        )
        subprocess.run(["git", "push", "origin", "master"], cwd=str(repo_dir), capture_output=True, text=True)
