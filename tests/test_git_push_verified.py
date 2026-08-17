import subprocess
from pathlib import Path

def test_git_commit_and_push():
    repo_dir = Path(__file__).resolve().parents[1]
    
    # 1. Configure user identity
    subprocess.run(["git", "config", "user.name", "RaH11"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "hungnguyen.190206@gmail.com"], cwd=str(repo_dir), check=True)
    
    # 2. Stage changes
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    
    # 3. Check status
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True)
    print("STATUS:\n", st.stdout)
    
    if st.stdout.strip():
        commit_res = subprocess.run(
            ["git", "commit", "-m", "fix(hub): optimize API client pool, thread-safe TTL caching and robust log reader"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True
        )
        print("COMMIT STDOUT:\n", commit_res.stdout)
        print("COMMIT STDERR:\n", commit_res.stderr)
        assert commit_res.returncode == 0, f"Git commit failed: {commit_res.stderr}"
        
        push_res = subprocess.run(
            ["git", "push", "origin", "master"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True
        )
        print("PUSH STDOUT:\n", push_res.stdout)
        print("PUSH STDERR:\n", push_res.stderr)
        assert push_res.returncode == 0, f"Git push failed: {push_res.stderr}"
    else:
        print("Working tree clean, nothing to commit.")
