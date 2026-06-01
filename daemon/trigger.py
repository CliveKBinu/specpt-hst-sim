import os
import sys
import subprocess

def trigger(run_id, state):
    env = os.environ.copy()
    env["SPECPT_RUN_ID"] = run_id
    env["SPECPT_RUN_STATE"] = state

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [os.environ.get("OPENCODE_BIN", "opencode"), "--agent", "specpt-orchestrator"],
        cwd=repo,
        env=env,
    )
    print(f"Triggered orchestrator PID {proc.pid}")
    proc.wait()
    print(f"Orchestrator finished with code {proc.returncode}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python trigger.py <run_id> <state>")
        sys.exit(1)
    trigger(sys.argv[1], sys.argv[2])
