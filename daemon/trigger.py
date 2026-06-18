import os
import sys
import subprocess
import argparse
import time
import json
import threading

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DEAD_LETTER_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "daemon", ".dead_letter.jsonl",
)

HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    "C:\\Users\\clive\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe",
)
WORKING_MODEL = os.environ.get("SPECPT_MODEL", "opencode-go/deepseek-v4-flash")
DELEGATE_MODEL = os.environ.get("SPECPT_DELEGATE_MODEL", "opencode-go/deepseek-v4-pro")


def append_dead_letter(run_id, run_name, error, timestamp):
    entry = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "error": error,
    }
    try:
        os.makedirs(os.path.dirname(DEAD_LETTER_FILE), exist_ok=True)
        with open(DEAD_LETTER_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARN] Could not write dead-letter: {e}")


def fetch_run_name(run_id):
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"ckb2084-rochester-institute-of-technology/specpt-hst-sim/{run_id}")
        return run.name
    except Exception as e:
        print(f"[WARN] Could not fetch run name from W&B: {e}")
        return None


def build_prompt(run_id, run_name, state):
    return f"""You are the SpecPT-HST-Sim orchestrator. Process this W&B run:
- Run ID: {run_id}
- Run name: {run_name}
- State: {state}

Workflow:
1. Read .hermes/SOUL.md, EXPERIMENTS.md, jobs.csv for current state
2. Follow .hermes/AGENTS.md (analyst -> experimenter -> runner -> memory)
3. Use delegate_task with model {DELEGATE_MODEL} for analyst and experimenter subagents
4. Execute the full chain and exit

Do not ask for confirmation. Execute autonomously."""


def detect_runner_marker(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if "[[RUNNER_SUCCEEDED]]" in content:
            return "succeeded"
        if "[[RUNNER_FAILED]]" in content:
            return "failed"
    except Exception:
        pass
    return None


def tee_output(stream, log_file):
    for line in iter(stream.readline, ""):
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.flush()
        log_file.write(line)
    stream.close()


def trigger(run_id, state, run_name=None, timeout=1800):
    valid_states = {"finished", "crashed", "failed"}
    if state not in valid_states:
        print(f"[ERROR] Invalid state '{state}'. Must be one of: {valid_states}")
        sys.exit(2)

    env = os.environ.copy()
    env["SPECPT_RUN_ID"] = run_id
    env["SPECPT_RUN_STATE"] = state
    if not run_name:
        run_name = fetch_run_name(run_id) or run_id
    env["SPECPT_RUN_NAME"] = run_name
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LANG"] = "C.UTF-8"

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt = build_prompt(run_id, run_name, state)

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"orchestrator_{run_id}_{timestamp}.log")

    print(f"[RUN] {run_name}  state={state}  run_id={run_id}")
    print(f"[CMD] hermes chat -q (cwd={repo}, timeout={timeout}s)")
    print(f"[LOG] {log_path}")

    proc = subprocess.Popen(
        [HERMES_BIN, "chat", "-q", prompt,
         "--provider", "opencode-go",
         "--model", WORKING_MODEL,
         "--profile", "specpt-hst",
         "--skills", "specpt-orchestrator",
         "--accept-hooks",
         "--quiet"],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    print(f"Triggered orchestrator PID {proc.pid}")

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            tee = threading.Thread(target=tee_output, args=(proc.stdout, log_file), daemon=True)
            tee.start()
            proc.wait(timeout=timeout)
            tee.join(timeout=10)
        print(f"Orchestrator finished with code {proc.returncode}")

        marker = detect_runner_marker(log_path)
        if marker == "failed":
            print("[[RUNNER_FAILED]] — see orchestrator log for details")
        elif marker == "succeeded":
            print("[[RUNNER_SUCCEEDED]]")

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"[ERROR] Orchestrator timed out after {timeout}s — killed")
        append_dead_letter(run_id, run_name, "timed_out", time.time())
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger specpt-orchestrator manually")
    parser.add_argument("run_id", help="W&B run ID")
    parser.add_argument("state", help="Run state (finished/crashed/failed)")
    parser.add_argument("--name", "-n", help="Run name (fetched from W&B if omitted)")
    parser.add_argument("--timeout", "-t", type=int, default=1800,
                        help="Timeout in seconds (default: 1800)")
    args = parser.parse_args()
    trigger(args.run_id, args.state, args.name, args.timeout)
