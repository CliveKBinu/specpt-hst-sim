import os
import sys
import subprocess
import argparse


HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    "C:\\Users\\clive\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe",
)
WORKING_MODEL = os.environ.get("SPECPT_MODEL", "opencode-go/deepseek-v4-flash")
DELEGATE_MODEL = os.environ.get("SPECPT_DELEGATE_MODEL", "opencode-go/deepseek-v4-pro")


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

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt = build_prompt(run_id, run_name, state)
    print(f"[RUN] {run_name}  state={state}  run_id={run_id}")
    print(f"[CMD] hermes chat -q (cwd={repo}, timeout={timeout}s)")

    proc = subprocess.Popen(
        [HERMES_BIN, "chat", "-q", prompt,
         "--provider", "opencode-go",
         "--model", WORKING_MODEL,
         "--accept-hooks",
         "--quiet"],
        cwd=repo,
        env=env,
        stdout=None,
        stderr=None,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    print(f"Triggered orchestrator PID {proc.pid}")
    try:
        proc.wait(timeout=timeout)
        print(f"Orchestrator finished with code {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"[ERROR] Orchestrator timed out after {timeout}s — killed")
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
