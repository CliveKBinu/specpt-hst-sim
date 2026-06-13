import os
import sys
import json
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ckb2084-rochester-institute-of-technology")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "specpt-hst-sim")
CHECK_INTERVAL = 60
STATE_FILE = os.path.join(os.path.dirname(__file__), ".watcher_state")
LOCK_FILE = os.path.join(os.path.dirname(__file__), ".orchestrator.lock")
MAX_RUN_AGE = 7 * 24 * 3600


def load_last_check():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Corrupt state file, starting fresh")
            return {"last_check": time.time(), "processed_runs": []}
        if "processed_runs" not in state:
            state["processed_runs"] = []
        return state
    return {"last_check": time.time(), "processed_runs": []}


def save_last_check(ts, processed_runs):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_check": ts, "processed_runs": list(processed_runs)}, f)


def get_run_time(run):
    created = getattr(run, "created_at", None)
    if hasattr(created, "timestamp"):
        return created.timestamp()
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return float("-inf")
    return float("-inf")


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return False
        except (OSError, ValueError):
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


def check_wandb():
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                        order="-created_at", per_page=500)
        state = load_last_check()
        processed = set(state.get("processed_runs", []))
        last_check = state["last_check"]
        cutoff = max(last_check, time.time() - MAX_RUN_AGE)

        unprocessed = []
        for run in runs:
            if run.state not in ("finished", "crashed", "failed"):
                continue
            if run.id in processed:
                continue
            rt = get_run_time(run)
            if rt < cutoff:
                continue
            unprocessed.append((rt, run))

        if not unprocessed:
            save_last_check(time.time(), processed)
            return

        unprocessed.sort(key=lambda x: x[0], reverse=True)
        mode = os.environ.get("WATCHER_MODE", "").lower()
        selected = unprocessed if mode == "all" else [unprocessed[0]]

        for rt, run in selected:
            if not acquire_lock():
                logger.warning(f"Orchestrator already running, skipping {run.name}")
                continue
            try:
                logger.info(f"Found terminal run: {run.name} ({run.state})")
                if trigger_orchestrator(run.id, run.name, run.state):
                    processed.add(run.id)
                    save_last_check(time.time(), processed)
                else:
                    logger.error(f"Failed to trigger orchestrator for {run.id}")
            finally:
                release_lock()
    except Exception as e:
        logger.error(f"W&B check failed: {e}")


HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    "C:\\Users\\clive\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe",
)
WORKING_MODEL = os.environ.get("SPECPT_MODEL", "opencode-go/deepseek-v4-flash")
DELEGATE_MODEL = os.environ.get("SPECPT_DELEGATE_MODEL", "opencode-go/deepseek-v4-pro")


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


def trigger_orchestrator(run_id, run_name, state):
    env = os.environ.copy()
    env["SPECPT_RUN_ID"] = run_id
    env["SPECPT_RUN_NAME"] = run_name
    env["SPECPT_RUN_STATE"] = state

    prompt = build_prompt(run_id, run_name, state)

    try:
        import subprocess
        proc = subprocess.Popen(
            [HERMES_BIN, "chat", "-q", prompt,
             "--provider", "opencode-go",
             "--model", WORKING_MODEL,
             "--accept-hooks",
             "--quiet"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        logger.info(f"Triggered orchestrator PID {proc.pid}")
        return True
    except Exception as e:
        logger.error(f"Failed to trigger orchestrator: {e}")
        return False


if __name__ == "__main__":
    mode = os.environ.get("WATCHER_MODE", "latest").lower()
    logger.info(f"Starting watcher (interval={CHECK_INTERVAL}s, mode={mode})")
    while True:
        check_wandb()
        time.sleep(CHECK_INTERVAL)
