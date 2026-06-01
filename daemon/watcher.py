import os
import sys
import json
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ckb2084-rochester-institute-of-technology")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "specpt-hst-sim")
CHECK_INTERVAL = 1800  # 30 minutes
STATE_FILE = os.path.join(os.path.dirname(__file__), ".watcher_state")


def load_last_check():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_check": 0}


def save_last_check(ts):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_check": ts}, f)


def check_wandb():
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
        state = load_last_check()
        last_check = state["last_check"]

        for run in runs:
            if run.state in ("finished", "crashed", "failed"):
                run_time = run.created_at.timestamp() if hasattr(run.created_at, 'timestamp') else 0
                if run_time > last_check:
                    logger.info(f"Found terminal run: {run.name} ({run.state})")
                    trigger_orchestrator(run.id, run.name, run.state)

        save_last_check(time.time())
    except Exception as e:
        logger.error(f"W&B check failed: {e}")


def trigger_orchestrator(run_id, run_name, state):
    env = os.environ.copy()
    env["SPECPT_RUN_ID"] = run_id
    env["SPECPT_RUN_NAME"] = run_name
    env["SPECPT_RUN_STATE"] = state

    try:
        import subprocess
        proc = subprocess.Popen(
            [os.environ.get("OPENCODE_BIN", "opencode"), "--agent", "specpt-orchestrator"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Triggered orchestrator PID {proc.pid}")
    except Exception as e:
        logger.error(f"Failed to trigger orchestrator: {e}")


if __name__ == "__main__":
    logger.info(f"Starting watcher (interval={CHECK_INTERVAL}s)")
    while True:
        check_wandb()
        time.sleep(CHECK_INTERVAL)
