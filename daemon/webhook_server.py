import json
import logging
import os
import subprocess
import sys
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("daemon/webhook.log"),
    ],
)
logger = logging.getLogger(__name__)

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
SPECPT_REPO = os.environ.get("SPECPT_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".orchestrator.lock")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".watcher_state")
RECENTLY_PROCESSED = set()
RECENTLY_PROCESSED_LOCK = threading.Lock()


def is_processed(run_id):
    with RECENTLY_PROCESSED_LOCK:
        if run_id in RECENTLY_PROCESSED:
            return True
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                state = json.load(f)
            if run_id in state.get("processed_runs", []):
                return True
    except (json.JSONDecodeError, IOError):
        pass
    return False


def mark_processed(run_id):
    with RECENTLY_PROCESSED_LOCK:
        RECENTLY_PROCESSED.add(run_id)


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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/wandb", methods=["POST"])
def wandb_webhook():
    if WEBHOOK_SECRET and request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        logger.warning("Unauthorized webhook attempt")
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"error": "no json payload"}), 400

    run_id = data.get("runId")
    run_name = data.get("runName")
    state = data.get("state")

    logger.info(f"Received W&B webhook: run_id={run_id}, state={state}")

    if state not in ("finished", "crashed", "failed"):
        return jsonify({"status": "ignored", "reason": f"state={state}"}), 200

    if is_processed(run_id):
        logger.info(f"Skipping duplicate webhook for {run_id}")
        return jsonify({"status": "duplicate"}), 200

    if not acquire_lock():
        logger.warning(f"Orchestrator already running, skipping {run_id}")
        return jsonify({"status": "locked"}), 200

    try:
        env = os.environ.copy()
        env["SPECPT_RUN_ID"] = run_id or ""
        env["SPECPT_RUN_NAME"] = run_name or ""
        env["SPECPT_RUN_STATE"] = state or ""

        prompt = build_prompt(run_id, run_name, state)
        proc = subprocess.Popen(
            [HERMES_BIN, "chat", "-q", prompt,
             "--provider", "opencode-go",
             "--model", WORKING_MODEL,
             "--accept-hooks",
             "--quiet"],
            cwd=SPECPT_REPO,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mark_processed(run_id)
        logger.info(f"Launched orchestrator PID {proc.pid}")
        return jsonify({"status": "launched", "pid": proc.pid}), 200
    except Exception as e:
        logger.error(f"Failed to launch orchestrator: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        release_lock()


if __name__ == "__main__":
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET is empty — webhook is unauthenticated!")
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port)
