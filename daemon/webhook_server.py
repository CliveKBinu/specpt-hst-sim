import json
import logging
import os
import subprocess
import sys
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

OPENCODE_BIN = os.environ.get("OPENCODE_BIN", "opencode")
SPECPT_REPO = os.environ.get("SPECPT_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


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

    logger.info(f"Received W&B webhook: run_id={run_id}, run_name={run_name}, state={state}")

    if state not in ("finished", "crashed", "failed"):
        logger.info(f"Ignoring non-terminal state: {state}")
        return jsonify({"status": "ignored", "reason": f"state={state}"}), 200

    env = os.environ.copy()
    env["SPECPT_RUN_ID"] = run_id or ""
    env["SPECPT_RUN_NAME"] = run_name or ""
    env["SPECPT_RUN_STATE"] = state or ""

    try:
        proc = subprocess.Popen(
            [OPENCODE_BIN, "--agent", "specpt-orchestrator"],
            cwd=SPECPT_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Launched orchestrator with PID {proc.pid}")
        return jsonify({"status": "launched", "pid": proc.pid}), 200
    except Exception as e:
        logger.error(f"Failed to launch orchestrator: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port)
