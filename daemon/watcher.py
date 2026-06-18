import os
import sys
import json
import logging
import time
import argparse
import platform
from datetime import datetime


def is_process_alive(pid):
    if platform.system() == "Windows":
        import ctypes
        """Check if a Windows process is alive by opening its handle."""
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x400, False, pid)
            if handle == 0:
                return False
            kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_ENTITY = "ckb2084-rochester-institute-of-technology"
DEFAULT_PROJECT = "specpt-hst-sim"
WANDB_ENTITY = DEFAULT_ENTITY
WANDB_PROJECT = DEFAULT_PROJECT
CHECK_INTERVAL = 60
STATE_FILE = os.path.join(os.path.dirname(__file__), ".watcher_state")
LOCK_FILE = os.path.join(os.path.dirname(__file__), ".orchestrator.lock")
DEAD_LETTER_FILE = os.path.join(os.path.dirname(__file__), ".dead_letter.jsonl")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MAX_RUN_AGE = 7 * 24 * 3600
RERUN_INTERVAL = 300  # 5 minutes between retries
MAX_RETRIES = 12  # 1 hour max


HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    "C:\\Users\\clive\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe",
)
WORKING_MODEL = os.environ.get("SPECPT_MODEL", "opencode-go/deepseek-v4-flash")
DELEGATE_MODEL = os.environ.get("SPECPT_DELEGATE_MODEL", "opencode-go/deepseek-v4-pro")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Corrupt state file, starting fresh")
            state = {}
        if "processed_runs" not in state:
            state["processed_runs"] = []
        if "failed_runs" not in state:
            state["failed_runs"] = {}
        if "pending_runs" not in state:
            state["pending_runs"] = {}
        if "experiments_triggered" not in state:
            state["experiments_triggered"] = 0
        return state
    return {
        "last_check": 0,
        "processed_runs": [],
        "failed_runs": {},
        "pending_runs": {},
        "experiments_triggered": 0,
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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
        logger.warning(f"Could not write dead-letter: {e}")


def count_dead_letter():
    if not os.path.exists(DEAD_LETTER_FILE):
        return 0
    try:
        with open(DEAD_LETTER_FILE) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


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
            if is_process_alive(pid):
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
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LANG"] = "C.UTF-8"

    prompt = build_prompt(run_id, run_name, state)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    import subprocess

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"orchestrator_{run_id}_{timestamp}.log")

    try:
        log_file = open(log_path, "w", encoding="utf-8")
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
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        log_file.close()
        logger.info(f"Triggered orchestrator PID {proc.pid} for {run_name} (log={log_path})")
        return proc.pid, log_path
    except Exception as e:
        logger.error(f"Failed to trigger orchestrator for {run_id}: {e}")
        return None, None


def check_orchestrator_results(state):
    """Check completed orchestrator runs and update state accordingly."""
    pending = state.get("pending_runs", {})
    still_pending = {}
    dead_pids = []
    dead_names = []

    for run_id, info in pending.items():
        pid = info.get("proc_pid")
        if pid:
            try:
                if is_process_alive(pid):
                    still_pending[run_id] = info
                    continue
            except Exception:
                pass

        log_path = info.get("log_path", "")
        marker = detect_runner_marker(log_path)
        run_name = info.get("run_name", run_id)

        if marker == "failed":
            failed = state.setdefault("failed_runs", {})
            if run_id not in failed:
                failed[run_id] = {
                    "run_name": info.get("run_name", ""),
                    "exp_name": info.get("exp_name", ""),
                    "attempts": 1,
                    "last_attempt": time.time(),
                    "next_retry": time.time() + RERUN_INTERVAL,
                    "error": "[[RUNNER_FAILED]] detected in orchestrator log",
                }
                logger.warning(f"[FAILED] {run_name} → Runner failed, will retry in {RERUN_INTERVAL}s")
        elif marker == "succeeded":
            processed = state.setdefault("processed_runs", [])
            if run_id not in processed:
                processed.append(run_id)
            logger.info(f"[SUCCESS] {run_name} → Runner succeeded, marked processed")
        else:
            logger.warning(f"[UNCLEAR] {run_name} → orchestrator finished but no marker found")
            processed = state.setdefault("processed_runs", [])
            processed.append(run_id)

    state["pending_runs"] = still_pending
    return state


def detect_runner_marker(log_path):
    if not log_path or not os.path.exists(log_path):
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


def check_new_runs(state):
    """Find unprocessed W&B runs and spawn orchestrator."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed, skipping W&B check")
        return state

    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        order="-created_at",
        per_page=500,
    )
    processed = set(state.get("processed_runs", []))
    pending = set(state.get("pending_runs", {}).keys())
    last_check = state.get("last_check", 0)
    cutoff = max(last_check, time.time() - MAX_RUN_AGE)

    unprocessed = []
    for i, run in enumerate(runs):
        if run.state not in ("finished", "crashed", "failed"):
            continue
        if run.id in processed or run.id in pending:
            continue
        rt = get_run_time(run)
        if rt < cutoff:
            continue
        unprocessed.append((rt, run))

    if not unprocessed:
        return state

    unprocessed.sort(key=lambda x: x[0], reverse=True)
    mode = os.environ.get("WATCHER_MODE", "").lower()
    selected = unprocessed if mode == "all" else [unprocessed[0]]

    for rt, run in selected:
        if not acquire_lock():
            logger.warning(f"Orchestrator already running, skipping {run.name}")
            continue
        try:
            logger.info(f"Found terminal run: {run.name} ({run.state})")
            pid, log_path = trigger_orchestrator(run.id, run.name, run.state)
            if pid:
                pending = state.setdefault("pending_runs", {})
                pending[run.id] = {
                    "proc_pid": pid,
                    "log_path": log_path,
                    "run_name": run.name,
                    "started_at": time.time(),
                }
                state["last_check"] = int(rt) + 1
                state["experiments_triggered"] = state.get("experiments_triggered", 0) + 1
            else:
                logger.error(f"Failed to trigger orchestrator for {run.id}")
        finally:
            release_lock()

    return state


def check_failed_retries(state):
    """Retry failed orchestrator runs that are due for retry."""
    failed = state.get("failed_runs", {})
    now = time.time()
    retried = []

    for run_id, info in list(failed.items()):
        if info.get("next_retry", 0) > now:
            continue

        if info.get("attempts", 0) >= MAX_RETRIES:
            append_dead_letter(
                run_id,
                info.get("exp_name", ""),
                f"exhausted {MAX_RETRIES} retries: {info.get('error', '')}",
                now,
            )
            del failed[run_id]
            logger.warning(f"[DEAD] {info.get('exp_name', run_id)} → max retries reached, moved to dead-letter")
            continue

        if not acquire_lock():
            logger.warning(f"Orchestrator already running, skipping retry for {run_id}")
            continue

        try:
            run_name = info.get("run_name", "")
            state_info = info.get("state", "finished")
            logger.info(f"[RETRY] {run_name} (attempt {info['attempts'] + 1}/{MAX_RETRIES})")
            pid, log_path = trigger_orchestrator(run_id, run_name, state_info)
            if pid:
                pending = state.setdefault("pending_runs", {})
                pending[run_id] = {
                    "proc_pid": pid,
                    "log_path": log_path,
                    "run_name": run_name,
                    "started_at": now,
                }
                info["attempts"] += 1
                info["last_attempt"] = now
                info["next_retry"] = now + RERUN_INTERVAL
                retried.append(run_id)
            else:
                info["attempts"] += 1
                info["last_attempt"] = now
                info["next_retry"] = now + RERUN_INTERVAL
                info["error"] = "Failed to spawn orchestrator process"
        finally:
            release_lock()

    return state


def cycle(state):
    """Run one full watcher cycle."""
    start = time.time()

    state = check_orchestrator_results(state)

    state = check_failed_retries(state)

    state = check_new_runs(state)

    elapsed = time.time() - start
    processed = len(state.get("processed_runs", []))
    pending = len(state.get("pending_runs", {}))
    failed = len(state.get("failed_runs", {}))
    triggered = state.get("experiments_triggered", 0)
    dead = count_dead_letter()
    logger.info(
        f"[CYCLE] {elapsed:.1f}s | "
        f"{processed} processed | "
        f"{pending} pending | "
        f"{failed} failed | "
        f"{triggered} triggered | "
        f"{dead} dead-letter"
    )

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecPT watcher daemon")
    parser.add_argument(
        "--once", action="store_true",
        help="Run one cycle then exit (for testing/debugging)",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to watcher log file (e.g. daemon/logs/watcher.log)",
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help=f"W&B project (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=None,
        help="Process N terminal runs total then exit (cumulative across restarts)",
    )
    parser.add_argument(
        "--max-new", type=int, default=None,
        help="Process N new runs in this session then exit (resets on restart)",
    )
    args = parser.parse_args()
    if args.entity:
        WANDB_ENTITY = args.entity
    if args.project:
        WANDB_PROJECT = args.project

    if args.log:
        file_handler = logging.FileHandler(args.log, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
        ))
        logger.addHandler(file_handler)

    mode = os.environ.get("WATCHER_MODE", "latest").lower()
    max_exp = getattr(args, 'max_experiments', None)
    logger.info(
        f"Starting watcher "
        f"(interval={CHECK_INTERVAL}s, mode={mode}, once={args.once}"
        f"{f', max_experiments={max_exp}' if max_exp else ''}"
        f"{f', max_new={args.max_new}' if args.max_new else ''})"
    )

    state = load_state()
    dead_count = count_dead_letter()
    if dead_count > 0:
        logger.warning(f"{dead_count} entries in dead-letter queue")
        try:
            with open(DEAD_LETTER_FILE) as f:
                lines = f.readlines()
            for line in lines[-3:]:
                entry = json.loads(line.strip())
                logger.warning(f"  DEAD: {entry['run_id']} — {entry['error'][:80]}")
        except Exception:
            pass

    if args.once:
        state = cycle(state)
        save_state(state)
        logger.info("Exiting (--once flag)")
    else:
        max_exp = args.max_experiments
        max_new = args.max_new
        session_triggered = 0
        prev_total = state.get("experiments_triggered", 0)
        logger.info(
            f"Entering main loop"
            f"{f' (max {max_exp} total)' if max_exp else ''}"
            f"{f' (max {max_new} new this session)' if max_new else ''}"
        )
        while True:
            try:
                state = cycle(state)
                save_state(state)
                new_total = state.get("experiments_triggered", 0)
                session_triggered += new_total - prev_total
                prev_total = new_total
                if max_exp and new_total >= max_exp:
                    logger.info(
                        f"Reached {max_exp} total experiments "
                        f"({new_total} triggered), exiting"
                    )
                    break
                if max_new and session_triggered >= max_new:
                    logger.info(
                        f"Reached {max_new} new experiments "
                        f"in this session, exiting"
                    )
                    break
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Shutting down watcher")
                break
            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)
                time.sleep(CHECK_INTERVAL)
