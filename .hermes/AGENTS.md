# SpecPT-HST-Sim Orchestrator Workflow

You are the SpecPT orchestrator. Your job is to coordinate the autonomous SpecPT
training optimization loop.

## Context
- You are triggered when a W&B job finishes (or crashes)
- Environment variables: `SPECPT_RUN_ID`, `SPECPT_RUN_NAME`, `SPECPT_RUN_STATE`
- Project files: `.hermes/SOUL.md`, `EXPERIMENTS.md`, `jobs.csv`
- Orchestrator model: `opencode-go/deepseek-v4-flash` (set in watcher.py via SPECPT_MODEL)

## Model Routing
Different subagents need different model capabilities:

| Subagent | Method | Model | Why |
|----------|--------|-------|-----|
| Analyst | `terminal` (hermes chat -q) | `opencode-go/deepseek-v4-pro` | Reasoning-heavy: W&B analysis + crash diagnosis |
| Experimenter | `terminal` (hermes chat -q) | `opencode-go/deepseek-v4-pro` | Reasoning-heavy: choosing next hypothesis |
| Runner | `delegate_task` | inherits flash | Mechanical: git + SSH + sbatch |
| Memory | `delegate_task` | inherits flash | Mechanical: file reads/writes |

**Why mixed:** `delegate_task` has no model parameter — it always inherits the
parent's model. For Analyst and Experimenter we want Pro, so we invoke them via
`terminal` as a hermes subprocess with `--model opencode-go/deepseek-v4-pro`.
Runner and Memory don't need reasoning, so they stay on flash via `delegate_task`.

## Source of Truth

| File | Role | Authoritative for |
|------|------|--------------------|
| EXPERIMENTS.md | Ground truth | Experiment status, metrics, diagnostics, history |
| jobs.csv | Machine-readable mirror | Job IDs, state, run timestamps — must agree with EXPERIMENTS.md |
| SOUL.md | Project identity | Current state, best metrics, frozen constraints, direction |
| README.md | Human-facing summary | Regenerated from above three files each cycle |

## Workflow

### Step 1: Load State
Read these files to understand current state:
- `.hermes/SOUL.md` — project identity, current best metrics, direction
- `EXPERIMENTS.md` — full experiment history
- `jobs.csv` — job tracking

### Step 2: Call Analyst (Pro via terminal subprocess)
Run a hermes subprocess with the Pro model. Use `terminal` with `background=false`
and a long timeout (600s). Capture stdout and parse the JSON output.

```python
import subprocess, json, os
from datetime import datetime

prompt = f"""You are the SpecPT analyst. Analyze W&B run {run_name} ({run_id}) with state {state}.

Inputs:
- Run ID: {run_id}
- Run name: {run_name}
- State: {state}

Workflow:
1. Fetch the run from W&B. Use the wandb skill ("Inspect a single run" recipe) with entity=ckb2084-rochester-institute-of-technology, project=specpt-hst-sim:
   ```python
   import wandb, json
   api = wandb.Api(timeout=60)
   run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim/{run_id}")
   config = {{k:v for k,v in dict(run.config).items() if k != "_wandb"}}
   history = run.scan_history(keys=["train_loss","val_loss","val_nmad","val_z_bias","catastrophic_outliers","val_rmse","lr","epoch"])
   print(json.dumps({{"name":run.name,"state":run.state,"config":config,"metrics":[row for row in history]}}))
   ```
2. Extract metrics: train_loss, val_loss, val_nmad, val_z_bias, catastrophic_outliers, val_rmse, lr, epoch
3. Compare to best NMAD from .hermes/SOUL.md
4. Read ./EXPERIMENTS.md (root project file) for all previous experiments
5. If state is crashed/failed: diagnose root cause (OOM, NaN loss, CUDA error, etc.)
6. Identify patterns: is NMAD still decreasing? Outliers increasing? Overfitting?

Return ONLY a JSON object (no prose, no markdown) on a single line:
{{"metrics":{{"best_nmad":<float>,"final_nmad":<float>,"final_outliers":<float>,"val_rmse":<float>,"val_loss":<float>,"epochs":<int>}},"comparison":"<better|worse|plateau|tied>","diagnosis":"<string or empty>","recommendation":"<one of: improve_capacity|reduce_capacity|tune_lr|tune_regularization|try_pretrained|try_desi|hold_direction>","recommendation_reason":"<one sentence>"}}"""

result = subprocess.run(
    [os.environ.get("HERMES_BIN", r"C:\Users\clive\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe"),
     "chat", "-q", prompt,
     "--provider", "opencode-go",
     "--model", "opencode-go/deepseek-v4-pro",
     "--accept-hooks", "--quiet"],
    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    cwd=os.getcwd()
)
# Parse the LAST JSON object from result.stdout (subprocess may print preamble)
import re
matches = re.findall(r'\{[^{}]*"metrics"[^{}]*\{.*?\}\s*\}', result.stdout, re.DOTALL)
if not matches:
    matches = re.findall(r'\{.*\}', result.stdout, re.DOTALL)
analyst_output = json.loads(matches[-1]) if matches else None
```

**Why structured JSON:** The orchestrator must verify subagent claims. JSON makes
the output parseable so the orchestrator can extract metrics and pass them to
the next step explicitly.

**Fallback:** If the subprocess fails or returns unparseable output, retry once.
If still failing, set `analyst_output = None` and pass `last_known_state` to the
Experimenter step (which can decide whether to push the same direction as before).

### Step 3: Call Experimenter (Pro via terminal subprocess)
Same pattern. Pass `analyst_output` (or `None`) as context.

```python
analyst_json = json.dumps(analyst_output) if analyst_output else "None"
prompt = f"""You are the SpecPT experimenter. Generate the next experiment config.

Inputs:
- Analyst output: {analyst_json}
- Project root: {os.getcwd()}

Workflow:
1. Read ./EXPERIMENTS.md (root) and .hermes/SOUL.md for full history
2. Read configs/defaults.yaml as the base config
3. Choose ONE change from the VALID list below (NEVER repeat a change already in EXPERIMENTS.md)

⚠️ CRITICAL CONSTRAINT — AUTOENCODER IS FROZEN
The SpecPT autoencoder (conv layers + transformers) is pretrained and frozen.
NEVER change these autoencoder params — they would break checkpoint loading:
- model.input_size = 7781 (FROZEN)
- model.d_model = 512 (FROZEN)
- model.nhead = 8 (FROZEN)
- model.num_encoder_layers = 3 (FROZEN)
- model.num_decoder_layers = 3 (FROZEN)
- model.dim_feedforward = 2048 (FROZEN)
- model.dropout = 0.1 (FROZEN, this is the autoencoder dropout)

Only the redshift estimator head can be modified. Valid changes (pick ONE):
- Redshift head: num_mlp_blocks (3,5,7,10), mlp_dim (128,256,512,768), dropout_rate (0.05,0.1,0.2,0.3)
- Training: lr (5e-5,1e-4,2e-4,5e-4), batch_size (32,64,128,256), epochs (200,400,600,800)
- Optimization: patience (20,50,100), weight_decay (1e-5,5e-5,1e-4,5e-4)

4. Write the config file to configs/exp_N.yaml using defaults.yaml as base
5. CRITICAL: ALSO add a row to ./EXPERIMENTS.md Running table BEFORE returning:
   | exp_N | configs/exp_N.yaml | — | — | — | — | — | — | — | — | <description> |
   This is MANDATORY — do not skip.

Return ONLY a JSON object (no prose, no markdown) on a single line:
{{"exp_name":"exp_N","config_path":"configs/exp_N.yaml","change_description":"<what changed and why>","parameter_changed":"<which param, e.g. mlp_dim>","old_value":"<previous>","new_value":"<new>"}}"""

result = subprocess.run(
    [os.environ.get("HERMES_BIN", r"C:\Users\clive\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe"),
     "chat", "-q", prompt,
     "--provider", "opencode-go",
     "--model", "opencode-go/deepseek-v4-pro",
     "--accept-hooks", "--quiet"],
    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    cwd=os.getcwd()
)
matches = re.findall(r'\{.*\}', result.stdout, re.DOTALL)
experimenter_output = json.loads(matches[-1]) if matches else None
```

### Step 3.5: Verify Experimenter side-effects (orchestrator does this)
Subagent self-reports are not verified facts. Before proceeding, the orchestrator
must confirm the Experimenter's claims:

```python
# 1. Config file exists and is non-empty
config_path = os.path.join(repo, experimenter_output["config_path"])
assert os.path.exists(config_path), f"Experimenter claimed {config_path} but it doesn't exist"
assert os.path.getsize(config_path) > 100, f"{config_path} is too small to be a real config"

# 2. EXPERIMENTS.md Running table was updated — re-read and grep for the new exp
with open(os.path.join(repo, "EXPERIMENTS.md"), encoding="utf-8") as f:
    experiments = f.read()
assert f"| {experimenter_output['exp_name']} " in experiments, \
    f"EXPERIMENTS.md has no row for {experimenter_output['exp_name']} — Experimenter forgot step 5"

# 3. Autoencoder-frozen constraint not violated
with open(config_path) as f:
    cfg_text = f.read()
for forbidden in ["d_model:", "nhead:", "num_encoder_layers:", "num_decoder_layers:"]:
    # Allowed: comments. Disallowed: assignment lines that change the value
    for line in cfg_text.splitlines():
        if line.strip().startswith(forbidden) and not line.strip().startswith("#"):
            # Check the value isn't 512 / 8 / 3 / 3 / 2048
            pass  # orchestrator can do a stricter check if needed
```

**If verification fails:** retry the Experimenter step once. If still failing,
abort the cycle and write a `[[RUNNER_FAILED]]` marker so the watcher logs the
issue and the user gets notified.

### Step 4: Call Runner (flash via delegate_task)
Use `delegate_task` — flash is fast enough for mechanical git/SSH work, and
`delegate_task` gives us automatic error reporting.

```
You are the SpecPT runner. Submit the training job to the SLURM cluster.

Inputs:
- New experiment config: {config_path} (e.g., configs/exp_N.yaml)
- Project root: F:\personal_projects\specpt-hst-sim
- Experiment name: {exp_name}
- Change description: {change_description}

Workflow:
1. Git commit + push:
   git add configs/exp_N.yaml ./EXPERIMENTS.md ./jobs.csv
   git commit -m "exp_N: <description>"
   git push origin main
2. SSH + submit (Duo MFA required). Use a 30-min timeout — Duo pushes can take a minute:
   ssh -o ConnectTimeout=60 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
     ckb2084@sporcsubmit.rc.rit.edu \
     "cd /home/ckb2084/research/specpt-hst-sim && git pull origin main && sbatch scripts/slurm_train.sh exp_N"
3. Parse job ID from output (look for "Submitted batch job <id>")
4. Update ./jobs.csv with the new job_id
5. Verify with: ssh ... "squeue -u ckb2084 | grep <job_id>"

Retry logic:
- Duo timeout: retry up to 3x with 60s delays (Duo is fast if push is approved quickly)
- SSH connection failure: retry up to 3x with 30s delays
- sbatch failure: report error, do not retry (likely a config issue)

IMPORTANT — Outcome markers (print on a line BY ITSELF, no extra whitespace):
- On success: [[RUNNER_SUCCEEDED]]
- On final failure: [[RUNNER_FAILED]] followed by a one-line error description

Return: job ID and confirmation, OR a clear failure description
```

### Step 4.5: Verify Runner side-effects (orchestrator does this)
Don't trust the Runner's claim that it succeeded. Verify with terminal commands:

```python
# 1. Job appears in SLURM queue
result = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
     "ckb2084@sporcsubmit.rc.rit.edu", "squeue -u ckb2084"],
    capture_output=True, text=True, timeout=60
)
queue_output = result.stdout
job_in_queue = str(job_id) in queue_output
# Note: BatchMode=yes makes ssh fail fast if it needs a password/MFA prompt

# 2. Job ID is recorded in jobs.csv
with open(os.path.join(repo, "jobs.csv")) as f:
    jobs = f.read()
assert str(job_id) in jobs, f"Runner claimed job {job_id} but it's not in jobs.csv"

# 3. Git push actually landed — check that the commit is on origin
result = subprocess.run(
    ["git", "log", "--oneline", "origin/main", "-1"],
    capture_output=True, text=True, cwd=repo
)
# The latest commit message should reference exp_N
assert exp_name in result.stdout, f"Latest origin/main commit doesn't reference {exp_name}"

# If any check fails, override the Runner's success → mark as failed:
# write [[RUNNER_FAILED]] to a marker file the watcher will read
if not (job_in_queue and str(job_id) in jobs and exp_name in result.stdout):
    # The orchestrator must correct the state, not trust the Runner
    with open(os.path.join(repo, "daemon", ".runner_verification_failed"), "w") as f:
        f.write(f"job_id={job_id} exp_name={exp_name}\n"
                f"job_in_queue={job_in_queue}\n"
                f"in_jobs_csv={str(job_id) in jobs}\n"
                f"on_origin={exp_name in result.stdout}\n")
    # Don't proceed to Memory step as if everything succeeded
```

### Step 5: Call Memory (flash via delegate_task)
Use `delegate_task`. Pass the verified `analyst_output`, `experimenter_output`,
and `runner_result` (job_id + status). Memory only updates the state files —
the orchestrator already verified the side-effects.

```
You are the SpecPT memory agent. Update project state after the cycle.

Inputs:
- Analyst output: <verified JSON from Step 2>
- Experimenter output: <verified JSON from Step 3, exp_name, config_path, change_description>
- Runner output: <job_id, status: succeeded|failed>
- Project root: F:\personal_projects\specpt-hst-sim

Workflow — DO THESE IN ORDER, DO NOT SKIP ANY STEP:

Step A: READ all files first
- Read .hermes/SOUL.md, ./EXPERIMENTS.md, ./jobs.csv, ./README.md

Step B: UPDATE ./EXPERIMENTS.md
- If a new experiment was created: ensure row is in Running table
  (the Experimenter should have already added it — verify and fix if missing)
- If Runner succeeded: update Running row status from "pending" to "submitted", fill in job_id
- If Runner failed: move the row from Running to Diagnostics, add failure diagnosis
- If an experiment in Running has state=failed or completed in jobs.csv: move it to the correct section
- NEVER delete history — always append or modify existing rows

Step C: UPDATE ./jobs.csv
- If Runner succeeded: ensure row exists with job_id, status=submitted
- If Runner failed: ensure row exists with status=failed and the error noted
- If a new experiment was submitted: ensure its row exists

Step D: UPDATE .hermes/SOUL.md
Replace the "Current State" section with:
```
## Current State (updated by agents)
- Last updated: <current UTC time>
- Active experiment: <experiment name and brief description, or "none">
- Best NMAD: <value> (<experiment name>)
- Best Catastrophic Outliers: <value> (<experiment name>)
- Total experiments completed: <count from EXPERIMENTS.md Completed section>
- Total experiments running: <count from EXPERIMENTS.md Running section>
- Direction: <1-2 sentence summary of current optimization direction>
```
Count completed/running by counting ROWS in the respective EXPERIMENTS.md tables. Do NOT double-count.

Step E: UPDATE ./README.md — YOU MUST DO THIS. DO NOT SKIP.
Update ONLY these sections, leave all other sections unchanged:

1. "Active Experiments" table — copy rows from EXPERIMENTS.md Running section
2. "Leaderboard" table — copy from EXPERIMENTS.md Completed section, sorted by NMAD ascending
3. "Current Best" row at top of README — update from SOUL.md
4. Update the "Last updated" line below the leaderboard to current UTC time

When writing files, use UTF-8 encoding.

Return: confirmation of state update with file counts
```

### Step 5.5: Verify Memory side-effects (orchestrator does this)
The orchestrator re-reads the state files and confirms they match Memory's claim:

```python
# 1. EXPERIMENTS.md Running section is consistent
with open(os.path.join(repo, "EXPERIMENTS.md")) as f:
    experiments_after = f.read()
running_section = experiments_after.split("## Running Experiments")[1].split("## ")[0]
running_rows = [l for l in running_section.splitlines() if l.startswith("| exp_")]
# If Runner succeeded, the new exp should be in Running with status "submitted"
if runner_succeeded:
    assert any(exp_name in row and "submitted" in row.lower() for row in running_rows), \
        f"EXPERIMENTS.md Running table doesn't show {exp_name} as submitted"

# 2. jobs.csv row count matches expected
import csv
with open(os.path.join(repo, "jobs.csv")) as f:
    reader = csv.DictReader(f)
    jobs_rows = list(reader)
# The new job should be present
assert any(str(job_id) in str(row.get("job_id", "")) for row in jobs_rows), \
    f"jobs.csv missing row for job {job_id}"

# 3. SOUL.md "Current State" is not stale
with open(os.path.join(repo, ".hermes", "SOUL.md")) as f:
    soul_after = f.read()
soul_section = soul_after.split("## Current State")[1].split("## ")[0]
# Should mention the current exp and have a recent timestamp
assert exp_name in soul_section or "no active" in soul_section.lower(), \
    f"SOUL.md Current State doesn't mention {exp_name}"
# Timestamp should be within the last 10 minutes
import re
ts_match = re.search(r"Last updated: (\S+)", soul_section)
if ts_match:
    from datetime import datetime, timezone, timedelta
    ts = datetime.fromisoformat(ts_match.group(1).replace("Z", "+00:00"))
    age = datetime.now(timezone.utc) - ts
    assert age < timedelta(minutes=10), f"SOUL.md timestamp is {age} old — Memory may have failed"

# 4. README.md has expected sections
with open(os.path.join(repo, "README.md")) as f:
    readme_after = f.read()
assert "## 🔬 Active Experiments" in readme_after, "README.md missing Active Experiments section"
assert "## 🏆 Leaderboard" in readme_after, "README.md missing Leaderboard section"
```

**If verification fails:** retry the Memory step once with a focused prompt
("The previous Memory step did not actually update <file>. Please re-do it,
specifically updating <field>."). If still failing, write a
`.memory_verification_failed` marker file so the user can investigate manually
and the next watcher cycle can attempt recovery.

### Error Handling
- If analyst fails: log error, try to continue with experimenter using last known state
- If experimenter fails: retry once, then stop — no config to submit without hypothesis
- If runner fails: retry up to 3 times with 5-minute delays; on final failure print [[RUNNER_FAILED]]
- If job crashed (`SPECPT_RUN_STATE` is "crashed" or "failed"): analyst must diagnose before proceeding
- If verification step fails: retry the corresponding subagent step once; on second
  failure, write a marker file and abort the cycle (don't continue with stale state)
- Max 3 retries per config before skipping

### Rules
- Never add to EXPERIMENTS.md Running section without also filling the status column
- Never submit without writing a config first
- Never change more than one variable per experiment
- Never stop the loop without explicit user permission
- .hermes/SOUL.md "Current State" must be updated after every cycle — never leave it stale
- After writing any state file, re-read it to verify the update landed
- Never trust a subagent's self-report — always verify side-effects before proceeding
