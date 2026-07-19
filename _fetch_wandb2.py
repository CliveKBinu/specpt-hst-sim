#!/usr/bin/env python3
"""Fetch W&B run errors/system logs for noble-frog-23."""
import json, os, sys

api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("ERROR: WANDB_API_KEY not set", file=sys.stderr)
    sys.exit(1)

import wandb
api = wandb.Api(timeout=60, api_key=api_key)
run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim-z/nsomfkte")

# Get system events/errors
print("=== RUN METADATA ===")
print(f"Name: {run.name}")
print(f"State: {run.state}")
print(f"Tags: {run.tags}")
print(f"Notes: {run.notes}")
print(f"Created at: {run.created_at}")
print(f"Runtime: {run.summary.get('_runtime', 'N/A')}")
print(f"Host: {run.summary.get('_hostname', 'N/A')}")

# Scan all history (any keys present)
print("\n=== ALL METRICS ===")
try:
    full_history = run.scan_history()
    rows = [row for row in full_history]
    print(f"Total rows: {len(rows)}")
    if rows:
        for r in rows[:10]:
            print(json.dumps(r, indent=2))
    else:
        print("No rows at all — run died before logging anything")
except Exception as e:
    print(f"Error scanning history: {e}")

# Check files
print("\n=== RUN FILES ===")
try:
    files = run.files()
    file_list = [f.name for f in files]
    print(json.dumps(file_list, indent=2))
except Exception as e:
    print(f"Error listing files: {e}")

# Check summary
print("\n=== SUMMARY ===")
try:
    summary_dict = dict(run.summary)
    print(json.dumps(summary_dict, indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")
