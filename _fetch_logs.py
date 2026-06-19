#!/usr/bin/env python3
"""Fetch output.log from noble-frog-23 run."""
import json, os, sys

api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("ERROR: WANDB_API_KEY not set", file=sys.stderr)
    sys.exit(1)

import wandb
api = wandb.Api(timeout=60, api_key=api_key)
run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim/nsomfkte")

# Download and read output.log
print("=== OUTPUT.LOG ===")
try:
    for f in run.files():
        if f.name == "output.log":
            content = f.download(replace=True, exist_ok=True)
            # Read the downloaded file
            with open("output.log", "r") as lf:
                print(lf.read()[:5000])
            break
except Exception as e:
    print(f"Error downloading output.log: {e}")
    import traceback
    traceback.print_exc()

# Also try wandb-metadata.json
print("\n=== WANDB-METADATA.JSON ===")
try:
    for f in run.files():
        if f.name == "wandb-metadata.json":
            content = f.download(replace=True, exist_ok=True)
            with open("wandb-metadata.json", "r") as mf:
                print(mf.read()[:2000])
            break
except Exception as e:
    print(f"Error: {e}")

# Try wandb-summary.json
print("\n=== WANDB-SUMMARY.JSON ===")
try:
    for f in run.files():
        if f.name == "wandb-summary.json":
            content = f.download(replace=True, exist_ok=True)
            with open("wandb-summary.json", "r") as sf:
                print(sf.read()[:2000])
            break
except Exception as e:
    print(f"Error: {e}")
