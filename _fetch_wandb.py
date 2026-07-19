#!/usr/bin/env python3
"""Fetch W&B run noble-frog-23 (nsomfkte) and print diagnostics."""
import json, os, sys

# Use the API key from environment
api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("ERROR: WANDB_API_KEY not set", file=sys.stderr)
    sys.exit(1)

import wandb
api = wandb.Api(timeout=60, api_key=api_key)
run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim-z/nsomfkte")

# Config (strip wandb internal keys)
config = {k: v for k, v in dict(run.config).items() if k != "_wandb"}

# History
history = run.scan_history(keys=[
    "train_loss", "val_loss", "val_nmad", "val_z_bias",
    "catastrophic_outliers", "val_rmse", "lr", "epoch"
])

output = {
    "name": run.name,
    "state": run.state,
    "config": config,
    "metrics": [row for row in history],
}
print(json.dumps(output, indent=2))
