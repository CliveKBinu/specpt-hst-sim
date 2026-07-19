#!/usr/bin/env python3
"""Check exp_009 (job 21351040) status."""
import json, os, sys
api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("ERROR: WANDB_API_KEY not set", file=sys.stderr)
    sys.exit(1)
import wandb
api = wandb.Api(timeout=60, api_key=api_key)

# List all runs in the project
project = api.project("specpt-hst-sim-z", entity="ckb2084-rochester-institute-of-technology")
runs = api.runs("ckb2084-rochester-institute-of-technology/specpt-hst-sim-z")
print("=== ALL RUNS IN PROJECT ===")
for r in runs:
    cfg = dict(r.config)
    mlp = cfg.get("model", {}).get("num_mlp_blocks", "?")
    mlp_dim = cfg.get("model", {}).get("mlp_dim", "?")
    lr = cfg.get("training", {}).get("lr", "?")
    print(f"  {r.name:30s} | state={r.state:10s} | blocks={mlp} | mlp_dim={mlp_dim} | lr={lr}")
