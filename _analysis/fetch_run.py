#!/usr/bin/env python
"""Fetch W&B run super-disco-12 (wijvni9f) metrics for analysis."""
import wandb
import json

api = wandb.Api(timeout=60)
run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim-z/wijvni9f")
config = {k: v for k, v in dict(run.config).items() if k != "_wandb"}
history = run.scan_history(keys=["train_loss", "val_loss", "val_nmad", "val_z_bias",
                                  "catastrophic_outliers", "val_rmse", "lr", "epoch"])
print(json.dumps({"name": run.name, "state": run.state, "config": config,
                   "metrics": [row for row in history]}))
