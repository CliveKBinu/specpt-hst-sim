import wandb, json

api = wandb.Api(timeout=60)
run = api.run("ckb2084-rochester-institute-of-technology/specpt-hst-sim-z/ld28scut")
config = {k: v for k, v in dict(run.config).items() if k != "_wandb"}

try:
    history = run.scan_history()
    rows = [row for row in history]
except Exception as e:
    print(f"scan_history failed: {e}")
    try:
        hist_df = run.history()
        if hist_df is not None and not hist_df.empty:
            rows = hist_df.to_dict('records')
        else:
            rows = []
    except Exception as e2:
        print(f"history() also failed: {e2}")
        rows = []

summary_clean = {}
for k, v in dict(run.summary).items():
    if not k.startswith("_") and not isinstance(v, dict):
        summary_clean[k] = v

result = {
    "name": run.name,
    "state": run.state,
    "config": config,
    "metrics": rows if rows else "no_history",
    "summary": summary_clean,
}

with open("_ld28scut_data.json", "w") as f:
    json.dump(result, f, default=str, indent=2)

print(f"Saved run {result['name']} ({len(rows)} rows)")
