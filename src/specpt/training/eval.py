import numpy as np
import json


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    delz = (y_pred - y_true) / (1 + y_true)
    nmad = 1.4826 * np.median(np.abs(delz))
    eta = 100.0 * np.mean(np.abs(delz) > 0.15)
    rmse = np.sqrt(np.mean(delz**2))
    bias = np.median(delz)
    return {"nmad": nmad, "eta": eta, "rmse": rmse, "bias": bias}


def compute_per_bin_metrics(y_true, y_pred, bins=None):
    if bins is None:
        bins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    results = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if np.sum(mask) > 0:
            metrics = compute_metrics(y_true[mask], y_pred[mask])
            metrics["bin"] = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
            metrics["count"] = int(np.sum(mask))
        else:
            metrics = {"bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})", "count": 0, "nmad": float("nan"), "eta": float("nan"), "rmse": float("nan"), "bias": float("nan")}
        results.append(metrics)
    return results


def export_metrics_json(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
