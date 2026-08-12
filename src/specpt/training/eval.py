import numpy as np
import json
import torch
from ..model import binned_predict


def decode_redshift_output(outputs, prediction_type="point"):
    """Return scalar z predictions from point, MDN, or binned outputs."""
    if prediction_type == "binned":
        return binned_predict(outputs)[0]
    if prediction_type == "mdn":
        means, _, mix_weights = outputs
        batch_size = means.shape[0]
        num_mixtures = mix_weights.shape[1]
        means_reshaped = means.view(batch_size, num_mixtures, 1)
        best_comp = torch.argmax(mix_weights, dim=-1)
        return means_reshaped[torch.arange(batch_size, device=means.device), best_comp, 0]
    return outputs


def predict_with_tta(model, test_loader, device, tta_config, prediction_type=None):
    """Run test-time augmentation and return averaged predictions.

    Args:
        model: The redshift prediction model (in eval mode).
        test_loader: DataLoader for test set.
        device: torch device.
        tta_config: Dict with keys:
            - n_augmentations (int): Number of augmented copies per sample.
            - noise_std (float): Std dev of Gaussian noise augmentation.
            - max_shift (int): Max pixel wavelength shift.
            - flux_scale_range (list): [min, max] flux scaling factor.

    Returns:
        y_true: np.array of true redshifts.
        y_pred: np.array of TTA-averaged predictions.
    """
    n_aug = tta_config.get("n_augmentations", 10)
    noise_std = tta_config.get("noise_std", 0.01)
    max_shift = tta_config.get("max_shift", 3)
    flux_scale_min, flux_scale_max = tta_config.get("flux_scale_range", [0.95, 1.05])

    if prediction_type is None:
        if hasattr(model, "binned_output") and model.binned_output:
            prediction_type = "binned"
        else:
            prediction_type = "mdn" if hasattr(model, "mdn_head") else "point"

    model.eval()
    all_true = []
    all_preds = []

    # Check if model is MDN by checking return type
    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X, Y = X.to(device), Y.to(device)
            batch_size = X.shape[0]

            # Collect predictions across all augmentations
            aug_preds = torch.zeros(batch_size, 1, device=device)

            for _ in range(n_aug):
                X_aug = X.clone()

                # 1. Gaussian noise injection
                if noise_std > 0:
                    noise = torch.randn_like(X_aug) * noise_std
                    X_aug = X_aug + noise

                # 2. Wavelength shift (circular shift along feature dim)
                if max_shift > 0:
                    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                    if shift != 0:
                        X_aug = torch.roll(X_aug, shifts=shift, dims=-1)

                # 3. Flux scaling
                scale = torch.empty(batch_size, 1, device=device).uniform_(
                    flux_scale_min, flux_scale_max
                )
                X_aug = X_aug * scale

                preds = decode_redshift_output(model(X_aug), prediction_type)
                aug_preds += preds.reshape(batch_size, 1)

            # Average predictions across augmentations
            aug_preds /= n_aug

            all_true.append(Y.cpu().numpy().flatten())
            all_preds.append(aug_preds.cpu().numpy().flatten())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred


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
