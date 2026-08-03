import sys, tempfile, os, glob, yaml
import torch

sys.path.insert(0, os.path.abspath("."))

from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift

ARCHS = [
    ("configs/autoencoder_tracka_control.yaml", "configs/tracka_control_z.yaml"),
    ("configs/autoencoder_tracka_small.yaml", "configs/tracka_small_z.yaml"),
    ("configs/autoencoder_tracka_tiny.yaml", "configs/tracka_tiny_z.yaml"),
]

def load_model_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]

fails = []
for ae_cfg_path, z_cfg_path in ARCHS:
    mc = load_model_config(ae_cfg_path)
    zc = load_model_config(z_cfg_path)
    tag = os.path.basename(ae_cfg_path).replace("autoencoder_tracka_", "").replace(".yaml", "")

    # 1. model configs agree between AE and redshift configs
    for k in ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward"]:
        if mc[k] != zc[k]:
            fails.append(f"[{tag}] AE vs Z model.{k} mismatch: {mc[k]} != {zc[k]}")

    # 2. nhead divides d_model
    if mc["d_model"] % mc["nhead"] != 0:
        fails.append(f"[{tag}] nhead={mc['nhead']} does not divide d_model={mc['d_model']}")

    # 3. instantiate + forward AE
    ae = SpecPT(input_size=mc["input_size"], d_model=mc["d_model"], nhead=mc["nhead"],
                num_encoder_layers=mc["num_encoder_layers"], num_decoder_layers=mc["num_decoder_layers"],
                dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"])
    ae.eval()
    n_ae = sum(p.numel() for p in ae.parameters())
    x = torch.randn(2, 7781)
    with torch.no_grad():
        y = ae(x)
    assert y.shape == (x.shape[0], 7781), f"[{tag}] AE out shape {y.shape}"
    print(f"[{tag}] AE params={n_ae:,}  out={tuple(y.shape)}")

    # 4. redshift model instantiates + forward (frozen backbone)
    rz = EnhancedSpecPTForRedshift(ae, output_features=1, num_mlp_blocks=5,
                                   mlp_dim=zc["mlp_dim"], dropout_rate=zc["dropout"])
    for p in rz.pretrained_model.parameters():
        p.requires_grad = False
    rz.eval()
    n_rz = sum(p.numel() for p in rz.parameters() if p.requires_grad)
    with torch.no_grad():
        z = rz(x)
    assert z.shape == (x.shape[0], 1), f"[{tag}] Z out shape {z.shape}"
    assert torch.isfinite(z).all(), f"[{tag}] non-finite z"
    print(f"[{tag}] Z trainable={n_rz:,}  out={tuple(z.shape)}  sample={z.flatten()[:3].tolist()}")

    # 5. checkpoint save/reload strict round-trip (weights-only form used by trainer).
    #    Skip for the 512 control (~1B params, ~4GB state dict IO) — it is the
    #    known-good baseline and identical to the historical 512 checkpoint.
    if tag != "control":
        tmp = os.path.join(tempfile.gettempdir(), f"smoke_{tag}.pth")
        torch.save(ae.state_dict(), tmp)
        ae2 = SpecPT(input_size=mc["input_size"], d_model=mc["d_model"], nhead=mc["nhead"],
                     num_encoder_layers=mc["num_encoder_layers"], num_decoder_layers=mc["num_decoder_layers"],
                     dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"])
        missing, unexpected = ae2.load_state_dict(torch.load(tmp, weights_only=False), strict=True)
        assert not missing and not unexpected, f"[{tag}] strict reload failed"
        # full-dict form also supported (best checkpoint)
        full = os.path.join(tempfile.gettempdir(), f"smoke_full_{tag}.pth")
        torch.save({"epoch": 1, "model_state_dict": ae.state_dict()}, full)
        sd = torch.load(full, weights_only=False)
        sd = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
        missing, unexpected = ae2.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"[{tag}] full-dict strict reload failed"
        os.remove(tmp); os.remove(full)

    # 6. mismatched arch must fail strict load (clear error, not silent)
    wrong = SpecPT(input_size=7781, d_model=64, nhead=2, num_encoder_layers=1,
                   num_decoder_layers=1, dim_feedforward=256, dropout=0.1)
    try:
        wrong.load_state_dict(ae.state_dict(), strict=True)
        fails.append(f"[{tag}] mismatched arch loaded silently!")
    except RuntimeError:
        print(f"[{tag}] mismatched arch correctly rejected")

# 7. scratch-init configs: pretrained_autoencoder empty
for ae_cfg_path, _ in ARCHS:
    with open(ae_cfg_path) as f:
        cfg = yaml.safe_load(f)
    pe = cfg["data"].get("pretrained_autoencoder", "")
    if pe not in ("", None):
        fails.append(f"[{os.path.basename(ae_cfg_path)}] pretrained_autoencoder not empty: {pe}")

print("\n" + "=" * 50)
if fails:
    print("SMOKE TEST FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
