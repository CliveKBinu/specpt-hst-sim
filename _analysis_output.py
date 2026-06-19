import json

d = json.load(open('_ld28scut_data.json'))
m = d['metrics']

train_losses = [r['train_loss'] for r in m]
val_losses = [r['val_loss'] for r in m]
val_nmads = [r['val_nmad'] for r in m]
val_outs = [r['catastrophic_outliers'] for r in m]
val_biases = [r['val_z_bias'] for r in m]
val_rmses = [r['val_rmse'] for r in m]
epochs = [r['epoch'] for r in m]

# Best NMAD
best_nmad_idx = min(range(len(val_nmads)), key=val_nmads.__getitem__)
best = m[best_nmad_idx]
last = m[-1]
first = m[0]

print("=== RUN ===")
print(f"Name: {d['name']}")
print(f"State: {d['state']}")
print(f"Config: num_mlp_blocks={d['config']['model']['num_mlp_blocks']}, mlp_dim={d['config']['model']['mlp_dim']}")
print(f"  lr={d['config']['training']['lr']}, weight_decay={d['config']['training']['weight_decay']}")
print(f"  autoencoder=HST_augmented (OLD)")
print(f"Total epochs: {len(m)} (range {first['epoch']}-{last['epoch']})")

print(f"\n=== FIRST EPOCH (ep {first['epoch']}) ===")
print(f"  NMAD={first['val_nmad']:.5f}, Loss={first['val_loss']:.4f}, Train={first['train_loss']:.4f}, Out={first['catastrophic_outliers']:.2f}%")

print(f"\n=== BEST NMAD (ep {best['epoch']}) ===")
print(f"  NMAD={best['val_nmad']:.5f}")
print(f"  Val Loss={best['val_loss']:.4f}, Train Loss={best['train_loss']:.4f}")
print(f"  Cat Out={best['catastrophic_outliers']:.2f}%")
print(f"  Z Bias={best['val_z_bias']:.5f}, RMSE={best['val_rmse']:.4f}")

print(f"\n=== LAST EPOCH (ep {last['epoch']}) ===")
print(f"  NMAD={last['val_nmad']:.5f}")
print(f"  Val Loss={last['val_loss']:.4f}, Train Loss={last['train_loss']:.4f}")
print(f"  Cat Out={last['catastrophic_outliers']:.2f}%")
print(f"  Z Bias={last['val_z_bias']:.5f}, RMSE={last['val_rmse']:.4f}")
print(f"  LR={last['lr']:.6e}")

# Trend
print(f"\n=== TRENDS ===")

# NMAD last 20
last20_nmad = val_nmads[-20:]
first_last20 = last20_nmad[0]
last_last20 = last20_nmad[-1]
print(f"NMAD last 20 epochs: {first_last20:.5f} -> {last_last20:.5f} (delta: {last_last20-first_last20:.5f})")
improving = all(last20_nmad[i] >= last20_nmad[i+1] for i in range(len(last20_nmad)-1))
print(f"Monotonically improving: {improving}")

# Overfitting check
avg_vl_50 = sum(val_losses[:50])/50
avg_vl_last50 = sum(val_losses[-50:])/50
print(f"Avg Val Loss first 50: {avg_vl_50:.4f}")
print(f"Avg Val Loss last 50: {avg_vl_last50:.4f}")
overfitting = avg_vl_last50 > avg_vl_50 * 1.05
print(f"Overfitting (val_loss >5% higher): {overfitting}")

# Catastrophic outliers
avg_out_first50 = sum(val_outs[:50])/50
avg_out_last50 = sum(val_outs[-50:])/50
print(f"Avg Outliers first 50: {avg_out_first50:.2f}%")
print(f"Avg Outliers last 50: {avg_out_last50:.2f}%")
out_increasing = avg_out_last50 > avg_out_first50
print(f"Outliers increasing: {out_increasing}")

# Train loss vs val loss gap
gap_first = val_losses[0] - train_losses[0]
gap_last = val_losses[-1] - train_losses[-1]
print(f"Train-Val gap first: {gap_first:.4f}")
print(f"Train-Val gap last: {gap_last:.4f}")
print(f"Gap widening (overfitting): {gap_last > gap_first * 1.5}")

# Best NMAD from context is 0.02565 (exp_007)
best_context = 0.02565
improvement = (best_context - best['val_nmad']) / best_context * 100
print(f"\n=== COMPARISON TO BEST (exp_007: {best_context}) ===")
print(f"drawn-sun-20 BEST NMAD: {best['val_nmad']:.5f}")
print(f"Improvement: {improvement:+.2f}%")
if best['val_nmad'] < best_context:
    print(f"*** NEW BEST! Beat exp_007 by {abs(improvement):.2f}% ***")
else:
    print(f"Did NOT beat exp_007. Worse by {abs(improvement):.2f}%")

# Catastrophic outliers comparison to best outlier run (exp_005: 23.18%)
print(f"\nBest outliers this run: {best['catastrophic_outliers']:.2f}%")
print(f"Best overall (exp_005): 23.18%")

print(f"\n=== DIAGNOSIS ===")
final_nmad = last['val_nmad']
best_nmad = best['val_nmad']
if final_nmad > best_nmad * 1.1:
    print("WARNING: Final NMAD significantly worse than best - model overfitting and reverting")
elif final_nmad > best_nmad * 1.05:
    print("NOTE: Final NMAD somewhat worse than best - mild overfitting/reversion")
else:
    print("OK: Final NMAD close to best - stable")
    
if overfitting:
    print("OVERFITTING DETECTED: Validation loss rising while train loss decreasing")
else:
    print("No significant overfitting detected")
    
if out_increasing:
    print("CATASTROPHIC OUTLIERS INCREASING: Model producing more large errors over time")
else:
    print("Catastrophic outliers stable or decreasing")

# Is NMAD still decreasing at termination?
last10_nmad = val_nmads[-10:]
if min(last10_nmad) < best_nmad * 1.01:
    print("NMAD near best at termination - capacity may still be beneficial")
else:
    print("NMAD has pulled away from best - consider early stopping or adjusting LR schedule")

print(f"\n=== RECOMMENDATION ===")
if best['val_nmad'] < 0.02565:
    print(f"Capacity increase to 12 blocks improved NMAD from 0.02565 to {best['val_nmad']:.5f} ({improvement:+.2f}%).")
    print("Direction: Continue pushing capacity (12->15 blocks) or try other head params.")
else:
    print(f"Capacity increase to 12 blocks did NOT improve over 10-block result ({best['val_nmad']:.5f} vs 0.02565).")
    print("Direction: Capacity bottleneck may be plateauing. Try other params (mlp_dim, dropout_rate, lr).")
