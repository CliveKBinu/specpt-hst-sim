#!/usr/bin/env python3
import torch
ckpt = torch.load('checkpoints/exp_032_best_model.pth', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
for k in ['prediction.0.weight', 'prediction.0.bias', 'prediction.3.weight', 'prediction.3.bias']:
    t = sd[k]
    print(f'{k}: shape={list(t.shape)}, NaN={torch.isnan(t).sum().item()}, sum={t.sum().item():.4f}')
