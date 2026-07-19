import pandas as pd, numpy as np
df = pd.read_csv('/home/ckb2084/research/specpt-hst-sim/outputs/real_3dhst/exp_032_real_eval.csv')
df['delz'] = (df['z_pred'] - df['z_true']) / (1 + df['z_true'])
print('n=', len(df))
print('z_true range:', round(df['z_true'].min(),2), '-', round(df['z_true'].max(),2))
print('z_pred range:', round(df['z_pred'].min(),2), '-', round(df['z_pred'].max(),2))
print('mean z_pred:', round(df['z_pred'].mean(),3))
print('mean z_true:', round(df['z_true'].mean(),3))
print('mean delz:', round(df['delz'].mean(),3))
print('|delz|>0.15:', round((np.abs(df['delz']) > 0.15).mean() * 100, 1))
print()
for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)]:
    m = (df['z_true'] >= lo) & (df['z_true'] < hi)
    print(f'z [{lo}, {hi}): n={m.sum()}, mean z_pred={df.loc[m,"z_pred"].mean():.3f}, mean z_true={df.loc[m,"z_true"].mean():.3f}, bias={df.loc[m,"delz"].mean():.3f}')
