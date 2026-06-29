import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from .model import SpectrumNormalizer


class HSTGrismDataset(Dataset):
    def __init__(self, df, normalize_fn=SpectrumNormalizer.zscore_normalize):
        x = []
        y = []
        target_id = []
        for _, row in df.iterrows():
            fl = row["spec"]
            fl = normalize_fn(fl)
            x.append(fl)
            y.append(np.array([row["z"]]))
            target_id.append(row["TARGETID"])
        self.X = torch.from_numpy(np.stack(x, axis=0))
        self.Y = torch.from_numpy(np.stack(y, axis=0))
        self.t_id = target_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx].float(), idx, self.t_id[idx]


def _patch_pickle_compat():
    import sys
    import numpy

    if numpy.__version__.startswith('1.'):
        import builtins
        _orig_import = builtins.__import__
        def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith('numpy._core'):
                redirected = 'numpy.core' + name[len('numpy._core'):]
                _orig_import(redirected, globals, locals, fromlist, level)
                sys.modules[name] = sys.modules[redirected]
                return sys.modules[redirected]
            return _orig_import(name, globals, locals, fromlist, level)
        builtins.__import__ = _patched_import

    from pandas import StringDtype
    _orig_init = StringDtype.__init__
    def _patched_init(self, *args, **kwargs):
        try:
            _orig_init(self, *args, **kwargs)
        except TypeError:
            _orig_init(self)
    StringDtype.__init__ = _patched_init


def load_grism_data(data_path, min_snr=2.5):
    _patch_pickle_compat()
    ext = os.path.splitext(data_path)[1]
    if ext == ".parquet":
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)
    data = data[data["SNR"] >= min_snr].copy()
    data.drop(columns=["spec"], inplace=True, errors="ignore")
    data.rename(
        columns={"clean_flux_resampled": "spec", "grism_id": "TARGETID"}, inplace=True
    )
    return data


def split_data(data, val_split=0.1, test_split=0.1, seed=42):
    from sklearn.model_selection import train_test_split

    train_df, temp_test_df = train_test_split(
        data, test_size=val_split + test_split, random_state=seed
    )
    test_df, val_df = train_test_split(
        temp_test_df, test_size=0.5, random_state=seed
    )
    return train_df, val_df, test_df


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size=128,
    val_batch_size=64,
    num_workers=4,
    normalize_fn=SpectrumNormalizer.zscore_normalize,
):
    from torch.utils.data import DataLoader

    train_data = HSTGrismDataset(train_df, normalize_fn=normalize_fn)
    val_data = HSTGrismDataset(val_df, normalize_fn=normalize_fn)
    test_data = HSTGrismDataset(test_df, normalize_fn=normalize_fn)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
