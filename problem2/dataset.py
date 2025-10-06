# problem2/dataset.py
# Compatible with setup_data.py outputs:
#   data/drums/patterns.npz with keys:
#       train_patterns [N,16,9], val_patterns [M,16,9]  (uint8/bool)
#       train_styles   [N],       val_styles   [M]      (int)
#   data/drums/patterns.json with keys:
#       instruments (len=9), styles (len=5), timesteps=16

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _default_dirs():
    """Resolve data directories relative to this file."""
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    drums_dir = os.path.join(data_dir, "drums")
    return data_dir, drums_dir


class DrumPatternDataset(Dataset):
    """
    Drum pattern dataset for HW2 Problem 2 (VAE).

    Returns by default:
        x : FloatTensor of shape [1, 16, 9] (values in {0.0, 1.0})
        y : LongTensor scalar (style index 0..4)
        density : float (optional, if return_density=True)

    Args:
        data_dir (str): path to '.../problem2/data/drums'. If None, resolve automatically.
        split (str): 'train' or 'val'
        add_channel_dim (bool): if True, returns [1,16,9]; else [16,9]
        dtype (torch.dtype): tensor dtype for x
        return_label (bool): if False, only returns x (useful for plain VAE)
        return_density (bool): if True, also returns density float
    """

    def __init__(
        self,
        data_dir: str | None = None,
        split: str = "train",
        add_channel_dim: bool = True,
        dtype: torch.dtype = torch.float32,
        return_label: bool = True,
        return_density: bool = False,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.split = split
        self.add_channel_dim = add_channel_dim
        self.dtype = dtype
        self.return_label = return_label
        self.return_density = return_density

        # Resolve directories
        if data_dir is None:
            _, drums_dir = _default_dirs()
        else:
            drums_dir = data_dir

        # ---------- Load NPZ ----------
        npz_path = os.path.join(drums_dir, "patterns.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Cannot find '{npz_path}'. "
                "Run setup_data.py with --output-dir problem2/data first."
            )
        npz = np.load(npz_path, allow_pickle=True)

        pat_key = f"{split}_patterns"  # 'train_patterns' or 'val_patterns'
        sty_key = f"{split}_styles"    # 'train_styles'   or 'val_styles'
        if pat_key not in npz or sty_key not in npz:
            raise KeyError(
                f"Expected keys '{pat_key}' and '{sty_key}' in {npz_path}. "
                f"Found keys: {list(npz.keys())}"
            )

        self.patterns = np.array(npz[pat_key], dtype=np.float32)  # [N,16,9] in {0,1}
        self.styles = np.array(npz[sty_key], dtype=np.int64)      # [N]

        # ---------- Load metadata JSON ----------
        meta_path = os.path.join(drums_dir, "patterns.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Cannot find '{meta_path}'. The generator should have created it."
            )
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.instrument_names = meta.get("instruments", [])
        self.style_names = meta.get("styles", [])
        self.timesteps = int(meta.get("timesteps", 16))

        # Basic sanity checks
        if self.patterns.ndim != 3 or self.patterns.shape[1:] != (16, 9):
            raise ValueError(
                f"Expected patterns of shape [N,16,9], got {self.patterns.shape}"
            )
        if len(self.styles) != len(self.patterns):
            raise ValueError(
                "Length mismatch: styles and patterns have different lengths."
            )

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx: int):
        pat = self.patterns[idx]  # np.float32, shape [16,9], values {0.0,1.0}
        style = int(self.styles[idx])

        # x tensor
        x = torch.from_numpy(pat).to(self.dtype)        # [16,9]
        if self.add_channel_dim:
            x = x.unsqueeze(0)                          # [1,16,9]

        if not self.return_label and not self.return_density:
            return x

        y = torch.tensor(style, dtype=torch.long)       # scalar label
        if not self.return_density:
            return x, y

        density = float(pat.sum() / (16 * 9))           # scalar float
        return x, y, density


# ---------- Convenience helpers ----------

def get_drum_dataloaders(
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    add_channel_dim: bool = True,
    return_label: bool = True,
    return_density: bool = False,
    dtype: torch.dtype = torch.float32,
    data_dir: str | None = None,
):
    """
    Quickly build train/val dataloaders consistent with this dataset.
    """
    if data_dir is None:
        _, drums_dir = _default_dirs()
    else:
        drums_dir = data_dir

    train_ds = DrumPatternDataset(
        data_dir=drums_dir,
        split="train",
        add_channel_dim=add_channel_dim,
        dtype=dtype,
        return_label=return_label,
        return_density=return_density,
    )
    val_ds = DrumPatternDataset(
        data_dir=drums_dir,
        split="val",
        add_channel_dim=add_channel_dim,
        dtype=dtype,
        return_label=return_label,
        return_density=return_density,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------- Quick smoke test ----------
if __name__ == "__main__":
    # Run a quick check to avoid runtime surprises.
    _, drums_dir = _default_dirs()
    ds = DrumPatternDataset(drums_dir, split="train")
    x, y, d = ds[0]
    print("x shape:", x.shape)        # [1,16,9]
    print("y:", y.item())             # 0..4
    print("density:", d)              # 0.0~1.0

    dl, vl = get_drum_dataloaders(batch_size=32)
    xb, yb, db = next(iter(dl))
    print("batch x:", xb.shape)       # [32,1,16,9]
    print("batch y:", yb.shape)       # [32]
