"""Memory-mapped sequence access for GPU training.

The compressed sequence file expands to about 19 GB. Copying it into a
second process while the SVM is resident will exhaust RAM. These helpers
read one window at a time from uncompressed .npy files so two GPU jobs
share the same file cache instead of each holding a private copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_PATH = Path(__file__).parent
MMAP_DIR = BASE_PATH / "outputs" / "processed_data" / "mmap"
SPLIT_KEYS = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")


def mmap_ready():
    return all((MMAP_DIR / f"{key}.npy").exists() for key in SPLIT_KEYS)


def load_mmap_splits():
    """Return train/val/test arrays and test subject ids as read-only memmaps."""
    arrays = {
        key: np.load(MMAP_DIR / f"{key}.npy", mmap_mode="r")
        for key in SPLIT_KEYS
    }
    subject_path = MMAP_DIR / "subject_id_test.npy"
    subject_id_test = np.load(subject_path, mmap_mode="r") if subject_path.exists() else None
    return (
        arrays["X_train"], arrays["y_train"],
        arrays["X_val"], arrays["y_val"],
        arrays["X_test"], arrays["y_test"],
        subject_id_test,
    )


class WindowDataset(Dataset):
    """One EEG window per item, copied out of the memmap only for that item."""

    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return int(self.windows.shape[0])

    def __getitem__(self, idx):
        window = np.asarray(self.windows[idx], dtype=np.float32).copy()
        return torch.from_numpy(window), int(self.labels[idx])


def apply_gpu_cap():
    """Limit this process to a fraction of the visible GPU, before any allocation."""
    fraction = os.environ.get("EEG_GPU_FRACTION")
    if not fraction or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)
    print(f"GPU memory cap: {float(fraction):.0%} of the visible device")


def worker_count(default):
    value = os.environ.get("EEG_WORKERS")
    if value is None:
        return default
    return max(0, int(value))
