"""
02b_preprocessing_bciiv2a.py
============================
BCI Competition IV - Dataset 2a Preprocessing Pipeline
(https://www.bbci.de/competition/iv/#dataset2a)

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework — generalization to motor-imagery BCI

Dataset (manually downloaded into ``archive (1)/`` as MATLAB .mat files)
------------------------------------------------------------------------
- 9 subjects (A01T - A09T), training files only
- 25 channels per subject (22 EEG + 3 EOG); we keep the 22 EEG channels
- Sampling rate: 250 Hz
- Each subject: 6 motor-imagery runs x 48 trials = 288 trials
- 4 balanced classes: 1=left hand, 2=right hand, 3=feet, 4=tongue
- Trial structure: cue at t=2 s, motor imagery 3-6 s; we extract the
  4-second window [2 s, 6 s] following each trial onset

Pipeline
--------
1. Load .mat per subject, concatenate MI runs.
2. Keep first 22 channels (drop EOG).
3. Bandpass 4 - 40 Hz (standard MI band), filtfilt order 4.
4. Slice trials: 1000 samples (4 s) starting 500 samples after cue onset.
5. Drop trials flagged with artifacts.
6. Per-channel z-score using *training-subject* statistics.
7. Subject-wise split: 6 train (A01-A06) / 1 val (A07) / 2 test (A08-A09).
8. Save as outputs/processed_data/bciiv2a_sequences.npz

The four motor-imagery classes are not mapped onto Active, Passive, and
Fatigued. That mapping has no correspondence to left hand, right hand,
feet, or tongue, so the cognitive ODE is not applied to this dataset.
Within-subject classification, which is the usual BCI-IV-2a protocol,
is in 12c_within_subject_mi.py.
"""

import json
import time
from pathlib import Path
import warnings

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

# Paths
BASE_PATH = Path(__file__).parent
DATASET_PATH = Path(__file__).parent / 'archive (1)'
OUTPUT_PATH = BASE_PATH / 'outputs'
PROCESSED_DATA_PATH = OUTPUT_PATH / 'processed_data'
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Acquisition / windowing constants
SAMPLING_RATE = 250
N_EEG_CHANNELS = 22          # drop the trailing 3 EOG channels
TRIAL_LEN_SAMPLES = 1000     # 4 s window
TRIAL_OFFSET_SAMPLES = 500   # start 2 s after cue (cue at trial onset + 2 s)
LOWCUT_HZ = 4.0
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4

CLASS_NAMES = ['left_hand', 'right_hand', 'feet', 'tongue']

# Subject-wise split
TRAIN_SUBJECTS = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T']
VAL_SUBJECTS = ['A07T']
TEST_SUBJECTS = ['A08T', 'A09T']


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply zero-phase butterworth bandpass per channel.

    data: (n_samples, n_channels)
    returns: (n_samples, n_channels)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_subject(mat_path):
    """Load a single subject's .mat and return MI runs.

    Returns
    -------
    runs : list of dict with keys X (samples, 25), trial (n_trials,), y (n_trials,), artifacts (n_trials,)
    """
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    data = m['data'][0]  # shape (n_runs,)
    runs = []
    for cell in data:
        r = cell[0, 0]
        if r.trial.shape[0] == 0:
            continue  # calibration run, no MI trials
        runs.append({
            'X': r.X.astype(np.float32),                       # (n_samples, 25)
            'trial': r.trial.flatten().astype(np.int64),       # trial start indices
            'y': r.y.flatten().astype(np.int64),               # class labels 1..4
            'artifacts': r.artifacts.flatten().astype(np.int64),
            'fs': int(r.fs[0, 0]),
        })
    return runs


def extract_trials(runs):
    """Slice cue-locked windows and apply bandpass per run.

    Returns
    -------
    X : (n_trials, TRIAL_LEN_SAMPLES, N_EEG_CHANNELS) float32
    y : (n_trials,) int64, classes mapped to 0..3
    """
    X_list, y_list = [], []
    for run in runs:
        if run['fs'] != SAMPLING_RATE:
            raise RuntimeError(f"Unexpected sampling rate {run['fs']} != {SAMPLING_RATE}")

        # Bandpass the full run (eeg + eog) - cheaper than per-trial filtering
        sig = bandpass_filter(run['X'], LOWCUT_HZ, HIGHCUT_HZ, SAMPLING_RATE,
                              order=FILTER_ORDER)
        sig = sig[:, :N_EEG_CHANNELS]

        for trial_start, label, artifact in zip(run['trial'], run['y'], run['artifacts']):
            if artifact != 0:
                continue  # drop flagged trial
            start = int(trial_start) + TRIAL_OFFSET_SAMPLES
            end = start + TRIAL_LEN_SAMPLES
            if end > sig.shape[0]:
                continue  # would run past the recording
            seg = sig[start:end]  # (TRIAL_LEN_SAMPLES, N_EEG_CHANNELS)
            X_list.append(seg.astype(np.float32))
            y_list.append(int(label) - 1)  # map 1..4 -> 0..3

    if not X_list:
        return (np.zeros((0, TRIAL_LEN_SAMPLES, N_EEG_CHANNELS), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


def process_subject_group(subject_ids, label):
    """Load + slice + return X, y for a list of subject IDs."""
    print(f"\n--- {label}: {subject_ids} ---")
    X_all, y_all, subject_labels = [], [], []
    for sid in subject_ids:
        mat_path = DATASET_PATH / f"{sid}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        runs = load_subject(mat_path)
        X, y = extract_trials(runs)
        print(f"  {sid}: trials={X.shape[0]:>4d}  "
              f"class counts={np.bincount(y, minlength=4).tolist()}")
        X_all.append(X)
        y_all.append(y)
        subject_labels.append(np.full(len(y), sid))
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(subject_labels, axis=0)
    print(f"  {label} total: X={X.shape}  y={y.shape}  "
          f"classes={np.bincount(y, minlength=4).tolist()}")
    return X, y, subjects


def zscore(X, mean=None, std=None):
    """Per-channel z-score. If mean/std not provided, compute from X.

    X: (N, T, C)
    """
    if mean is None:
        # Channel-wise mean/std across (N, T)
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def main():
    t0 = time.time()
    print("=" * 60)
    print("BCI-IV-2a Preprocessing")
    print("=" * 60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Sampling: {SAMPLING_RATE} Hz | Channels: {N_EEG_CHANNELS} | "
          f"Bandpass: {LOWCUT_HZ}-{HIGHCUT_HZ} Hz")
    print(f"Trial window: {TRIAL_LEN_SAMPLES} samples "
          f"({TRIAL_LEN_SAMPLES/SAMPLING_RATE:.1f} s) starting "
          f"{TRIAL_OFFSET_SAMPLES/SAMPLING_RATE:.1f} s after cue onset")

    X_train, y_train, s_train = process_subject_group(TRAIN_SUBJECTS, "TRAIN")
    X_val, y_val, s_val = process_subject_group(VAL_SUBJECTS, "VAL")
    X_test, y_test, s_test = process_subject_group(TEST_SUBJECTS, "TEST")

    # Copy before the cross-subject z-score. Within-subject folds normalize
    # from their own training trials.
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    subject_all = np.concatenate([s_train, s_val, s_test], axis=0)

    # Z-score using train stats only
    X_train, mean, std = zscore(X_train)
    X_val, _, _ = zscore(X_val, mean, std)
    X_test, _, _ = zscore(X_test, mean, std)

    out_path = PROCESSED_DATA_PATH / 'bciiv2a_sequences.npz'
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_all=X_all, y_all=y_all, subject_all=subject_all,
    )
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    metadata = {
        'dataset': 'BCI-IV-2a',
        'sampling_rate': SAMPLING_RATE,
        'n_channels': N_EEG_CHANNELS,
        'trial_length_samples': TRIAL_LEN_SAMPLES,
        'trial_offset_samples': TRIAL_OFFSET_SAMPLES,
        'lowcut_hz': LOWCUT_HZ,
        'highcut_hz': HIGHCUT_HZ,
        'filter_order': FILTER_ORDER,
        'classes': CLASS_NAMES,
        'n_classes': 4,
        'train_subjects': TRAIN_SUBJECTS,
        'val_subjects': VAL_SUBJECTS,
        'test_subjects': TEST_SUBJECTS,
        'train_shape': list(X_train.shape),
        'val_shape': list(X_val.shape),
        'test_shape': list(X_test.shape),
        'train_class_counts': np.bincount(y_train, minlength=4).tolist(),
        'val_class_counts': np.bincount(y_val, minlength=4).tolist(),
        'test_class_counts': np.bincount(y_test, minlength=4).tolist(),
        'channel_mean_first5': mean.flatten()[:5].tolist(),
        'channel_std_first5': std.flatten()[:5].tolist(),
    }
    with open(PROCESSED_DATA_PATH / 'bciiv2a_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}  "
          f"counts={metadata['train_class_counts']}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}    "
          f"counts={metadata['val_class_counts']}")
    print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}   "
          f"counts={metadata['test_class_counts']}")
    print(f"Done in {(time.time()-t0):.1f} s")


if __name__ == "__main__":
    main()
