"""
12d_bciiv2b.py
=============
Classification-only EEGNet on BCI Competition IV, Dataset 2b.

Left hand and right hand are not assigned to Active, Passive, or
Fatigued. The cognitive ODE is not run.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'bciiv2b'
RESULTS = BASE_PATH / 'outputs' / 'results'
PROCESSED = BASE_PATH / 'outputs' / 'processed_data'
RESULTS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 250
N_EEG_CHANNELS = 3
CUE_OFFSET_SAMPLES = 750   # cue at 3.0 s after trial onset
TRIAL_LEN_SAMPLES = 1000   # 4.0 s from the cue
LOWCUT_HZ = 4.0
HIGHCUT_HZ = 40.0
SUBJECTS = [f'B0{i}' for i in range(1, 10)]


def _eegnet():
    import importlib.util
    path = BASE_PATH / '12_deep_baselines_bciiv2a.py'
    spec = importlib.util.spec_from_file_location('bci_baselines', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EEGNet, module.DEVICE


def _bandpass(data):
    nyq = 0.5 * SAMPLING_RATE
    b, a = butter(4, [LOWCUT_HZ / nyq, HIGHCUT_HZ / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def _load_file(path):
    saved = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
    windows, labels = [], []
    for cell in saved['data'][0]:
        run = cell[0, 0]
        if run.trial.size == 0:
            continue
        if int(np.array(run.fs).flatten()[0]) != SAMPLING_RATE:
            raise RuntimeError(f'Unexpected sampling rate in {path.name}')
        signal = _bandpass(run.X.astype(np.float64))[:, :N_EEG_CHANNELS]
        for trial_start, label, artifact in zip(
            run.trial.flatten(), run.y.flatten(), run.artifacts.flatten(),
        ):
            if int(artifact) != 0:
                continue
            start = int(trial_start) - 1 + CUE_OFFSET_SAMPLES
            end = start + TRIAL_LEN_SAMPLES
            if start < 0 or end > signal.shape[0]:
                continue
            windows.append(signal[start:end].astype(np.float32))
            labels.append(int(label) - 1)
    if not windows:
        empty = np.zeros((0, TRIAL_LEN_SAMPLES, N_EEG_CHANNELS), np.float32)
        return empty, np.zeros((0,), np.int64)
    return np.stack(windows), np.array(labels, dtype=np.int64)


def _fold_zscore(X_train, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _train_one(model, X_train, y_train, X_test, y_test, device, n_classes, epochs=30):
    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.FloatTensor(len(y_train) / (n_classes * np.maximum(counts, 1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    return {
        'accuracy': float(accuracy_score(y_test, prediction)),
        'macro_f1': float(f1_score(y_test, prediction, average='macro', zero_division=0)),
    }


def _score_subject(EEGNet, device, X, y, n_classes):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, test_idx in splitter.split(X, y):
        X_fit, X_hold = _fold_zscore(X[train_idx], X[test_idx])
        model = EEGNet(
            n_channels=X.shape[2], n_samples=X.shape[1], n_classes=n_classes,
        ).to(device)
        fold_scores.append(_train_one(
            model, X_fit, y[train_idx], X_hold, y[test_idx], device, n_classes,
        ))
        del model
    return {
        'mean_accuracy': float(np.mean([item['accuracy'] for item in fold_scores])),
        'mean_macro_f1': float(np.mean([item['macro_f1'] for item in fold_scores])),
        'n_trials': int(len(y)),
    }


def main():
    missing = [name for name in SUBJECTS if not (DATA_PATH / f'{name}T.mat').exists()]
    if missing:
        print('BCI-IV-2b training files are missing:', ', '.join(missing))
        return

    EEGNet, device = _eegnet()
    print(f'Using device: {device}')
    rows = []
    session_rows = []
    pool_x, pool_y, pool_subject, pool_session = [], [], [], []

    for subject in SUBJECTS:
        parts = {}
        for session in ('T', 'E'):
            X, y = _load_file(DATA_PATH / f'{subject}{session}.mat')
            parts[session] = (X, y)
            if len(y):
                pool_x.append(X)
                pool_y.append(y)
                pool_subject.append(np.full(len(y), subject))
                pool_session.append(np.full(len(y), session))
        X = np.concatenate([parts['T'][0], parts['E'][0]], axis=0)
        y = np.concatenate([parts['T'][1], parts['E'][1]], axis=0)
        if len(np.unique(y)) < 2 or len(y) < 20:
            print(f'{subject}: skipped, trials={len(y)}')
            continue
        scored = _score_subject(EEGNet, device, X, y, n_classes=2)
        scored['subject'] = subject
        rows.append(scored)
        print(f"{subject}: accuracy={scored['mean_accuracy']:.3f} trials={scored['n_trials']}")

        X_fit, y_fit = parts['T']
        X_hold, y_hold = parts['E']
        if len(np.unique(y_fit)) >= 2 and len(y_hold) >= 10:
            X_fit, X_hold = _fold_zscore(X_fit, X_hold)
            model = EEGNet(n_channels=3, n_samples=TRIAL_LEN_SAMPLES, n_classes=2).to(device)
            transferred = _train_one(model, X_fit, y_fit, X_hold, y_hold, device, n_classes=2)
            del model
            transferred['subject'] = subject
            transferred['n_train'] = int(len(y_fit))
            transferred['n_test'] = int(len(y_hold))
            session_rows.append(transferred)
            print(f"{subject}: session-transfer accuracy={transferred['accuracy']:.3f}")

    np.savez(
        PROCESSED / 'bciiv2b_sequences.npz',
        X_all=np.concatenate(pool_x, axis=0),
        y_all=np.concatenate(pool_y, axis=0),
        subject_all=np.concatenate(pool_subject, axis=0),
        session_all=np.concatenate(pool_session, axis=0),
    )
    report = {
        'dataset': 'BCI Competition IV Dataset 2b',
        'classes': ['left_hand', 'right_hand'],
        'chance': 0.5,
        'channels': 'C3, Cz, C4',
        'window': '4.0 s starting at the cue (3.0 s after trial onset)',
        'bandpass_hz': [LOWCUT_HZ, HIGHCUT_HZ],
        'artifacts': 'trials flagged in the file were dropped',
        'ode_applied': False,
        'reason': (
            'Active, Passive, and Fatigued have no correspondence to '
            'left hand and right hand.'
        ),
        'within_subject': {
            'protocol': '5-fold within-subject EEGNet, sessions T and E pooled',
            'subjects': rows,
            'mean_accuracy': float(np.mean([row['mean_accuracy'] for row in rows])) if rows else None,
            'mean_macro_f1': float(np.mean([row['mean_macro_f1'] for row in rows])) if rows else None,
        },
        'session_transfer': {
            'protocol': 'train on session file T, test on session file E',
            'subjects': session_rows,
            'mean_accuracy': float(np.mean([row['accuracy'] for row in session_rows])) if session_rows else None,
        },
    }
    with open(RESULTS / 'bciiv2b_within_subject.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print('Saved outputs/results/bciiv2b_within_subject.json')
    print(f"within-subject mean={report['within_subject']['mean_accuracy']}")
    print(f"session-transfer mean={report['session_transfer']['mean_accuracy']}")


if __name__ == '__main__':
    main()
