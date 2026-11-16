"""
12c_within_subject_mi.py
========================
Within-subject EEGNet on BCI Competition IV, Dataset 2a.

Left hand, right hand, feet, and tongue are not assigned to Active,
Passive, or Fatigued. The cognitive ODE is not run. The cross-subject
split in 12_deep_baselines_bciiv2a.py remains available; this script is
the within-subject protocol used by the BCI-IV-2a literature.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

BASE_PATH = Path(__file__).parent
PROCESSED = BASE_PATH / 'outputs' / 'processed_data' / 'bciiv2a_sequences.npz'
RESULTS = BASE_PATH / 'outputs' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)


def _eegnet():
    import importlib.util
    path = BASE_PATH / '12_deep_baselines_bciiv2a.py'
    spec = importlib.util.spec_from_file_location('bci_baselines', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EEGNet, module.DEVICE


def _fold_zscore(X_train, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _train_one(model, X_train, y_train, X_test, y_test, device, epochs=30):
    counts = np.bincount(y_train, minlength=4)
    weights = torch.FloatTensor(len(y_train) / (4 * np.maximum(counts, 1))).to(device)
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
        logits = model(torch.FloatTensor(X_test).to(device))
        prediction = logits.argmax(dim=1).cpu().numpy()
    return {
        'accuracy': float(accuracy_score(y_test, prediction)),
        'macro_f1': float(f1_score(y_test, prediction, average='macro', zero_division=0)),
    }


def main():
    if not PROCESSED.exists():
        print('BCI-IV-2a sequences were not found.')
        print('Place A01T.mat ... A09T.mat in "archive (1)/" and run:')
        print('  python 02b_preprocessing_bciiv2a.py')
        print('The cognitive ODE is intentionally not applied to motor imagery.')
        return

    saved = np.load(PROCESSED, allow_pickle=True)
    if 'subject_all' not in saved.files and 'subject_id' not in saved.files:
        print('bciiv2a_sequences.npz has no per-trial subject id.')
        print('Re-run 02b_preprocessing_bciiv2a.py after this revision.')
        return

    EEGNet, device = _eegnet()
    X = saved['X_train']
    # The published split already separated subjects. Within-subject CV needs
    # every trial, so the preprocessing file must also store the unsplit pool.
    if 'X_all' in saved.files:
        X = saved['X_all']
        y = saved['y_all']
        subjects = saved['subject_all']
    else:
        print('The saved 2a file is the cross-subject split only.')
        print('Within-subject folds need X_all, which 02b now writes when re-run.')
        return

    rows = []
    for subject in dict.fromkeys(subjects.tolist()):
        mask = subjects == subject
        X_subject = X[mask]
        y_subject = y[mask]
        if len(np.unique(y_subject)) < 2 or len(y_subject) < 20:
            continue
        fold_scores = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in splitter.split(X_subject, y_subject):
            X_fit, X_hold = _fold_zscore(X_subject[train_idx], X_subject[test_idx])
            model = EEGNet(
                n_channels=X_subject.shape[2],
                n_samples=X_subject.shape[1],
                n_classes=4,
            ).to(device)
            fold_scores.append(_train_one(
                model, X_fit, y_subject[train_idx],
                X_hold, y_subject[test_idx], device,
            ))
            del model
        rows.append({
            'subject': str(subject),
            'mean_accuracy': float(np.mean([item['accuracy'] for item in fold_scores])),
            'mean_macro_f1': float(np.mean([item['macro_f1'] for item in fold_scores])),
        })
        print(f"{subject}: accuracy={rows[-1]['mean_accuracy']:.3f}")

    report = {
        'protocol': '5-fold within-subject EEGNet',
        'ode_applied': False,
        'reason': (
            'Active, Passive, and Fatigued have no correspondence to '
            'left hand, right hand, feet, and tongue.'
        ),
        'subjects': rows,
        'mean_accuracy': float(np.mean([row['mean_accuracy'] for row in rows])) if rows else None,
    }
    with open(RESULTS / 'bciiv2a_within_subject.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print('Saved outputs/results/bciiv2a_within_subject.json')


if __name__ == '__main__':
    main()
