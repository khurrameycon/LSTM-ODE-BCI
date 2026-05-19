"""
11_deep_baselines.py
====================
Deep Learning Baselines: EEGNet and DeepConvNet
OpenNeuro ds004148: Test-Retest Resting and Cognitive State EEG Dataset

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

This script provides two widely used EEG deep-learning baselines, trained
on exactly the same train/val/test split produced by 02_preprocessing.py
(processed_sequences.npz) and evaluated with the same metrics as the LSTM
model in 04_lstm_model.py.

Models
------
1. EEGNet     - Lawhern et al. (2018)  https://doi.org/10.1088/1741-2552/aace8c
2. DeepConvNet - Schirrmeister et al. (2017)  https://doi.org/10.1002/hbm.23730

Outputs
-------
- outputs/models/eegnet.pt, deepconvnet.pt
- outputs/results/deep_baselines_results.json
- outputs/figures/fig26_deep_baselines_training_history.png
- outputs/figures/fig27_deep_baselines_confusion_matrices.png
- outputs/figures/fig28_deep_baselines_roc_curves.png
- outputs/figures/fig29_deep_baselines_comparison.png
"""

import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend - avoids Qt/xcb plugin failures
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Paths (match 04_lstm_model.py layout)
BASE_PATH = Path(__file__).parent.parent
PROCESSED_DATA_PATH = BASE_PATH / 'outputs' / 'processed_data'
OUTPUT_PATH = BASE_PATH / 'outputs'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'
for p in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018).

    Compact CNN designed for EEG.  Input shape expected by forward is
    (batch, time, channels) - matching the data layout in processed_sequences.npz -
    and is internally reshaped to (batch, 1, channels, time).
    """

    def __init__(self, n_channels=61, n_samples=256, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: temporal conv + depthwise spatial conv
        self.conv_temporal = nn.Conv2d(
            1, F1, kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn_temporal = nn.BatchNorm2d(F1)

        self.conv_depthwise = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1),
            groups=F1, bias=False
        )
        self.bn_depthwise = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise + pointwise)
        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn_separable = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier - dimension determined by a dry forward pass
        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self._features(dummy)
            return x.numel()

    def _features(self, x):
        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        x = self.conv_depthwise(x)
        x = self.bn_depthwise(x)
        x = torch.nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn_separable(x)
        x = torch.nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        # x: (B, T, C) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1)
        x = self._features(x)
        x = x.flatten(1)
        return self.classifier(x)


class DeepConvNet(nn.Module):
    """DeepConvNet / Deep4Net (Schirrmeister et al., 2017).

    Adapted to T=256 by using 'same' padding for temporal convs so the
    four-block depth still fits.  Spatial reduction happens through max-pool.
    """

    def __init__(self, n_channels=61, n_samples=256, n_classes=2, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: split conv (temporal + spatial), as in the original paper
        self.conv_time = nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.conv_spatial = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout)

        # Block 2
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(dropout)

        # Block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(dropout)

        # Block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(dropout)

        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            return self._features(dummy).numel()

    def _features(self, x):
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.bn1(x)
        x = torch.nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x); x = self.bn2(x); x = torch.nn.functional.elu(x)
        x = self.pool2(x); x = self.drop2(x)

        x = self.conv3(x); x = self.bn3(x); x = torch.nn.functional.elu(x)
        x = self.pool3(x); x = self.drop3(x)

        x = self.conv4(x); x = self.bn4(x); x = torch.nn.functional.elu(x)
        x = self.pool4(x); x = self.drop4(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1)
        x = self._features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Load preprocessed sequence data - identical logic to 04_lstm_model.py."""
    print("=" * 60)
    print("Loading Processed Data")
    print("=" * 60)

    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    if 'X_val' in data.files and len(data['X_val']) > 0:
        X_val = data['X_val']
        y_val = data['y_val']
    else:
        n_val = int(len(X_train) * 0.15)
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
        print("Created validation set from training data (15%)")

    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Channels: {X_train.shape[2]}, Sequence length: {X_train.shape[1]}")
    print(f"Class balance - Train: {np.mean(y_train):.1%}, "
          f"Val: {np.mean(y_val):.1%}, Test: {np.mean(y_test):.1%}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size=128):
    pin = DEVICE.type == 'cuda'
    train = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=pin, drop_last=False,
    )
    val = DataLoader(
        TensorDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va)),
        batch_size=batch_size * 4, shuffle=False, pin_memory=pin,
    )
    test = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=batch_size * 4, shuffle=False, pin_memory=pin,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, y_train, model_name,
                epochs=50, lr=1e-3, weight_decay=1e-4, patience=10):
    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    model = model.to(DEVICE)

    # Class-weighted loss to handle imbalance (same idea as LSTM script)
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(
        len(y_train) / (len(class_counts) * class_counts)
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'val_f1': []}

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        # ----- train -----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ----- val -----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_preds, v_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(X)
                loss = criterion(logits, y)
                v_loss += loss.item() * X.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += X.size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_true.extend(y.cpu().numpy())

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_f1 = f1_score(v_true, v_preds, zero_division=0)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")
    return model, history


def evaluate(model, test_loader, y_test, model_name):
    print(f"\nEvaluating {model_name}")
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(DEVICE, non_blocking=True)
            logits = model(X)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
    preds, probs = np.array(preds), np.array(probs)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    mcc = matthews_corrcoef(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Bootstrap CI on accuracy (same as 04_lstm_model.py)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_test), len(y_test))
        boot.append(accuracy_score(y_test[idx], preds[idx]))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    print(f"  Accuracy:  {acc:.4f}  (95% CI [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc if not np.isnan(auc) else float('nan'):.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': None if np.isnan(auc) else float(auc),
        'mcc': float(mcc),
        'accuracy_ci_low': float(ci_low),
        'accuracy_ci_high': float(ci_high),
        'confusion_matrix': cm.tolist(),
        'y_pred': preds.tolist(),
        'y_prob': probs.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_histories(histories, output_path):
    n = len(histories)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (name, h) in enumerate(histories.items()):
        ax = axes[i, 0]
        ax.plot(h['train_loss'], label='Train', color='C0')
        ax.plot(h['val_loss'], label='Val', color='C3')
        ax.set_title(f'{name} — Loss', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[i, 1]
        ax.plot(h['train_acc'], label='Train', color='C0')
        ax.plot(h['val_acc'], label='Val', color='C3')
        ax.set_title(f'{name} — Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[i, 2]
        ax.plot(h['val_f1'], label='Val F1', color='C2')
        ax.set_title(f'{name} — Val F1', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'fig26_deep_baselines_training_history.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig26_deep_baselines_training_history.pdf',
                bbox_inches='tight')
    plt.close()
    print("Saved: fig26_deep_baselines_training_history.png/pdf")


def plot_confusion_matrices(results, output_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    class_names = ['Eyes Open', 'Eyes Closed']
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, ax=ax, annot_kws={'size': 13})
        ax.set_title(f'{name}\nAcc={r["accuracy"]:.3f}  F1={r["f1"]:.3f}',
                     fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path / 'fig27_deep_baselines_confusion_matrices.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig27_deep_baselines_confusion_matrices.pdf',
                bbox_inches='tight')
    plt.close()
    print("Saved: fig27_deep_baselines_confusion_matrices.png/pdf")


def plot_roc_curves(results, y_test, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        if r['auc'] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Deep Baselines', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'fig28_deep_baselines_roc_curves.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig28_deep_baselines_roc_curves.pdf',
                bbox_inches='tight')
    plt.close()
    print("Saved: fig28_deep_baselines_roc_curves.png/pdf")


def plot_full_comparison(results, output_path):
    """Bar chart comparing new deep baselines against any prior results on disk."""
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name, 'Accuracy': r['accuracy'], 'F1': r['f1'],
            'AUC': r['auc'] if r['auc'] is not None else 0.0,
        })

    # Pick up prior baselines / LSTM results if present
    prior_files = {
        'Baselines (SVM/RF/XGB)': RESULTS_PATH / 'baseline_results.json',
        'LSTM-Attention': RESULTS_PATH / 'lstm_results.json',
    }
    for label, path in prior_files.items():
        if not path.exists():
            continue
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            continue
        # baseline_results.json may store dict of {model: metrics}
        if isinstance(blob, dict) and any(isinstance(v, dict) for v in blob.values()):
            for sub_name, sub in blob.items():
                if not isinstance(sub, dict): continue
                if 'accuracy' not in sub: continue
                rows.append({
                    'Model': sub_name,
                    'Accuracy': sub.get('accuracy', 0.0),
                    'F1': sub.get('f1', 0.0),
                    'AUC': sub.get('auc') or 0.0,
                })
        elif isinstance(blob, dict) and 'accuracy' in blob:
            rows.append({
                'Model': blob.get('model_name', label),
                'Accuracy': blob.get('accuracy', 0.0),
                'F1': blob.get('f1', 0.0),
                'AUC': blob.get('auc') or 0.0,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=['Model'])
    df.to_csv(RESULTS_PATH / 'all_models_comparison.csv', index=False)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df)), 6))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['Accuracy'], w, label='Accuracy', color='#4C72B0')
    ax.bar(x,     df['F1'],       w, label='F1',       color='#55A868')
    ax.bar(x + w, df['AUC'],      w, label='AUC',      color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=20, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Trained Models', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(i - w, row['Accuracy'] + 0.01, f"{row['Accuracy']:.2f}",
                ha='center', fontsize=8)
        ax.text(i,     row['F1']       + 0.01, f"{row['F1']:.2f}",
                ha='center', fontsize=8)
        ax.text(i + w, row['AUC']      + 0.01, f"{row['AUC']:.2f}",
                ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / 'fig29_deep_baselines_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig29_deep_baselines_comparison.pdf',
                bbox_inches='tight')
    plt.close()
    print("Saved: fig29_deep_baselines_comparison.png/pdf")
    print(f"Saved: all_models_comparison.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    n_channels = X_train.shape[2]
    n_samples = X_train.shape[1]

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    models_to_run = {
        'EEGNet':       EEGNet(n_channels=n_channels, n_samples=n_samples),
        'DeepConvNet':  DeepConvNet(n_channels=n_channels, n_samples=n_samples),
    }

    histories = {}
    results = {}
    for name, model in models_to_run.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{name} parameters: {n_params:,}")

        model, history = train_model(
            model, train_loader, val_loader, y_train, name,
            epochs=50, lr=1e-3, weight_decay=1e-4, patience=10,
        )
        histories[name] = history

        r = evaluate(model, test_loader, y_test, name)
        results[name] = r

        # Save weights + per-model results
        ckpt_path = MODELS_PATH / f"{name.lower()}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': name,
            'n_channels': n_channels,
            'n_samples': n_samples,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        with open(RESULTS_PATH / f"{name.lower()}_results.json", 'w') as f:
            json.dump({**r, 'history': history}, f, indent=2)

    # Combined results
    with open(RESULTS_PATH / 'deep_baselines_results.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk not in ('y_pred', 'y_prob')} for k, v in results.items()},
                  f, indent=2)

    # Plots
    plot_training_histories(histories, FIGURES_PATH)
    plot_confusion_matrices(results, FIGURES_PATH)
    plot_roc_curves(results, y_test, FIGURES_PATH)
    plot_full_comparison(results, FIGURES_PATH)

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame([
        {'Model': k, 'Accuracy': v['accuracy'], 'F1': v['f1'],
         'AUC': v['auc'], 'MCC': v['mcc']}
        for k, v in results.items()
    ])
    print(summary.to_string(index=False))
    print("\nAll outputs written to:")
    print(f"  - {MODELS_PATH}")
    print(f"  - {RESULTS_PATH}")
    print(f"  - {FIGURES_PATH}")


if __name__ == "__main__":
    main()
