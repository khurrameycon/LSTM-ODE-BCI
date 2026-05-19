"""
12_deep_baselines_bciiv2a.py
============================
Deep Learning Baselines for BCI Competition IV - Dataset 2a (4-class MI)

Authors: Ayesha, Muhammad Khurram Umair

Same architectures as 11_deep_baselines.py (EEGNet, DeepConvNet) but
configured for the BCI-IV-2a motor-imagery task:
    n_channels = 22, n_samples = 1000, n_classes = 4

Inputs:  outputs/processed_data/bciiv2a_sequences.npz (from 02b_preprocessing_bciiv2a.py)
Outputs:
- outputs/models/bciiv2a_eegnet.pt, bciiv2a_deepconvnet.pt
- outputs/results/bciiv2a_deep_baselines_results.json
- outputs/figures/fig30_bciiv2a_training_history.png
- outputs/figures/fig31_bciiv2a_confusion_matrices.png
- outputs/figures/fig32_bciiv2a_roc_curves.png
- outputs/figures/fig33_bciiv2a_comparison.png
"""

import sys
import time
import json
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
from sklearn.preprocessing import label_binarize

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BASE_PATH = Path(__file__).parent.parent
PROCESSED_DATA_PATH = BASE_PATH / 'outputs' / 'processed_data'
OUTPUT_PATH = BASE_PATH / 'outputs'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'
for p in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
N_CLASSES = 4


# ---------------------------------------------------------------------------
# Models (re-import from 11_deep_baselines? we keep self-contained for clarity)
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_samples=1000, n_classes=4,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

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

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn_separable = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self):
        with torch.no_grad():
            return self._features(torch.zeros(1, 1, self.n_channels, self.n_samples)).numel()

    def _features(self, x):
        x = self.conv_temporal(x); x = self.bn_temporal(x)
        x = self.conv_depthwise(x); x = self.bn_depthwise(x)
        x = torch.nn.functional.elu(x); x = self.pool1(x); x = self.drop1(x)
        x = self.conv_separable_depth(x); x = self.conv_separable_point(x)
        x = self.bn_separable(x); x = torch.nn.functional.elu(x)
        x = self.pool2(x); x = self.drop2(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1)
        return self.classifier(self._features(x).flatten(1))


class DeepConvNet(nn.Module):
    def __init__(self, n_channels=22, n_samples=1000, n_classes=4, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.conv_time = nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.conv_spatial = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(dropout)

        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self):
        with torch.no_grad():
            return self._features(torch.zeros(1, 1, self.n_channels, self.n_samples)).numel()

    def _features(self, x):
        x = self.conv_time(x); x = self.conv_spatial(x); x = self.bn1(x)
        x = torch.nn.functional.elu(x); x = self.pool1(x); x = self.drop1(x)
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
        return self.classifier(self._features(x).flatten(1))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    print("=" * 60)
    print("Loading BCI-IV-2a Processed Data")
    print("=" * 60)
    data = np.load(PROCESSED_DATA_PATH / 'bciiv2a_sequences.npz')
    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']
    print(f"X_train: {X_tr.shape}  classes: {np.bincount(y_tr).tolist()}")
    print(f"X_val:   {X_va.shape}  classes: {np.bincount(y_va).tolist()}")
    print(f"X_test:  {X_te.shape}  classes: {np.bincount(y_te).tolist()}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size=64):
    pin = DEVICE.type == 'cuda'
    train = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=pin,
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
                epochs=100, lr=1e-3, weight_decay=1e-3, patience=20):
    print("\n" + "=" * 60)
    print(f"Training {model_name} (BCI-IV-2a)")
    print("=" * 60)
    model = model.to(DEVICE)

    counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = torch.FloatTensor(len(y_train) / (N_CLASSES * counts)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf'); best_state = None; patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        rl, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            rl += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)
        train_loss = rl / total; train_acc = correct / total

        model.eval()
        vl, vc, vt = 0.0, 0, 0; vp, vy = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(X)
                loss = criterion(logits, y)
                vl += loss.item() * X.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += X.size(0)
                vp.extend(logits.argmax(1).cpu().numpy())
                vy.extend(y.cpu().numpy())
        val_loss = vl / vt; val_acc = vc / vt
        val_f1 = f1_score(vy, vp, average='macro', zero_division=0)
        scheduler.step()

        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
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
    print(f"\nEvaluating {model_name} on BCI-IV-2a test set")
    model.eval()
    preds = []; probs = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(DEVICE, non_blocking=True)
            logits = model(X)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy())
    preds = np.array(preds)
    probs = np.array(probs)  # (N, n_classes)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=list(range(N_CLASSES)))

    # Macro-OvR AUC
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    try:
        auc_macro = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc_macro = float('nan')

    # Per-class AUC (also for ROC plotting)
    auc_per_class = {}
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            auc_per_class[CLASS_NAMES[c]] = None
        else:
            auc_per_class[CLASS_NAMES[c]] = float(roc_auc_score(y_bin[:, c], probs[:, c]))

    # Bootstrap CI on accuracy
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_test), len(y_test))
        boot.append(accuracy_score(y_test[idx], preds[idx]))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    chance = 1.0 / N_CLASSES
    print(f"  Accuracy:    {acc:.4f}  (chance {chance:.3f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Macro Prec:  {prec:.4f}")
    print(f"  Macro Recall:{rec:.4f}")
    print(f"  Macro F1:    {f1:.4f}")
    print(f"  Macro AUC:   {auc_macro if not np.isnan(auc_macro) else float('nan'):.4f}")
    print(f"  MCC:         {mcc:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'auc_macro': None if np.isnan(auc_macro) else float(auc_macro),
        'auc_per_class': auc_per_class,
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
    if n == 1: axes = axes.reshape(1, -1)
    for i, (name, h) in enumerate(histories.items()):
        ax = axes[i, 0]
        ax.plot(h['train_loss'], label='Train', color='C0')
        ax.plot(h['val_loss'], label='Val', color='C3')
        ax.set_title(f'{name} (BCI-IV-2a) — Loss', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(alpha=0.3)
        ax = axes[i, 1]
        ax.plot(h['train_acc'], label='Train', color='C0')
        ax.plot(h['val_acc'], label='Val', color='C3')
        ax.axhline(0.25, color='gray', linestyle=':', label='Chance (25%)')
        ax.set_title(f'{name} — Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(alpha=0.3)
        ax = axes[i, 2]
        ax.plot(h['val_f1'], label='Macro F1', color='C2')
        ax.set_title(f'{name} — Val Macro F1', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'fig30_bciiv2a_training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig30_bciiv2a_training_history.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig30_bciiv2a_training_history.png/pdf")


def plot_confusion_matrices(results, output_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r['confusion_matrix'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar=False, ax=ax, annot_kws={'size': 11},
                    vmin=0, vmax=1)
        ax.set_title(f'{name}\nAcc={r["accuracy"]:.3f}  Macro F1={r["f1_macro"]:.3f}',
                     fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for tick in ax.get_xticklabels(): tick.set_rotation(30)
    plt.tight_layout()
    plt.savefig(output_path / 'fig31_bciiv2a_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig31_bciiv2a_confusion_matrices.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig31_bciiv2a_confusion_matrices.png/pdf")


def plot_roc_curves(results, y_test, output_path):
    """One-vs-rest ROC curve per class per model."""
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), sharey=True)
    if n == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        probs = np.array(r['y_prob'])
        for c in range(N_CLASSES):
            if y_bin[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
            auc_c = roc_auc_score(y_bin[:, c], probs[:, c])
            ax.plot(fpr, tpr, lw=2, label=f"{CLASS_NAMES[c]} (AUC={auc_c:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        macro_str = f", Macro={r['auc_macro']:.3f}" if r['auc_macro'] is not None else ""
        ax.set_title(f'{name} — ROC (OvR{macro_str})', fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'fig32_bciiv2a_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig32_bciiv2a_roc_curves.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig32_bciiv2a_roc_curves.png/pdf")


def plot_comparison(results, output_path):
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Accuracy': r['accuracy'],
            'Macro F1': r['f1_macro'],
            'Macro AUC': r['auc_macro'] if r['auc_macro'] is not None else 0.0,
            'MCC': r['mcc'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH / 'bciiv2a_comparison.csv', index=False)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(df)), 6))
    metrics = ['Accuracy', 'Macro F1', 'Macro AUC', 'MCC']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    w = 0.2
    x = np.arange(len(df))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        offset = (i - (len(metrics) - 1) / 2) * w
        ax.bar(x + offset, df[m], w, label=m, color=c)
        for j, v in enumerate(df[m]):
            ax.text(x[j] + offset, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)
    ax.axhline(0.25, color='gray', linestyle=':', label='Chance (25%)', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(df['Model'])
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    ax.set_title('BCI-IV-2a — Deep Baselines Comparison', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'fig33_bciiv2a_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig33_bciiv2a_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig33_bciiv2a_comparison.png/pdf, bciiv2a_comparison.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()
    n_channels = X_tr.shape[2]
    n_samples = X_tr.shape[1]

    train_loader, val_loader, test_loader = make_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size=64
    )

    models = {
        'EEGNet':      EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=N_CLASSES),
        'DeepConvNet': DeepConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=N_CLASSES),
    }

    histories, results = {}, {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{name} parameters: {n_params:,}")

        model, history = train_model(
            model, train_loader, val_loader, y_tr, name,
            epochs=100, lr=1e-3, weight_decay=1e-3, patience=20,
        )
        histories[name] = history

        r = evaluate(model, test_loader, y_te, name)
        results[name] = r

        ckpt = MODELS_PATH / f"bciiv2a_{name.lower()}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': name,
            'dataset': 'BCI-IV-2a',
            'n_channels': n_channels,
            'n_samples': n_samples,
            'n_classes': N_CLASSES,
        }, ckpt)
        print(f"Saved checkpoint: {ckpt}")

        with open(RESULTS_PATH / f"bciiv2a_{name.lower()}_results.json", 'w') as f:
            json.dump({**r, 'history': history}, f, indent=2)

    with open(RESULTS_PATH / 'bciiv2a_deep_baselines_results.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk not in ('y_pred', 'y_prob')} for k, v in results.items()},
                  f, indent=2)

    plot_training_histories(histories, FIGURES_PATH)
    plot_confusion_matrices(results, FIGURES_PATH)
    plot_roc_curves(results, y_te, FIGURES_PATH)
    plot_comparison(results, FIGURES_PATH)

    print("\n" + "=" * 60)
    print("BCI-IV-2a SUMMARY (chance = 25%)")
    print("=" * 60)
    summary = pd.DataFrame([
        {'Model': k, 'Accuracy': v['accuracy'], 'Macro F1': v['f1_macro'],
         'Macro AUC': v['auc_macro'], 'MCC': v['mcc']}
        for k, v in results.items()
    ])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
