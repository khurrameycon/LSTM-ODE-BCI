"""
04_lstm_model.py
================
Enhanced LSTM Model with Attention for EEG Eye State Classification

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Improvements:
- Weighted loss for class imbalance
- Deeper architecture with residual connections
- Multi-head attention mechanism
- Learning rate warmup and cosine annealing
- Data augmentation for robustness
"""

import sys
import time
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Paths
BASE_PATH = Path(__file__).parent.parent
PROCESSED_DATA_PATH = BASE_PATH / 'outputs' / 'processed_data'
OUTPUT_PATH = BASE_PATH / 'outputs'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'

for path in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# 61 EEG channels from OpenNeuro ds004148 dataset (Brain Products)
EEG_CHANNELS = [
    'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7',
    'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9',
    'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1',
    'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8',
    'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2'
]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, hidden_size, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.out(context)

        # Average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)

        return output, avg_attention


class Attention(nn.Module):
    """Simple self-attention mechanism for sequence data."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        attention_weights = attention_weights.squeeze(-1)
        return context, attention_weights


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, hidden_size, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class EnhancedLSTMModel(nn.Module):
    """Enhanced Bidirectional LSTM with Multi-Head Attention.

    Architecture:
    - Input embedding layer
    - Stacked Bidirectional LSTM layers
    - Multi-head attention
    - Residual classification head
    """

    def __init__(self, input_size=14, hidden_size=128, num_layers=3,
                 num_classes=2, dropout=0.4, bidirectional=True, num_heads=4):
        super(EnhancedLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention
        lstm_output_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.attention = Attention(lstm_output_size)

        # Classification head with residual blocks
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        # Input projection
        x = self.input_proj(x)

        # LSTM
        lstm_output, _ = self.lstm(x)
        lstm_output = self.layer_norm(lstm_output)

        # Attention pooling
        context, attention_weights = self.attention(lstm_output)

        # Classification
        output = self.classifier(context)

        if return_attention:
            return output, attention_weights
        return output


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def load_data():
    """Load preprocessed sequence data from BIDS format dataset."""
    print("=" * 60)
    print("Loading Processed Data")
    print("=" * 60)

    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Check if validation set exists and has data (may be empty for single-subject runs)
    if 'X_val' in data.files and len(data['X_val']) > 0:
        X_val = data['X_val']
        y_val = data['y_val']
    else:
        # Split training data to create validation set (15% of training)
        n_val = int(len(X_train) * 0.15)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        print("Created validation set from training data (15%)")

    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Number of EEG channels: {X_train.shape[2]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Class balance - Train: {np.mean(y_train):.1%}, Val: {np.mean(y_val):.1%}, Test: {np.mean(y_test):.1%}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def augment_data(X, y, noise_factor=0.05, time_shift_max=5):
    """Simple data augmentation for EEG sequences."""
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        # Original
        X_aug.append(X[i])
        y_aug.append(y[i])

        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, X[i].shape)
        X_aug.append(X[i] + noise)
        y_aug.append(y[i])

        # Time shift
        shift = np.random.randint(-time_shift_max, time_shift_max + 1)
        if shift != 0:
            shifted = np.roll(X[i], shift, axis=0)
            X_aug.append(shifted)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


def create_weighted_dataloader(X, y, batch_size=32):
    """Create dataloader with class-weighted sampling."""
    # Compute class weights
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True
    )

    # Create dataset and loader
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                        batch_size=512, val_batch_size=1024):
    """Create PyTorch dataloaders with GPU optimization.

    Optimized for RTX 3090 (24GB VRAM):
    - Training batch: 512 (with gradient accumulation for effective 2048)
    - Validation/test batch: 1024 (no gradients needed)
    - Multi-worker data loading with pinned memory
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Weighted sampling for training
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[y] for y in y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )

    # Optimized DataLoader settings for GPU
    # Note: On Windows, num_workers>0 can cause issues, so we check platform
    import platform
    if platform.system() == 'Windows':
        num_workers = 0  # Windows multiprocessing issues
        persistent_workers = False
    else:
        num_workers = 8
        persistent_workers = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, y_train, epochs=100,
                learning_rate=3e-4, patience=15, weight_decay=1e-4,
                warmup_epochs=5, gradient_accumulation_steps=4):
    """Train the LSTM model with GPU optimization.

    Features:
    - Mixed precision training (FP16) for faster computation
    - Gradient accumulation for effective batch size 2048
    - Learning rate warmup + cosine annealing
    - Gradient clipping for stability
    """
    print("\n" + "=" * 60)
    print("Training Enhanced LSTM Model (GPU Optimized)")
    print("=" * 60)
    print(f"  Mixed Precision: Enabled")
    print(f"  Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"  Warmup Epochs: {warmup_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Early Stopping Patience: {patience}")
    print("=" * 60)

    start_time = time.time()

    # Compute class weights for loss
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    # Learning rate scheduler: Linear warmup + Cosine annealing
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return (current_epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (current_epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    # Early stopping
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_model_state = None
    epochs_without_improvement = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch) / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)

                if use_amp:
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress every 5 epochs or at key points
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == warmup_epochs - 1:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"F1: {val_f1:.4f} | LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            sys.stdout.flush()

        # Early stopping based on F1 (better for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            sys.stdout.flush()
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation F1: {best_val_f1:.4f}")
    sys.stdout.flush()

    return model, history


def evaluate_model(model, test_loader, y_test):
    """Comprehensive model evaluation."""
    print("\n" + "=" * 60)
    print("Evaluating LSTM Model")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_probs = []
    all_attention = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs, attention = model(X_batch, return_attention=True)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_attention.extend(attention.cpu().numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    attention_weights = np.array(all_attention)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan
    except:
        auc = np.nan

    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        bootstrap_accs.append(accuracy_score(y_test[indices], y_pred[indices]))
    ci_low = np.percentile(bootstrap_accs, 2.5)
    ci_high = np.percentile(bootstrap_accs, 97.5)

    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC:   N/A")
    print(f"  MCC:       {mcc:.4f}")
    print(f"\n95% CI for Accuracy: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"\nConfusion Matrix:")
    print(cm)

    results = {
        'model_name': 'LSTM-Attention',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if not np.isnan(auc) else None,
        'mcc': mcc,
        'accuracy_ci_low': ci_low,
        'accuracy_ci_high': ci_high,
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist()
    }

    return results, attention_weights


def plot_training_history(history, output_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss plot
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', color='blue')
    ax.plot(history['val_loss'], label='Validation Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 plot
    ax = axes[2]
    ax.plot(history['val_f1'], label='Validation F1', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'fig09_lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig09_lstm_training_history.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig09_lstm_training_history.png/pdf")


def plot_attention_analysis(attention_weights, y_test, output_path):
    """Plot attention weight analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Average attention
    ax = axes[0, 0]
    mean_attention = np.mean(attention_weights, axis=0)
    std_attention = np.std(attention_weights, axis=0)
    time_steps = np.arange(len(mean_attention))

    ax.plot(time_steps, mean_attention, 'b-', linewidth=2, label='Mean')
    ax.fill_between(time_steps, mean_attention - std_attention,
                   mean_attention + std_attention, alpha=0.3, color='blue')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Average Temporal Attention Pattern', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Attention by class
    ax = axes[0, 1]
    for class_idx, (class_name, color) in enumerate([('Open', '#3498db'), ('Closed', '#e74c3c')]):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_attention = attention_weights[class_mask]
            mean_class = np.mean(class_attention, axis=0)
            ax.plot(time_steps, mean_class, color=color, linewidth=2, label=class_name)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Pattern by Eye State', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Attention heatmap
    ax = axes[1, 0]
    n_samples_show = min(50, len(attention_weights))
    im = ax.imshow(attention_weights[:n_samples_show], aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Sample')
    ax.set_title('Attention Weights Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # 4. Distribution
    ax = axes[1, 1]
    ax.hist(attention_weights.flatten(), bins=50, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Attention Weights', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'fig10_attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig10_attention_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig10_attention_analysis.png/pdf")


def compare_with_baseline(lstm_results, output_path):
    """Compare LSTM results with baseline models."""
    print("\n" + "=" * 60)
    print("Comparison with Baseline Models")
    print("=" * 60)

    baseline_path = RESULTS_PATH / 'baseline_results.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
    else:
        print("Baseline results not found.")
        return

    all_results = baseline_results.copy()
    all_results['LSTM-Attention'] = lstm_results

    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']

    comparison_data = []
    for model in models:
        row = {'Model': model}
        for metric in metrics:
            value = all_results[model].get(metric, 0)
            row[metric.upper()] = value if value is not None else 0
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(models)]

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [all_results[model].get(m, 0) or 0 for m in metrics]
        ax.bar(x + i * width, values, width, label=model, color=color)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'fig11_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig11_model_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("\nSaved: fig11_model_comparison.png/pdf")

    with open(output_path / 'all_model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("ENHANCED LSTM-ATTENTION MODEL TRAINING (GPU OPTIMIZED)")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Data augmentation (light augmentation for 30-subject dataset)
    print("\n" + "=" * 60)
    print("Applying Light Data Augmentation")
    print("=" * 60)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_factor=0.01)
    print(f"Augmented training data: {X_train_aug.shape}")

    # 3. Create dataloaders (optimized for RTX 3090)
    batch_size = 512  # Fits in 24GB VRAM
    val_batch_size = 1024  # Larger for inference (no gradients)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_aug, y_train_aug, X_val, y_val, X_test, y_test,
        batch_size=batch_size, val_batch_size=val_batch_size
    )
    print(f"\nBatch sizes: train={batch_size}, val/test={val_batch_size}")
    print(f"Effective batch size: {batch_size * 4} (with gradient accumulation)")

    # 4. Create model - input_size from data
    n_channels = X_train.shape[2]  # Get number of channels from data
    hidden_size = 256 if n_channels > 30 else 128  # Larger hidden for more channels

    model = EnhancedLSTMModel(
        input_size=n_channels,
        hidden_size=hidden_size,
        num_layers=3,
        num_classes=2,
        dropout=0.4,
        bidirectional=True,
        num_heads=4
    ).to(DEVICE)

    print(f"\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 5. Train model (GPU optimized settings)
    model, history = train_model(
        model, train_loader, val_loader, y_train_aug,
        epochs=100,
        learning_rate=3e-4,
        patience=15,
        weight_decay=1e-4,
        warmup_epochs=5,
        gradient_accumulation_steps=4
    )

    # 6. Evaluate
    results, attention_weights = evaluate_model(model, test_loader, y_test)

    # 7. Plots
    plot_training_history(history, FIGURES_PATH)
    plot_attention_analysis(attention_weights, y_test, FIGURES_PATH)
    compare_with_baseline(results, RESULTS_PATH)

    # 8. Save
    print("\n" + "=" * 60)
    print("Saving Model and Results")
    print("=" * 60)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': n_channels,
            'hidden_size': hidden_size,
            'num_layers': 3,
            'num_classes': 2,
            'dropout': 0.4,
            'bidirectional': True,
            'num_heads': 4
        },
        'history': history
    }, MODELS_PATH / 'lstm_attention_model.pt')

    print("Saved: lstm_attention_model.pt")

    with open(RESULTS_PATH / 'lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: lstm_results.json")

    np.save(RESULTS_PATH / 'attention_weights.npy', attention_weights)
    print("Saved: attention_weights.npy")

    print("\n" + "=" * 60)
    print("LSTM MODEL TRAINING COMPLETE!")
    print("=" * 60)

    return model, results, attention_weights


if __name__ == "__main__":
    model, results, attention = main()
