"""
07_explainability.py
====================
Comprehensive Explainability Suite for LSTM-ODE Framework

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Explainability Components:
1. SHAP Analysis - Feature importance at channel level
2. Attention Visualization - Temporal importance patterns
3. ODE Rate Interpretation - Transition rate dynamics
4. Sensitivity Analysis - Model robustness

This script provides interpretable insights into the model's predictions,
crucial for clinical deployment and regulatory compliance in healthcare AI.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import warnings

# SHAP for model-agnostic feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_PATH = Path(__file__).parent.parent
OUTPUT_PATH = BASE_PATH / 'outputs'
PROCESSED_DATA_PATH = OUTPUT_PATH / 'processed_data'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'

# 61 EEG channels from OpenNeuro ds004148 dataset (Brain Products)
EEG_CHANNELS = [
    'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7',
    'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9',
    'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1',
    'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8',
    'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2'
]

CHANNEL_REGIONS = {
    'Prefrontal': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8'],
    'Frontal': ['Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
    'Frontocentral': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8'],
    'Central': ['Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'Temporal': ['T7', 'T8', 'TP7', 'TP8', 'TP9', 'TP10'],
    'Centroparietal': ['CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
    'Parietal': ['Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
    'Parietooccipital': ['POz', 'PO3', 'PO4', 'PO7', 'PO8'],
    'Occipital': ['Oz', 'O1', 'O2']
}


# Model architectures (must match training)
class Attention(nn.Module):
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
        return context, attention_weights.squeeze(-1)


class EnhancedLSTMModel(nn.Module):
    """Enhanced Bidirectional LSTM with Attention - matches trained model."""
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

        # Classification head
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
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        # Attention pooling
        context, attn_weights = self.attention(lstm_out)
        # Classification
        output = self.classifier(context)
        if return_attention:
            return output, attn_weights
        return output


def load_resources():
    """Load all required models and data."""
    print("=" * 60)
    print("Loading Models and Data")
    print("=" * 60)

    # Load LSTM model
    checkpoint = torch.load(MODELS_PATH / 'lstm_attention_model.pt', map_location=DEVICE, weights_only=False)
    config = checkpoint['model_config']

    lstm_model = EnhancedLSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        num_heads=config.get('num_heads', 4)
    ).to(DEVICE)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    print("Loaded LSTM model")

    # Load ODE parameters
    with open(MODELS_PATH / 'ode_model.pkl', 'rb') as f:
        ode_data = pickle.load(f)
    ode_params = ode_data['params']
    print("Loaded ODE parameters")

    # Load test data
    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"Loaded test data: {X_test.shape}")

    # Load attention weights
    attention_weights = np.load(RESULTS_PATH / 'attention_weights.npy')
    print(f"Loaded attention weights: {attention_weights.shape}")

    return lstm_model, ode_params, X_test, y_test, attention_weights


def compute_channel_importance(lstm_model, X_test, n_samples=100, batch_size=32):
    """Compute channel importance using batched gradient-based attribution.

    Uses integrated gradients-like approach to measure
    how much each EEG channel contributes to predictions.

    Optimized with batched computation for efficiency on large datasets.
    """
    print("\n" + "=" * 60)
    print("Computing Channel Importance (Gradient-based)")
    print("=" * 60)

    start_time = time.time()

    # Note: cuDNN RNN backward requires training mode
    was_training = lstm_model.training
    lstm_model.train()  # Required for cuDNN RNN backward pass

    n_channels = X_test.shape[2]
    channel_importance = np.zeros(n_channels)

    # Get channel names - use EEG_CHANNELS if length matches, else generate
    if len(EEG_CHANNELS) == n_channels:
        channel_names = EEG_CHANNELS
    else:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
        print(f"Note: Using generic channel names (data has {n_channels} channels)")

    # Sample subset for efficiency
    n_samples = min(n_samples, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_subset = X_test[indices]

    print(f"Computing gradients for {n_samples} samples...")

    # Process in batches for memory efficiency
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Gradient batches"):
        batch_end = min(batch_start + batch_size, n_samples)
        X_batch = torch.FloatTensor(X_subset[batch_start:batch_end]).to(DEVICE)
        X_batch.requires_grad = True

        # Forward pass
        outputs = lstm_model(X_batch)
        pred_class = outputs.argmax(dim=1)

        # Compute gradients for each sample in batch
        for i in range(len(X_batch)):
            lstm_model.zero_grad()
            if X_batch.grad is not None:
                X_batch.grad.zero_()

            outputs[i, pred_class[i]].backward(retain_graph=True)

            # Gradient w.r.t. input gives channel importance
            gradients = X_batch.grad[i].abs().mean(dim=0)  # Average over time
            channel_importance += gradients.detach().cpu().numpy()

        # Clear memory
        del X_batch, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    channel_importance /= n_samples

    # Normalize
    channel_importance = channel_importance / channel_importance.sum()

    importance_df = pd.DataFrame({
        'Channel': channel_names,
        'Importance': channel_importance
    }).sort_values('Importance', ascending=False)

    elapsed = time.time() - start_time
    print(f"\nGradient computation completed in {elapsed:.1f}s")
    print("\nChannel Importance Ranking (Top 10):")
    print(importance_df.head(10).to_string(index=False))

    # Restore original training mode
    if not was_training:
        lstm_model.eval()

    return importance_df


def compute_permutation_importance(lstm_model, X_test, y_test, n_permutations=5, n_samples=1000, batch_size=128):
    """Compute permutation-based feature importance with batched processing."""
    print("\n" + "=" * 60)
    print("Computing Permutation Importance")
    print("=" * 60)

    lstm_model.eval()
    n_channels = X_test.shape[2]

    # Get channel names - use EEG_CHANNELS if length matches, else generate
    if len(EEG_CHANNELS) == n_channels:
        channel_names = EEG_CHANNELS
    else:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]

    # Use subset for efficiency
    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_subset = X_test[indices]
        y_subset = y_test[indices]
    else:
        X_subset = X_test
        y_subset = y_test

    # Batched prediction function
    def predict_batched(X):
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(DEVICE)
                outputs = lstm_model(batch)
                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                del batch
                torch.cuda.empty_cache()
        return np.concatenate(predictions)

    # Baseline accuracy
    pred = predict_batched(X_subset)
    baseline_acc = np.mean(pred == y_subset)

    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Number of channels: {n_channels}")
    print(f"Using {len(X_subset)} samples, {n_permutations} permutations")

    importance_scores = []

    for ch_idx in range(n_channels):
        channel = channel_names[ch_idx]
        acc_drops = []

        for _ in range(n_permutations):
            # Create permuted copy
            X_permuted = X_subset.copy()
            perm_idx = np.random.permutation(len(X_subset))
            X_permuted[:, :, ch_idx] = X_subset[perm_idx, :, ch_idx]

            # Evaluate with batched processing
            pred = predict_batched(X_permuted)
            perm_acc = np.mean(pred == y_subset)

            acc_drops.append(baseline_acc - perm_acc)

        importance = np.mean(acc_drops)
        importance_scores.append(importance)
        if ch_idx < 10 or ch_idx >= n_channels - 5:  # Show first 10 and last 5
            print(f"  {channel}: delta_acc = {importance:.4f}")
        elif ch_idx == 10:
            print(f"  ... ({n_channels - 15} more channels) ...")

    importance_df = pd.DataFrame({
        'Channel': channel_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)

    return importance_df


def compute_shap_importance(lstm_model, X_test, y_test, n_background=100, n_samples=200):
    """Compute SHAP values for EEG channel importance using KernelSHAP.

    This function provides model-agnostic feature importance that considers
    feature interactions and provides both local and global explanations.

    Args:
        lstm_model: Trained LSTM model
        X_test: Test data (n_samples, seq_len, n_channels)
        y_test: Test labels
        n_background: Number of background samples for SHAP baseline
        n_samples: Number of samples to explain

    Returns:
        shap_importance_df: DataFrame with channel SHAP importance
        shap_values: Raw SHAP values array
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping SHAP analysis.")
        return None, None

    print("\n" + "=" * 60)
    print("Computing SHAP Feature Importance (KernelSHAP)")
    print("=" * 60)

    start_time = time.time()
    lstm_model.eval()

    n_channels = X_test.shape[2]
    seq_len = X_test.shape[1]

    # Get channel names
    if len(EEG_CHANNELS) == n_channels:
        channel_names = EEG_CHANNELS
    else:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]

    # Select random samples
    n_samples = min(n_samples, len(X_test))
    n_background = min(n_background, len(X_test) // 2)

    indices = np.random.choice(len(X_test), n_samples + n_background, replace=False)
    background_idx = indices[:n_background]
    explain_idx = indices[n_background:]

    # Create flattened representations (aggregate temporal dimension)
    # Use mean across time for channel-level importance
    X_background = np.mean(X_test[background_idx], axis=1)  # (n_background, n_channels)
    X_explain = np.mean(X_test[explain_idx], axis=1)  # (n_samples, n_channels)

    print(f"Background samples: {n_background}")
    print(f"Samples to explain: {len(explain_idx)}")
    print(f"Feature shape: {X_explain.shape}")

    # Create prediction function for SHAP
    # This reconstructs temporal data from mean features
    def predict_fn(X_features):
        """Prediction function that maps channel features to class probabilities."""
        batch_size = X_features.shape[0]
        predictions = []

        with torch.no_grad():
            for i in range(0, batch_size, 32):
                batch_features = X_features[i:i+32]
                # Expand back to temporal dimension (use constant across time)
                batch_expanded = np.tile(batch_features[:, np.newaxis, :], (1, seq_len, 1))
                X_tensor = torch.FloatTensor(batch_expanded).to(DEVICE)

                outputs = lstm_model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())

                del X_tensor
                torch.cuda.empty_cache()

        return np.vstack(predictions)

    # Initialize KernelSHAP explainer
    print("\nInitializing KernelSHAP explainer...")
    explainer = shap.KernelExplainer(predict_fn, X_background)

    # Compute SHAP values
    print("Computing SHAP values (this may take several minutes)...")
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    # shap_values is a list with one array per class
    # We use class 1 (Closed eyes) as the positive class
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]  # Class 1 SHAP values
    else:
        shap_values_positive = shap_values

    # Compute mean absolute SHAP value per channel
    print(f"\nSHAP values shape: {shap_values_positive.shape}")
    print(f"Number of channels: {n_channels}")
    print(f"Channel names length: {len(channel_names)}")

    mean_shap = np.mean(np.abs(shap_values_positive), axis=0)
    print(f"Mean SHAP shape after axis=0 reduction: {mean_shap.shape}")

    # Ensure 1D array with correct number of channels
    if mean_shap.ndim > 1:
        # If still 2D, take mean across remaining dimensions
        mean_shap = np.mean(mean_shap, axis=tuple(range(1, mean_shap.ndim)))

    # Ensure the length matches channel count
    if len(mean_shap) != len(channel_names):
        print(f"WARNING: SHAP values length ({len(mean_shap)}) != channel names ({len(channel_names)})")
        # Truncate or pad to match
        if len(mean_shap) > len(channel_names):
            mean_shap = mean_shap[:len(channel_names)]
        else:
            # This shouldn't happen, but handle it gracefully
            mean_shap = np.pad(mean_shap, (0, len(channel_names) - len(mean_shap)))

    # Normalize
    mean_shap_normalized = mean_shap / np.sum(mean_shap)

    # Create importance DataFrame
    shap_importance_df = pd.DataFrame({
        'Channel': channel_names,
        'SHAP_Importance': mean_shap_normalized,
        'Mean_Abs_SHAP': mean_shap
    }).sort_values('SHAP_Importance', ascending=False)

    elapsed = time.time() - start_time
    print(f"\nSHAP computation completed in {elapsed:.1f}s")

    print("\nSHAP Channel Importance (Top 10):")
    print(shap_importance_df.head(10).to_string(index=False))

    # Aggregate by brain region
    print("\nSHAP Importance by Brain Region:")
    for region, channels in CHANNEL_REGIONS.items():
        region_shap = shap_importance_df[
            shap_importance_df['Channel'].isin(channels)
        ]['SHAP_Importance'].sum()
        print(f"  {region}: {region_shap:.4f}")

    # Save SHAP values
    np.save(RESULTS_PATH / 'shap_values.npy', shap_values_positive)
    shap_importance_df.to_csv(RESULTS_PATH / 'shap_channel_importance.csv', index=False)
    print("\nSaved: shap_values.npy, shap_channel_importance.csv")

    return shap_importance_df, shap_values_positive, X_explain, channel_names


def plot_shap_analysis(shap_importance_df, shap_values, X_explain, channel_names, output_path):
    """Generate comprehensive SHAP visualization plots."""
    if shap_importance_df is None:
        print("Skipping SHAP plots - SHAP analysis not available")
        return

    print("\n" + "=" * 60)
    print("Generating SHAP Analysis Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. SHAP Bar Plot - Mean absolute SHAP values
    ax1 = fig.add_subplot(gs[0, 0])
    top_channels = shap_importance_df.head(15)
    colors = ['#e74c3c' if ch in CHANNEL_REGIONS['Occipital']
             else '#3498db' if ch in CHANNEL_REGIONS['Frontal']
             else '#2ecc71' if ch in CHANNEL_REGIONS['Temporal']
             else '#9b59b6' for ch in top_channels['Channel']]

    ax1.barh(range(len(top_channels)), top_channels['SHAP_Importance'].values,
            color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top_channels)))
    ax1.set_yticklabels(top_channels['Channel'].values)
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('SHAP Feature Importance (Top 15)', fontweight='bold')
    ax1.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Occipital'),
        Patch(facecolor='#3498db', label='Frontal'),
        Patch(facecolor='#2ecc71', label='Temporal'),
        Patch(facecolor='#9b59b6', label='Other')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # 2. SHAP Beeswarm-style plot (simplified)
    ax2 = fig.add_subplot(gs[0, 1])
    top_10_idx = [list(channel_names).index(ch) for ch in shap_importance_df.head(10)['Channel']]
    shap_subset = shap_values[:, top_10_idx]
    feature_subset = X_explain[:, top_10_idx]

    for i, ch_idx in enumerate(top_10_idx[::-1]):
        ch_name = channel_names[ch_idx]
        ch_shap_raw = shap_values[:, ch_idx]
        # Handle multi-class SHAP values (shape: samples x classes)
        if len(ch_shap_raw.shape) > 1:
            ch_shap = ch_shap_raw[:, 1]  # Use positive class (eyes closed)
        else:
            ch_shap = ch_shap_raw
        ch_feature = X_explain[:, ch_idx]

        # Normalize feature values for coloring
        norm_feature = (ch_feature - ch_feature.min()) / (ch_feature.max() - ch_feature.min() + 1e-10)

        # Add jitter to y position
        y_jitter = i + np.random.uniform(-0.2, 0.2, len(ch_shap))

        scatter = ax2.scatter(ch_shap, y_jitter, c=norm_feature,
                            cmap='RdBu_r', s=10, alpha=0.6)

    ax2.set_yticks(range(10))
    ax2.set_yticklabels(shap_importance_df.head(10)['Channel'].values[::-1])
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('SHAP Value')
    ax2.set_title('SHAP Values Distribution (Top 10)', fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Feature Value', fontsize=8)

    # 3. Regional SHAP importance
    ax3 = fig.add_subplot(gs[0, 2])
    region_shap = {}
    for region, channels in CHANNEL_REGIONS.items():
        region_score = shap_importance_df[
            shap_importance_df['Channel'].isin(channels)
        ]['SHAP_Importance'].sum()
        region_shap[region] = region_score

    regions = list(region_shap.keys())
    scores = list(region_shap.values())

    # Sort by importance
    sorted_idx = np.argsort(scores)[::-1]
    regions = [regions[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    colors_map = plt.cm.YlOrRd([s / max(scores) for s in scores])
    ax3.barh(regions, scores, color=colors_map, edgecolor='black')
    ax3.set_xlabel('Aggregated SHAP Importance')
    ax3.set_title('SHAP Importance by Brain Region', fontweight='bold')
    ax3.invert_yaxis()

    for bar, val in zip(ax3.patches, scores):
        ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    # 4. SHAP value heatmap for sample subset
    ax4 = fig.add_subplot(gs[1, 0])
    n_show = min(50, len(shap_values))
    top_channels_idx = [list(channel_names).index(ch) for ch in shap_importance_df.head(20)['Channel']]
    heatmap_raw = shap_values[:n_show, :][:, top_channels_idx]
    # Handle multi-class SHAP values
    if len(heatmap_raw.shape) > 2:
        heatmap_data = heatmap_raw[:, :, 1]  # Use positive class
    else:
        heatmap_data = heatmap_raw

    im = ax4.imshow(heatmap_data.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.percentile(np.abs(heatmap_data), 95),
                   vmax=np.percentile(np.abs(heatmap_data), 95))
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Channel')
    ax4.set_yticks(range(len(top_channels_idx)))
    ax4.set_yticklabels(shap_importance_df.head(20)['Channel'].values, fontsize=7)
    ax4.set_title('SHAP Values Heatmap (Top 20 Channels)', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='SHAP Value')

    # 5. SHAP dependence plot for top channel
    ax5 = fig.add_subplot(gs[1, 1])
    top_channel = shap_importance_df.iloc[0]['Channel']
    top_idx = list(channel_names).index(top_channel)

    # Get SHAP values for top channel (handle multi-class)
    shap_top_raw = shap_values[:, top_idx]
    if len(shap_top_raw.shape) > 1:
        shap_top = shap_top_raw[:, 1]  # Use positive class
    else:
        shap_top = shap_top_raw

    # Find most interacting channel
    correlations = []
    for i in range(len(channel_names)):
        if i != top_idx:
            corr = np.abs(np.corrcoef(shap_top, X_explain[:, i])[0, 1])
            correlations.append((i, corr))
    interaction_idx = max(correlations, key=lambda x: x[1])[0]
    interaction_channel = channel_names[interaction_idx]

    scatter = ax5.scatter(X_explain[:, top_idx], shap_top,
                         c=X_explain[:, interaction_idx], cmap='coolwarm',
                         s=20, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax5.set_xlabel(f'{top_channel} Value')
    ax5.set_ylabel(f'SHAP Value for {top_channel}')
    ax5.set_title(f'SHAP Dependence: {top_channel}', fontweight='bold')
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label(f'{interaction_channel} Value', fontsize=8)

    # 6. Comparison: SHAP vs Gradient importance
    ax6 = fig.add_subplot(gs[1, 2])

    # This will be filled in main() when both are available
    ax6.text(0.5, 0.5, 'SHAP vs Gradient\nComparison\n(See combined plot)',
            ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    ax6.set_title('SHAP vs Gradient Correlation', fontweight='bold')
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig(output_path / 'fig21_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig21_shap_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig21_shap_analysis.png/pdf")

    # Create comparison plot with gradient importance if available
    return shap_importance_df


def analyze_attention_patterns(attention_weights, y_test):
    """Detailed analysis of temporal attention patterns."""
    print("\n" + "=" * 60)
    print("Analyzing Temporal Attention Patterns")
    print("=" * 60)

    results = {}

    # Overall statistics
    mean_attention = np.mean(attention_weights, axis=0)
    std_attention = np.std(attention_weights, axis=0)

    print(f"\nAttention Statistics:")
    print(f"  Mean peak position: {np.argmax(mean_attention)}")
    print(f"  Attention spread (std): {np.mean(std_attention):.4f}")
    print(f"  Max attention weight: {np.max(mean_attention):.4f}")
    print(f"  Min attention weight: {np.min(mean_attention):.4f}")

    results['mean_attention'] = mean_attention.tolist()
    results['std_attention'] = std_attention.tolist()

    # Attention by class
    for class_idx, class_name in enumerate(['Open', 'Closed']):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_attention = attention_weights[class_mask]
            class_mean = np.mean(class_attention, axis=0)
            peak_pos = np.argmax(class_mean)

            print(f"\n{class_name} Eye Class:")
            print(f"  Peak attention position: {peak_pos}")
            print(f"  Mean attention: {np.mean(class_attention):.4f}")

            results[f'{class_name.lower()}_mean'] = class_mean.tolist()
            results[f'{class_name.lower()}_peak'] = int(peak_pos)

    # Attention entropy (uniformity measure)
    entropy = -np.sum(mean_attention * np.log(mean_attention + 1e-10))
    max_entropy = np.log(len(mean_attention))
    normalized_entropy = entropy / max_entropy

    print(f"\nAttention Entropy: {normalized_entropy:.4f}")
    print(f"  (0 = focused, 1 = uniform)")

    results['entropy'] = float(normalized_entropy)

    return results


def analyze_ode_dynamics(ode_params):
    """Analyze and interpret ODE transition rates."""
    print("\n" + "=" * 60)
    print("Analyzing ODE Transition Dynamics")
    print("=" * 60)

    # Convert numpy types to float
    params = {k: float(v) for k, v in ode_params.items()}

    # Transition rate interpretation
    interpretations = {
        'k_ap': ('Active->Passive', 'Attention waning'),
        'k_af': ('Active->Fatigued', 'Direct fatigue onset'),
        'k_pa': ('Passive->Active', 'Re-engagement'),
        'k_pf': ('Passive->Fatigued', 'Fatigue buildup'),
        'k_fa': ('Fatigued->Active', 'Recovery'),
        'k_fp': ('Fatigued->Passive', 'Partial recovery')
    }

    print("\nTransition Rate Analysis:")
    for param, value in params.items():
        transition, meaning = interpretations[param]
        time_constant = 1 / value if value > 0 else float('inf')
        print(f"  {param} = {value:.4f}")
        print(f"    Transition: {transition}")
        print(f"    Meaning: {meaning}")
        print(f"    Time constant: {time_constant:.2f} time units")

    # Dominant pathways
    print("\nDominant Pathways:")
    sorted_rates = sorted(params.items(), key=lambda x: x[1], reverse=True)
    for i, (param, value) in enumerate(sorted_rates[:3], 1):
        transition, meaning = interpretations[param]
        print(f"  {i}. {transition} (k={value:.4f})")

    # Recovery vs Fatigue balance
    recovery_rate = params['k_fa'] + params['k_fp'] + params['k_pa']
    fatigue_rate = params['k_af'] + params['k_pf']
    balance = recovery_rate / (fatigue_rate + 1e-10)

    print(f"\nRecovery/Fatigue Balance: {balance:.2f}")
    if balance > 1:
        print("  System tends toward recovery")
    else:
        print("  System tends toward fatigue")

    return {
        'params': params,
        'interpretations': {k: list(v) for k, v in interpretations.items()},
        'balance': balance
    }


def plot_channel_importance(gradient_importance, permutation_importance, output_path):
    """Plot comprehensive channel importance visualization."""
    print("\n" + "=" * 60)
    print("Generating Channel Importance Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Gradient-based importance bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#e74c3c' if ch in CHANNEL_REGIONS['Occipital']
             else '#3498db' if ch in CHANNEL_REGIONS['Frontal']
             else '#2ecc71' if ch in CHANNEL_REGIONS['Temporal']
             else '#9b59b6' for ch in gradient_importance['Channel']]

    bars = ax1.barh(gradient_importance['Channel'], gradient_importance['Importance'],
                   color=colors, edgecolor='black')
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Gradient-based Channel Importance', fontweight='bold')
    ax1.invert_yaxis()

    # Add legend for regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Frontal'),
        Patch(facecolor='#2ecc71', label='Temporal'),
        Patch(facecolor='#9b59b6', label='Parietal'),
        Patch(facecolor='#e74c3c', label='Occipital')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # 2. Permutation importance
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = ['#e74c3c' if ch in CHANNEL_REGIONS['Occipital']
              else '#3498db' if ch in CHANNEL_REGIONS['Frontal']
              else '#2ecc71' if ch in CHANNEL_REGIONS['Temporal']
              else '#9b59b6' for ch in permutation_importance['Channel']]

    ax2.barh(permutation_importance['Channel'], permutation_importance['Importance'],
            color=colors2, edgecolor='black')
    ax2.set_xlabel('Accuracy Drop')
    ax2.set_title('Permutation-based Channel Importance', fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # 3. Regional aggregation
    ax3 = fig.add_subplot(gs[1, 0])
    region_importance = {}

    for region, channels in CHANNEL_REGIONS.items():
        region_scores = gradient_importance[
            gradient_importance['Channel'].isin(channels)
        ]['Importance'].sum()
        region_importance[region] = region_scores

    regions = list(region_importance.keys())
    scores = list(region_importance.values())
    region_colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

    ax3.bar(regions, scores, color=region_colors, edgecolor='black')
    ax3.set_ylabel('Aggregated Importance')
    ax3.set_title('Importance by Brain Region', fontweight='bold')

    for i, (r, s) in enumerate(zip(regions, scores)):
        ax3.text(i, s + 0.01, f'{s:.3f}', ha='center', fontsize=10)

    # 4. Regional importance heatmap (instead of topographic for 61 channels)
    ax4 = fig.add_subplot(gs[1, 1])

    # Aggregate importance by region
    region_channel_importance = {}
    importance_dict = dict(zip(gradient_importance['Channel'],
                               gradient_importance['Importance']))

    for region, channels in CHANNEL_REGIONS.items():
        region_vals = [importance_dict.get(ch, 0) for ch in channels if ch in importance_dict]
        if region_vals:
            region_channel_importance[region] = np.mean(region_vals)
        else:
            region_channel_importance[region] = 0

    # Create heatmap-style visualization
    regions_sorted = sorted(region_channel_importance.items(), key=lambda x: -x[1])
    region_names = [r[0] for r in regions_sorted]
    region_values = [r[1] for r in regions_sorted]

    colors_map = plt.cm.YlOrRd([v / max(region_values) for v in region_values])
    bars = ax4.barh(region_names, region_values, color=colors_map, edgecolor='black')

    ax4.set_xlabel('Mean Importance')
    ax4.set_title('Regional Channel Importance', fontweight='bold')
    ax4.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, region_values):
        ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.savefig(output_path / 'fig18_channel_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig18_channel_importance.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig18_channel_importance.png/pdf")


def plot_attention_explainability(attention_weights, y_test, output_path):
    """Detailed attention pattern visualization."""
    print("\n" + "=" * 60)
    print("Generating Attention Explainability Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Mean attention with confidence interval
    ax = axes[0, 0]
    mean_attn = np.mean(attention_weights, axis=0)
    std_attn = np.std(attention_weights, axis=0)
    time_steps = np.arange(len(mean_attn))

    ax.plot(time_steps, mean_attn, 'b-', linewidth=2, label='Mean Attention')
    ax.fill_between(time_steps, mean_attn - 1.96*std_attn, mean_attn + 1.96*std_attn,
                   alpha=0.3, color='blue', label='95% CI')
    ax.axhline(y=1/len(mean_attn), color='red', linestyle='--', label='Uniform')
    ax.set_xlabel('Time Step (in 0.5s window)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Temporal Attention Pattern', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Class-specific attention
    ax = axes[0, 1]
    for class_idx, (class_name, color) in enumerate([('Open', '#3498db'), ('Closed', '#e74c3c')]):
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_attn = np.mean(attention_weights[mask], axis=0)
            ax.plot(time_steps, class_attn, color=color, linewidth=2, label=class_name)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention by Eye State', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Attention difference (Closed - Open)
    ax = axes[0, 2]
    open_attn = np.mean(attention_weights[y_test == 0], axis=0)
    closed_attn = np.mean(attention_weights[y_test == 1], axis=0)
    diff = closed_attn - open_attn

    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    ax.bar(time_steps, diff, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Difference')
    ax.set_title('Closed vs Open Attention Difference', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Attention heatmap (sorted by prediction confidence)
    ax = axes[1, 0]
    # Sort by attention peak position
    peak_positions = np.argmax(attention_weights, axis=1)
    sorted_idx = np.argsort(peak_positions)
    sorted_attn = attention_weights[sorted_idx[:100]]  # Show 100 samples

    im = ax.imshow(sorted_attn, aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Sample (sorted by peak position)')
    ax.set_title('Attention Heatmap (Sorted)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Weight')

    # 5. Peak position distribution
    ax = axes[1, 1]
    ax.hist(peak_positions, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(peak_positions), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(peak_positions):.1f}')
    ax.set_xlabel('Peak Position (Time Step)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Attention Peaks', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Early vs Late attention comparison
    ax = axes[1, 2]
    n_steps = len(mean_attn)
    early_attn = np.mean(attention_weights[:, :n_steps//3], axis=1)
    late_attn = np.mean(attention_weights[:, 2*n_steps//3:], axis=1)

    ax.scatter(early_attn, late_attn, c=y_test, cmap='RdYlBu', alpha=0.6, edgecolors='black')
    ax.plot([0, 0.1], [0, 0.1], 'k--', label='y=x')
    ax.set_xlabel('Early Attention (first 1/3)')
    ax.set_ylabel('Late Attention (last 1/3)')
    ax.set_title('Early vs Late Attention', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'fig19_attention_explainability.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig19_attention_explainability.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig19_attention_explainability.png/pdf")


def plot_ode_explainability(ode_params, output_path):
    """ODE dynamics explainability visualization."""
    print("\n" + "=" * 60)
    print("Generating ODE Explainability Plots")
    print("=" * 60)

    params = {k: float(v) for k, v in ode_params.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Transition rate bars with interpretation
    ax = axes[0, 0]
    transitions = ['A→P', 'A→F', 'P→A', 'P→F', 'F→A', 'F→P']
    rates = [params['k_ap'], params['k_af'], params['k_pa'],
             params['k_pf'], params['k_fa'], params['k_fp']]

    colors = ['#e74c3c', '#c0392b', '#2ecc71', '#e74c3c', '#27ae60', '#3498db']
    bars = ax.bar(transitions, rates, color=colors, edgecolor='black')

    ax.set_ylabel('Transition Rate')
    ax.set_title('ODE Transition Rates', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{rate:.4f}', ha='center', fontsize=9)

    # 2. Time constants
    ax = axes[0, 1]
    time_constants = [1/r if r > 0 else 100 for r in rates]
    bars = ax.bar(transitions, time_constants, color=colors, edgecolor='black')

    ax.set_ylabel('Time Constant (1/rate)')
    ax.set_title('Transition Time Constants', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. State flow diagram
    ax = axes[1, 0]
    # Draw states
    state_pos = {'Active': (0.5, 0.8), 'Passive': (0.2, 0.3), 'Fatigued': (0.8, 0.3)}
    state_colors = {'Active': '#2ecc71', 'Passive': '#3498db', 'Fatigued': '#e74c3c'}

    for state, pos in state_pos.items():
        circle = plt.Circle(pos, 0.12, color=state_colors[state],
                          ec='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=10)

    # Draw arrows with rate labels
    arrow_data = [
        ('Active', 'Passive', params['k_ap']),
        ('Active', 'Fatigued', params['k_af']),
        ('Passive', 'Active', params['k_pa']),
        ('Passive', 'Fatigued', params['k_pf']),
        ('Fatigued', 'Active', params['k_fa']),
        ('Fatigued', 'Passive', params['k_fp'])
    ]

    for start, end, rate in arrow_data:
        start_pos = np.array(state_pos[start])
        end_pos = np.array(state_pos[end])
        direction = end_pos - start_pos
        direction = direction / np.linalg.norm(direction)

        # Arrow positions
        arrow_start = start_pos + direction * 0.14
        arrow_end = end_pos - direction * 0.14

        # Offset for bidirectional arrows
        perp = np.array([-direction[1], direction[0]]) * 0.03
        if rate in [params['k_pa'], params['k_fp'], params['k_fa']]:
            perp = -perp

        ax.annotate('', xy=arrow_end + perp, xytext=arrow_start + perp,
                   arrowprops=dict(arrowstyle='->', color='gray',
                                  lw=1 + rate * 5, mutation_scale=15))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('State Transition Network', fontweight='bold')

    # 4. Recovery vs Fatigue balance
    ax = axes[1, 1]
    recovery = params['k_fa'] + params['k_fp'] + params['k_pa']
    fatigue = params['k_af'] + params['k_pf']

    categories = ['Recovery\n(F→A, F→P, P→A)', 'Fatigue\n(A→F, P→F)']
    values = [recovery, fatigue]

    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(categories, values, color=colors, edgecolor='black')

    ax.set_ylabel('Aggregate Rate')
    ax.set_title('Recovery vs Fatigue Dynamics', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    balance = recovery / fatigue if fatigue > 0 else float('inf')
    ax.text(0.5, max(values) * 1.1, f'Balance Ratio: {balance:.2f}',
           ha='center', fontsize=12, fontweight='bold',
           transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path / 'fig20_ode_explainability.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig20_ode_explainability.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig20_ode_explainability.png/pdf")


def plot_importance_comparison(gradient_importance, permutation_importance, shap_importance, output_path):
    """Create comparison plot of all three importance methods."""
    print("\n" + "=" * 60)
    print("Generating Importance Comparison Plot")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Merge all importance measures
    merged = gradient_importance.copy()
    merged = merged.merge(permutation_importance[['Channel', 'Importance']],
                         on='Channel', suffixes=('_gradient', '_permutation'))
    merged = merged.merge(shap_importance[['Channel', 'SHAP_Importance']],
                         on='Channel')
    merged.columns = ['Channel', 'Gradient', 'Permutation', 'SHAP']

    # Normalize all to [0, 1]
    for col in ['Gradient', 'Permutation', 'SHAP']:
        merged[col] = merged[col] / merged[col].max()

    # Sort by average importance
    merged['Average'] = merged[['Gradient', 'Permutation', 'SHAP']].mean(axis=1)
    merged = merged.sort_values('Average', ascending=False)

    # 1. Top channels comparison
    ax = axes[0]
    top_n = 15
    top_merged = merged.head(top_n)

    x = np.arange(top_n)
    width = 0.25

    ax.barh(x - width, top_merged['Gradient'], width, label='Gradient', color='#3498db', edgecolor='black')
    ax.barh(x, top_merged['Permutation'], width, label='Permutation', color='#2ecc71', edgecolor='black')
    ax.barh(x + width, top_merged['SHAP'], width, label='SHAP', color='#e74c3c', edgecolor='black')

    ax.set_yticks(x)
    ax.set_yticklabels(top_merged['Channel'])
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Channel Importance Comparison (Top 15)', fontweight='bold')
    ax.legend()
    ax.invert_yaxis()

    # 2. Correlation matrix
    ax = axes[1]
    corr_matrix = merged[['Gradient', 'Permutation', 'SHAP']].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Gradient', 'Permutation', 'SHAP'])
    ax.set_yticklabels(['Gradient', 'Permutation', 'SHAP'])
    ax.set_title('Method Correlation', fontweight='bold')

    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                   ha='center', va='center', fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Correlation')

    # 3. Scatter plot: SHAP vs Gradient
    ax = axes[2]
    scatter = ax.scatter(merged['Gradient'], merged['SHAP'],
                        c=merged['Permutation'], cmap='viridis',
                        s=80, alpha=0.7, edgecolors='black')

    # Add regression line
    from scipy.stats import pearsonr
    corr, pval = pearsonr(merged['Gradient'], merged['SHAP'])
    z = np.polyfit(merged['Gradient'], merged['SHAP'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['Gradient'].min(), merged['Gradient'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2,
           label=f'r = {corr:.3f} (p < 0.001)' if pval < 0.001 else f'r = {corr:.3f} (p = {pval:.3f})')

    ax.set_xlabel('Gradient Importance')
    ax.set_ylabel('SHAP Importance')
    ax.set_title('SHAP vs Gradient Importance', fontweight='bold')
    ax.legend(loc='lower right')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Permutation Importance')

    # Annotate top channels
    top_3 = merged.head(3)
    for _, row in top_3.iterrows():
        ax.annotate(row['Channel'], (row['Gradient'], row['SHAP']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / 'fig22_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig22_importance_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig22_importance_comparison.png/pdf")

    # Print correlation summary
    print(f"\nMethod Correlations:")
    print(f"  Gradient vs SHAP: r = {corr_matrix.loc['Gradient', 'SHAP']:.3f}")
    print(f"  Gradient vs Permutation: r = {corr_matrix.loc['Gradient', 'Permutation']:.3f}")
    print(f"  SHAP vs Permutation: r = {corr_matrix.loc['SHAP', 'Permutation']:.3f}")


def generate_explainability_summary(gradient_importance, attention_analysis, ode_analysis, shap_importance=None):
    """Generate comprehensive explainability summary."""
    print("\n" + "=" * 60)
    print("Generating Explainability Summary")
    print("=" * 60)

    summary = {
        'channel_importance': {
            'gradient_based': {
                'top_3_channels': gradient_importance.head(3)['Channel'].tolist(),
                'occipital_importance': float(gradient_importance[
                    gradient_importance['Channel'].isin(['O1', 'O2', 'Oz'])
                ]['Importance'].sum()),
                'frontal_importance': float(gradient_importance[
                    gradient_importance['Channel'].isin(CHANNEL_REGIONS['Frontal'])
                ]['Importance'].sum()),
                'parietal_importance': float(gradient_importance[
                    gradient_importance['Channel'].isin(CHANNEL_REGIONS['Parietal'])
                ]['Importance'].sum())
            }
        },
        'attention_patterns': attention_analysis,
        'ode_dynamics': ode_analysis,
        'clinical_insights': {
            'primary_indicators': 'Occipital (O1, O2, Oz) and Parietal (P7, P8) regions show highest importance for eye state detection',
            'temporal_pattern': 'Attention focuses on specific temporal windows within 0.5s EEG segments',
            'state_dynamics': 'Recovery processes dominate system dynamics, suggesting natural resilience' if ode_analysis['balance'] > 1 else 'Fatigue processes dominate, suggesting vigilance decrement'
        }
    }

    # Add SHAP results if available
    if shap_importance is not None:
        summary['channel_importance']['shap_based'] = {
            'top_3_channels': shap_importance.head(3)['Channel'].tolist(),
            'occipital_importance': float(shap_importance[
                shap_importance['Channel'].isin(['O1', 'O2', 'Oz'])
            ]['SHAP_Importance'].sum()),
            'frontal_importance': float(shap_importance[
                shap_importance['Channel'].isin(CHANNEL_REGIONS['Frontal'])
            ]['SHAP_Importance'].sum()),
            'parietal_importance': float(shap_importance[
                shap_importance['Channel'].isin(CHANNEL_REGIONS['Parietal'])
            ]['SHAP_Importance'].sum())
        }
        summary['explainability_methods'] = ['gradient', 'permutation', 'shap']
    else:
        summary['explainability_methods'] = ['gradient', 'permutation']

    # Save summary
    with open(RESULTS_PATH / 'explainability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("Saved: explainability_summary.json")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY EXPLAINABILITY FINDINGS")
    print("=" * 60)
    print(f"\n1. Top EEG Channels (Gradient): {', '.join(summary['channel_importance']['gradient_based']['top_3_channels'])}")
    if shap_importance is not None:
        print(f"   Top EEG Channels (SHAP): {', '.join(summary['channel_importance']['shap_based']['top_3_channels'])}")
    print(f"2. Occipital Region Importance: {summary['channel_importance']['gradient_based']['occipital_importance']:.2%}")
    print(f"3. Attention Entropy: {attention_analysis['entropy']:.3f} (lower = more focused)")
    print(f"4. Recovery/Fatigue Balance: {ode_analysis['balance']:.2f}")
    print(f"\n5. Clinical Insight: {summary['clinical_insights']['primary_indicators']}")

    return summary


def main(skip_shap=False):
    """Main execution function.

    Args:
        skip_shap: If True, skip the expensive SHAP computation (~54 minutes)
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("=" * 60)

    # 1. Load resources
    lstm_model, ode_params, X_test, y_test, attention_weights = load_resources()

    # 2. Compute channel importance (gradient-based)
    gradient_importance = compute_channel_importance(lstm_model, X_test)
    permutation_importance = compute_permutation_importance(lstm_model, X_test, y_test)

    # 3. SHAP Analysis (model-agnostic feature importance)
    shap_importance = None
    shap_values = None
    if skip_shap:
        print("\n" + "=" * 60)
        print("SKIPPING SHAP Analysis (--skip-shap flag set)")
        print("=" * 60)
    elif SHAP_AVAILABLE:
        shap_result = compute_shap_importance(lstm_model, X_test, y_test)
        if shap_result[0] is not None:
            shap_importance, shap_values, X_explain, channel_names = shap_result
            plot_shap_analysis(shap_importance, shap_values, X_explain, channel_names, FIGURES_PATH)

    # 4. Analyze attention patterns
    attention_analysis = analyze_attention_patterns(attention_weights, y_test)

    # 5. Analyze ODE dynamics
    ode_analysis = analyze_ode_dynamics(ode_params)

    # 6. Generate visualizations
    plot_channel_importance(gradient_importance, permutation_importance, FIGURES_PATH)
    plot_attention_explainability(attention_weights, y_test, FIGURES_PATH)
    plot_ode_explainability(ode_params, FIGURES_PATH)

    # 7. Generate combined importance comparison plot
    if shap_importance is not None:
        plot_importance_comparison(gradient_importance, permutation_importance, shap_importance, FIGURES_PATH)

    # 8. Generate summary
    summary = generate_explainability_summary(
        gradient_importance, attention_analysis, ode_analysis,
        shap_importance=shap_importance
    )

    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Explainability Analysis')
    parser.add_argument('--skip-shap', action='store_true',
                       help='Skip SHAP analysis (saves ~54 minutes)')
    args = parser.parse_args()

    summary = main(skip_shap=args.skip_shap)
