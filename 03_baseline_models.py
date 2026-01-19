"""
03_baseline_models.py
=====================
Baseline Models for EEG Eye State Classification
OpenNeuro ds004148: Test-Retest Resting and Cognitive State EEG Dataset

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Baseline Models:
1. Support Vector Machine (SVM) with RBF kernel
2. Random Forest with optimized hyperparameters
3. XGBoost with gradient boosting
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings('ignore')

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using GradientBoostingClassifier instead")

# Set random seed
np.random.seed(42)

# Paths
BASE_PATH = Path(__file__).parent.parent
PROCESSED_DATA_PATH = BASE_PATH / 'outputs' / 'processed_data'
OUTPUT_PATH = BASE_PATH / 'outputs'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'

for path in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset parameters
SAMPLING_RATE = 500  # Hz


def load_processed_data():
    """Load preprocessed sequence data using memory mapping for large files."""
    print("=" * 60)
    print("Loading Processed Data")
    print("=" * 60)

    # Use memory mapping for large files
    print("Loading with memory mapping (this may take a moment for 48GB file)...")
    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz', allow_pickle=True, mmap_mode='r')

    X_train = data['X_train']
    y_train = np.array(data['y_train'])  # Labels need to be in memory
    X_val = data['X_val'] if len(data['X_val']) > 0 else None
    y_val = np.array(data['y_val']) if X_val is not None else None
    X_test = data['X_test'] if len(data['X_test']) > 0 else None
    y_test = np.array(data['y_test']) if X_test is not None else None

    # Load metadata
    with open(PROCESSED_DATA_PATH / 'preprocessing_metadata.json', 'r') as f:
        metadata = json.load(f)

    channel_names = metadata.get('channel_names', [f'CH{i}' for i in range(X_train.shape[2])])

    print(f"X_train: {X_train.shape}")
    if X_val is not None:
        print(f"X_val: {X_val.shape}")
    else:
        print("X_val: None (using train/test only)")
    if X_test is not None:
        print(f"X_test: {X_test.shape}")
    print(f"N channels: {len(channel_names)}")
    print(f"Class balance - Train: {np.mean(y_train):.1%}")

    return X_train, y_train, X_val, y_val, X_test, y_test, channel_names


def load_or_extract_features(X_train, y_train, X_val, y_val, X_test, y_test, channel_names):
    """Load features from cache or extract them."""
    cache_file = PROCESSED_DATA_PATH / 'extracted_features.npz'

    if cache_file.exists():
        print("Loading features from cache...")
        cached = np.load(cache_file, allow_pickle=True)
        X_train_feat = cached['X_train_feat']
        X_val_feat = cached['X_val_feat'] if 'X_val_feat' in cached else None
        X_test_feat = cached['X_test_feat']
        feature_names = list(cached['feature_names'])
        print(f"Loaded features: {X_train_feat.shape}")
        return X_train_feat, X_val_feat, X_test_feat, feature_names

    print("Extracting features (will cache for future runs)...")

    # Extract features using GPU
    X_train_feat, feature_names = extract_features_gpu(X_train, channel_names)

    if X_val is not None:
        X_val_feat, _ = extract_features_gpu(X_val, channel_names)
    else:
        X_val_feat = None

    X_test_feat, _ = extract_features_gpu(X_test, channel_names)

    # Cache features
    print("Caching extracted features...")
    if X_val_feat is not None:
        np.savez_compressed(cache_file,
                           X_train_feat=X_train_feat,
                           X_val_feat=X_val_feat,
                           X_test_feat=X_test_feat,
                           feature_names=feature_names)
    else:
        np.savez_compressed(cache_file,
                           X_train_feat=X_train_feat,
                           X_test_feat=X_test_feat,
                           feature_names=feature_names)
    print(f"Cached to: {cache_file}")

    return X_train_feat, X_val_feat, X_test_feat, feature_names


def extract_features_gpu(X, channel_names, fs=500, batch_size=10000):
    """GPU-accelerated feature extraction using PyTorch.

    Features per channel (20 total):
    - Time domain: mean, std, var, min, max, range, skew, kurt, zcr, energy,
                   activity, mobility, complexity (13 features)
    - Frequency: delta, theta, alpha, beta, gamma power ratios,
                 alpha/theta, alpha/beta (7 features)
    """
    print("Extracting features using GPU acceleration...")

    n_samples, seq_len, n_channels = X.shape
    n_features_per_channel = 20  # 13 time + 7 freq

    # Convert to torch tensor on GPU
    X_tensor = torch.from_numpy(X).float().to(DEVICE)

    # Prepare frequency bands indices
    freqs = torch.fft.rfftfreq(seq_len, 1/fs).to(DEVICE)
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs < 45)

    all_features = []

    for start_idx in tqdm(range(0, n_samples, batch_size), desc="GPU feature extraction"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X_tensor[start_idx:end_idx]  # (batch, seq, ch)
        batch_size_actual = batch.shape[0]

        batch_features = []

        for ch in range(n_channels):
            signal = batch[:, :, ch]  # (batch, seq)

            # Time domain features
            mean = signal.mean(dim=1)
            std = signal.std(dim=1)
            var = signal.var(dim=1)
            min_val = signal.min(dim=1)[0]
            max_val = signal.max(dim=1)[0]
            range_val = max_val - min_val

            # Skewness and kurtosis (using moments)
            centered = signal - mean.unsqueeze(1)
            m2 = (centered ** 2).mean(dim=1)
            m3 = (centered ** 3).mean(dim=1)
            m4 = (centered ** 4).mean(dim=1)

            skew = m3 / (m2 ** 1.5 + 1e-10)
            kurt = m4 / (m2 ** 2 + 1e-10) - 3

            # Zero crossing rate
            zero_crossings = torch.abs(torch.diff(torch.sign(centered), dim=1)).sum(dim=1) / 2
            zcr = zero_crossings / seq_len

            # Energy
            energy = (signal ** 2).mean(dim=1)

            # Hjorth parameters
            diff1 = torch.diff(signal, dim=1)
            diff2 = torch.diff(diff1, dim=1)

            activity = var
            mobility = diff1.std(dim=1) / (std + 1e-10)
            complexity = (diff2.std(dim=1) / (diff1.std(dim=1) + 1e-10)) / (mobility + 1e-10)

            # FFT for frequency features
            fft_vals = torch.abs(torch.fft.rfft(signal, dim=1))
            fft_power = fft_vals ** 2

            delta_power = fft_power[:, delta_mask].sum(dim=1)
            theta_power = fft_power[:, theta_mask].sum(dim=1)
            alpha_power = fft_power[:, alpha_mask].sum(dim=1)
            beta_power = fft_power[:, beta_mask].sum(dim=1)
            gamma_power = fft_power[:, gamma_mask].sum(dim=1)

            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power + 1e-10

            delta_ratio = delta_power / total_power
            theta_ratio = theta_power / total_power
            alpha_ratio = alpha_power / total_power
            beta_ratio = beta_power / total_power
            gamma_ratio = gamma_power / total_power
            alpha_theta = alpha_power / (theta_power + 1e-10)
            alpha_beta = alpha_power / (beta_power + 1e-10)

            # Stack all features for this channel
            ch_features = torch.stack([
                mean, std, var, min_val, max_val, range_val,
                skew, kurt, zcr, energy, activity, mobility, complexity,
                delta_ratio, theta_ratio, alpha_ratio, beta_ratio, gamma_ratio,
                alpha_theta, alpha_beta
            ], dim=1)  # (batch, 20)

            batch_features.append(ch_features)

        # Concatenate all channels: (batch, n_channels * 20)
        batch_features = torch.cat(batch_features, dim=1)
        all_features.append(batch_features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    # Clean up NaN/Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate feature names
    feature_names = []
    for ch in range(n_channels):
        ch_name = channel_names[ch] if ch < len(channel_names) else f'CH{ch}'
        feature_names.extend([
            f'{ch_name}_mean', f'{ch_name}_std', f'{ch_name}_var',
            f'{ch_name}_min', f'{ch_name}_max', f'{ch_name}_range',
            f'{ch_name}_skew', f'{ch_name}_kurt', f'{ch_name}_zcr',
            f'{ch_name}_energy', f'{ch_name}_activity',
            f'{ch_name}_mobility', f'{ch_name}_complexity',
            f'{ch_name}_delta', f'{ch_name}_theta', f'{ch_name}_alpha',
            f'{ch_name}_beta', f'{ch_name}_gamma',
            f'{ch_name}_alpha_theta', f'{ch_name}_alpha_beta'
        ])

    print(f"Extracted {features.shape[1]} features from {n_samples} samples")
    return features, feature_names


def extract_frequency_features(segment, fs=500):
    """Extract frequency domain features from an EEG segment."""
    n_samples = len(segment)

    # Compute FFT
    fft_vals = np.abs(fft(segment))[:n_samples // 2]
    freqs = fftfreq(n_samples, 1/fs)[:n_samples // 2]

    # Band power features
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs < 45)

    delta_power = np.sum(fft_vals[delta_mask] ** 2) if np.any(delta_mask) else 0
    theta_power = np.sum(fft_vals[theta_mask] ** 2) if np.any(theta_mask) else 0
    alpha_power = np.sum(fft_vals[alpha_mask] ** 2) if np.any(alpha_mask) else 0
    beta_power = np.sum(fft_vals[beta_mask] ** 2) if np.any(beta_mask) else 0
    gamma_power = np.sum(fft_vals[gamma_mask] ** 2) if np.any(gamma_mask) else 0

    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power + 1e-10

    return [
        delta_power / total_power,
        theta_power / total_power,
        alpha_power / total_power,
        beta_power / total_power,
        gamma_power / total_power,
        alpha_power / (theta_power + 1e-10),
        alpha_power / (beta_power + 1e-10)
    ]


def extract_time_features(X, channel_names):
    """Extract time-domain statistical features from EEG sequences.

    Features per channel:
    - Mean, Std, Variance
    - Min, Max, Range
    - Skewness, Kurtosis
    - Zero-crossing rate
    - Energy
    - Hjorth parameters (activity, mobility, complexity)

    Total: 13 features per channel
    """
    print("Extracting time-domain features...")
    n_samples = X.shape[0]
    n_channels = X.shape[2]

    feature_names = []
    features_list = []

    for i in tqdm(range(n_samples), desc="Extracting features"):
        sample_features = []

        for ch in range(n_channels):
            signal = X[i, :, ch]

            # Basic statistics
            mean = np.mean(signal)
            std = np.std(signal)
            var = np.var(signal)
            min_val = np.min(signal)
            max_val = np.max(signal)
            range_val = max_val - min_val

            # Higher-order statistics
            if std < 1e-10:
                skew = 0.0
                kurt = 0.0
            else:
                skew = stats.skew(signal)
                kurt = stats.kurtosis(signal)
                if np.isnan(skew) or np.isinf(skew):
                    skew = 0.0
                if np.isnan(kurt) or np.isinf(kurt):
                    kurt = 0.0

            # Zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(signal - mean)))) / 2
            zcr = zero_crossings / len(signal)

            # Energy
            energy = np.sum(signal ** 2) / len(signal)

            # Hjorth parameters
            diff1 = np.diff(signal)
            diff2 = np.diff(diff1)

            activity = var
            mobility = np.std(diff1) / (std + 1e-10)
            complexity = (np.std(diff2) / (np.std(diff1) + 1e-10)) / (mobility + 1e-10)

            sample_features.extend([mean, std, var, min_val, max_val, range_val,
                                   skew, kurt, zcr, energy, activity, mobility, complexity])

            # Add frequency features
            freq_feats = extract_frequency_features(signal, SAMPLING_RATE)
            sample_features.extend(freq_feats)

            if i == 0:
                ch_name = channel_names[ch] if ch < len(channel_names) else f'CH{ch}'
                feature_names.extend([
                    f'{ch_name}_mean', f'{ch_name}_std', f'{ch_name}_var',
                    f'{ch_name}_min', f'{ch_name}_max', f'{ch_name}_range',
                    f'{ch_name}_skew', f'{ch_name}_kurt', f'{ch_name}_zcr',
                    f'{ch_name}_energy', f'{ch_name}_activity',
                    f'{ch_name}_mobility', f'{ch_name}_complexity',
                    f'{ch_name}_delta', f'{ch_name}_theta', f'{ch_name}_alpha',
                    f'{ch_name}_beta', f'{ch_name}_gamma',
                    f'{ch_name}_alpha_theta', f'{ch_name}_alpha_beta'
                ])

        features_list.append(sample_features)

    features = np.array(features_list)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, feature_names


def train_svm(X_train, y_train, X_val, y_val, max_samples=50000):
    """Train SVM classifier with RBF kernel.

    Note: SVM has O(n²) complexity, so we subsample for large datasets.
    """
    print("\n" + "=" * 60)
    print("Training SVM Classifier")
    print("=" * 60)

    # Subsample if dataset is too large (SVM is O(n²))
    if len(X_train) > max_samples:
        print(f"Subsampling training data: {len(X_train)} -> {max_samples} samples")
        np.random.seed(42)
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Compute class weight
    n_samples = len(y_train_sub)
    n_class_0 = np.sum(y_train_sub == 0)
    n_class_1 = np.sum(y_train_sub == 1)
    class_weight = {0: n_samples / (2 * n_class_0), 1: n_samples / (2 * n_class_1)}

    # Reduced grid for faster training
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale']
    }

    best_score = 0
    best_params = None
    best_model = None

    val_data = X_val if X_val is not None else X_train_sub
    val_labels = y_val if y_val is not None else y_train_sub

    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            print(f"  Testing C={C}, gamma={gamma}...")
            model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True,
                       random_state=42, class_weight=class_weight, max_iter=10000)
            model.fit(X_train_sub, y_train_sub)
            score = model.score(val_data, val_labels)

            if score > best_score:
                best_score = score
                best_params = {'C': C, 'gamma': gamma}
                best_model = model

    print(f"Best SVM: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"Best Score: {best_score:.4f}")

    return best_model, best_params


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier."""
    print("\n" + "=" * 60)
    print("Training Random Forest Classifier")
    print("=" * 60)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    best_score = 0
    best_params = None
    best_model = None

    val_data = X_val if X_val is not None else X_train
    val_labels = y_val if y_val is not None else y_train

    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=min_split,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                score = model.score(val_data, val_labels)

                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': min_split
                    }
                    best_model = model

    print(f"Best RF: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}")
    print(f"Best Score: {best_score:.4f}")

    return best_model, best_params


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    print("\n" + "=" * 60)
    print("Training XGBoost/GradientBoosting Classifier")
    print("=" * 60)

    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)
    scale_pos_weight = n_class_0 / n_class_1 if n_class_1 > 0 else 1

    val_data = X_val if X_val is not None else X_train
    val_labels = y_val if y_val is not None else y_train

    if HAS_XGBOOST:
        # Use GPU for XGBoost if available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("Using GPU for XGBoost training...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }

        best_score = 0
        best_params = None
        best_model = None

        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    model = XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        eval_metric='logloss',
                        verbosity=0,
                        tree_method='hist',  # Use histogram-based method
                        device='cuda' if use_gpu else 'cpu'  # GPU acceleration
                    )
                    model.fit(X_train, y_train)
                    score = model.score(val_data, val_labels)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'learning_rate': lr
                        }
                        best_model = model
    else:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }

        best_score = 0
        best_params = None
        best_model = None

        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    model = GradientBoostingClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    score = model.score(val_data, val_labels)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'learning_rate': lr
                        }
                        best_model = model

    print(f"Best XGB: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, lr={best_params['learning_rate']}")
    print(f"Best Score: {best_score:.4f}")

    return best_model, best_params


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation with bootstrap CI."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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

    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC:   N/A")
    print(f"  MCC:       {mcc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        bootstrap_accs.append(accuracy_score(y_test[indices], y_pred[indices]))

    ci_low = np.percentile(bootstrap_accs, 2.5)
    ci_high = np.percentile(bootstrap_accs, 97.5)

    print(f"\n95% CI for Accuracy: [{ci_low:.4f}, {ci_high:.4f}]")

    results = {
        'model_name': model_name,
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

    return results


def plot_comparison(results_dict, output_path):
    """Plot comparison of all baseline models."""
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar chart of metrics
    ax = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)]

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results_dict[model].get(m, 0) or 0 for m in metrics]
        ax.bar(x + i * width, values, width, label=model, color=color)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Key Metrics', fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1.1])

    # 2. Accuracy with CI
    ax = axes[0, 1]
    accuracies = [results_dict[m]['accuracy'] for m in models]
    ci_lows = [results_dict[m]['accuracy_ci_low'] for m in models]
    ci_highs = [results_dict[m]['accuracy_ci_high'] for m in models]
    errors = [[a - l for a, l in zip(accuracies, ci_lows)],
              [h - a for a, h in zip(accuracies, ci_highs)]]

    bars = ax.bar(models, accuracies, color=colors, edgecolor='black',
                  yerr=errors, capsize=5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy with 95% Bootstrap CI', fontweight='bold')
    ax.set_ylim([0, 1.1])
    plt.xticks(rotation=15, ha='right')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
               f'{acc:.3f}', ha='center', fontsize=10)

    # 3. Confusion matrix for best model
    ax = axes[1, 0]
    best_model = max(models, key=lambda m: results_dict[m]['accuracy'])
    cm = np.array(results_dict[best_model]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Open', 'Closed'],
               yticklabels=['Open', 'Closed'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {best_model}', fontweight='bold')

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for model in models:
        r = results_dict[model]
        table_data.append([
            model,
            f"{r['accuracy']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['mcc']:.3f}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Model Performance Summary', fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_path / 'fig08_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig08_baseline_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig08_baseline_comparison.png/pdf")


def save_results(results_dict, feature_names, output_path):
    """Save baseline results to files."""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    results_serializable = {}
    for model_name, results in results_dict.items():
        results_serializable[model_name] = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in results.items()
        }

    with open(output_path / 'baseline_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Saved: baseline_results.json")

    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"Saved: feature_names.json")

    summary = []
    for model_name, results in results_dict.items():
        summary.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1': f"{results['f1']:.4f}",
            'MCC': f"{results['mcc']:.4f}",
            '95% CI': f"[{results['accuracy_ci_low']:.4f}, {results['accuracy_ci_high']:.4f}]"
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path / 'baseline_summary.csv', index=False)
    print(f"Saved: baseline_summary.csv")

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("BASELINE MODELS TRAINING")
    print("OpenNeuro ds004148")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("Models: SVM, Random Forest, XGBoost")
    print("=" * 60)

    # Check for cached features first (much faster than loading raw data)
    cache_file = PROCESSED_DATA_PATH / 'extracted_features.npz'
    labels_file = PROCESSED_DATA_PATH / 'labels.npz'

    if cache_file.exists() and labels_file.exists():
        print("\n" + "=" * 60)
        print("Loading Cached Features (Fast Path)")
        print("=" * 60)

        cached = np.load(cache_file, allow_pickle=True)
        X_train_feat = cached['X_train_feat']
        X_val_feat = cached['X_val_feat'] if 'X_val_feat' in cached else None
        X_test_feat = cached['X_test_feat']
        feature_names = list(cached['feature_names'])

        labels = np.load(labels_file)
        y_train = labels['y_train']
        y_val = labels['y_val'] if 'y_val' in labels else None
        y_test = labels['y_test']

        print(f"X_train_feat: {X_train_feat.shape}")
        print(f"X_val_feat: {X_val_feat.shape if X_val_feat is not None else 'None'}")
        print(f"X_test_feat: {X_test_feat.shape}")
        print(f"Features per sample: {X_train_feat.shape[1]}")

    else:
        # Slow path: Load raw data and extract features
        print("\n" + "=" * 60)
        print("No cached features found. Loading raw data...")
        print("=" * 60)

        # 1. Load data
        X_train, y_train, X_val, y_val, X_test, y_test, channel_names = load_processed_data()

        # If no test data, split train
        if X_test is None:
            print("\nNo test data found. Using last 20% of training data as test.")
            split_idx = int(len(X_train) * 0.8)
            X_test = X_train[split_idx:]
            y_test = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

        # 2. Extract features using GPU
        print("\n" + "=" * 60)
        print("Extracting Features (GPU Accelerated)")
        print("=" * 60)

        X_train_feat, feature_names = extract_features_gpu(X_train, channel_names)
        if X_val is not None:
            X_val_feat, _ = extract_features_gpu(X_val, channel_names)
        else:
            X_val_feat = None
        X_test_feat, _ = extract_features_gpu(X_test, channel_names)

        # Cache features and labels for future runs
        print("\nCaching features and labels for future runs...")
        if X_val_feat is not None:
            np.savez_compressed(cache_file,
                               X_train_feat=X_train_feat,
                               X_val_feat=X_val_feat,
                               X_test_feat=X_test_feat,
                               feature_names=feature_names)
            np.savez_compressed(labels_file,
                               y_train=y_train,
                               y_val=y_val,
                               y_test=y_test)
        else:
            np.savez_compressed(cache_file,
                               X_train_feat=X_train_feat,
                               X_test_feat=X_test_feat,
                               feature_names=feature_names)
            np.savez_compressed(labels_file,
                               y_train=y_train,
                               y_test=y_test)
        print(f"Cached to: {cache_file}")

    print(f"Feature shape: {X_train_feat.shape}")
    print(f"Features per sample: {X_train_feat.shape[1]}")

    # 3. Scale features
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    if X_val_feat is not None:
        X_val_feat = scaler.transform(X_val_feat)
    X_test_feat = scaler.transform(X_test_feat)

    # Save scaler
    with open(MODELS_PATH / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    results_dict = {}

    # 4. Train SVM
    svm_model, svm_params = train_svm(X_train_feat, y_train, X_val_feat, y_val)
    results_dict['SVM'] = evaluate_model(svm_model, X_test_feat, y_test, 'SVM')
    results_dict['SVM']['params'] = svm_params

    with open(MODELS_PATH / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    # 5. Train Random Forest
    rf_model, rf_params = train_random_forest(X_train_feat, y_train, X_val_feat, y_val)
    results_dict['Random Forest'] = evaluate_model(rf_model, X_test_feat, y_test, 'Random Forest')
    results_dict['Random Forest']['params'] = rf_params

    with open(MODELS_PATH / 'rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    # 6. Train XGBoost
    xgb_model, xgb_params = train_xgboost(X_train_feat, y_train, X_val_feat, y_val)
    results_dict['XGBoost'] = evaluate_model(xgb_model, X_test_feat, y_test, 'XGBoost')
    results_dict['XGBoost']['params'] = xgb_params

    with open(MODELS_PATH / 'xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    # 7. Feature importance from RF
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 60)
    print("Top 15 Important Features (Random Forest)")
    print("=" * 60)
    print(feature_importance.head(15).to_string(index=False))

    feature_importance.to_csv(RESULTS_PATH / 'rf_feature_importance.csv', index=False)

    # 8. Generate plots
    plot_comparison(results_dict, FIGURES_PATH)

    # 9. Save results
    save_results(results_dict, feature_names, RESULTS_PATH)

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE!")
    print("=" * 60)

    return results_dict, feature_names


if __name__ == "__main__":
    results, features = main()
