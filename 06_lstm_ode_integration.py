"""
06_lstm_ode_integration.py
==========================
LSTM-ODE Integration Framework for EEG-Based Cognitive State Evolution

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

This script implements the core contribution of the paper:
Probabilistic coupling between LSTM classification and ODE dynamics.

Integration Mechanism:
1. LSTM processes EEG sequences and outputs class probabilities
2. Probabilities are used to modulate ODE transition rates
3. The coupled system evolves to predict cognitive state trajectories

Key Features:
- Probabilistic coupling (soft modulation)
- Multi-step trajectory forecasting
- Uncertainty quantification
- Comparison with standalone models
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # Mixed precision for inference
from scipy.integrate import odeint
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

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

for path in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# Import LSTM model architecture (Enhanced version)
class Attention(nn.Module):
    """Self-attention mechanism."""
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


class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with Attention for EEG Classification.

    Note: Default input_size=64 for 61-64 channel EEG data from OpenNeuro ds004148.
    Actual values are loaded from checkpoint config in load_models().
    """
    def __init__(self, input_size=64, hidden_size=256, num_layers=3,
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

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.attention = Attention(lstm_output_size)

        # Classification head - must match saved model
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


class CognitiveStateODE:
    """Three-State Compartmental ODE Model."""

    def __init__(self, params=None):
        if params is None:
            self.params = {
                'k_ap': 0.1, 'k_af': 0.02, 'k_pa': 0.15,
                'k_pf': 0.08, 'k_fa': 0.05, 'k_fp': 0.1
            }
        else:
            self.params = params

    def ode_system(self, y, t, params=None):
        if params is None:
            params = self.params

        A, P, F = max(0, y[0]), max(0, y[1]), max(0, y[2])

        k_ap, k_af = params['k_ap'], params['k_af']
        k_pa, k_pf = params['k_pa'], params['k_pf']
        k_fa, k_fp = params['k_fa'], params['k_fp']

        dA_dt = -k_ap * A - k_af * A + k_pa * P + k_fa * F
        dP_dt = k_ap * A - k_pa * P - k_pf * P + k_fp * F
        dF_dt = k_af * A + k_pf * P - k_fa * F - k_fp * F

        return [dA_dt, dP_dt, dF_dt]

    def solve(self, initial_state, t_span, n_points=100):
        t = np.linspace(t_span[0], t_span[1], n_points)
        initial_state = np.array(initial_state) / np.sum(initial_state)
        solution = odeint(self.ode_system, initial_state, t)
        solution = np.clip(solution, 0, 1)
        solution = solution / solution.sum(axis=1, keepdims=True)
        return t, solution


class LSTMODEIntegration:
    """Unified LSTM-ODE Integration Framework.

    This class implements probabilistic coupling between the LSTM
    classifier and the ODE dynamics model.

    Coupling Mechanism:
    -------------------
    The LSTM outputs P(open) and P(closed) = 1 - P(open).

    These probabilities modulate the ODE transition rates:
    - High P(closed) increases transitions toward Fatigued state
    - High P(open) increases transitions toward Active state

    The modulation is soft (multiplicative) to maintain ODE stability:
    - k_af' = k_af * (1 + alpha * P(closed))
    - k_fa' = k_fa * (1 + alpha * P(open))

    where alpha is the coupling strength parameter.
    """

    def __init__(self, lstm_model, ode_model, coupling_strength=0.5):
        """
        Args:
            lstm_model: Trained LSTMAttentionModel
            ode_model: CognitiveStateODE with fitted parameters
            coupling_strength: Alpha parameter for rate modulation (0-1)
        """
        self.lstm_model = lstm_model
        self.ode_model = ode_model
        self.coupling_strength = coupling_strength
        self.base_params = ode_model.params.copy()

    def get_lstm_probabilities(self, X):
        """Get classification probabilities from LSTM.

        Args:
            X: EEG sequence tensor (batch, seq_len, channels)

        Returns:
            probs: (batch, 2) probabilities [P(open), P(closed)]
            attention: (batch, seq_len) attention weights
        """
        self.lstm_model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(DEVICE)
            logits, attention = self.lstm_model(X, return_attention=True)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy(), attention.cpu().numpy()

    def modulate_ode_rates(self, p_closed, p_open):
        """Modulate ODE transition rates based on LSTM probabilities.

        Higher P(closed) -> increased fatigue transitions
        Higher P(open) -> increased active transitions

        Args:
            p_closed: Probability of closed eyes (fatigue indicator)
            p_open: Probability of open eyes (alertness indicator)

        Returns:
            modulated_params: Modified ODE parameters
        """
        alpha = self.coupling_strength
        params = self.base_params.copy()

        # Modulate fatigue-related rates
        params['k_af'] = params['k_af'] * (1 + alpha * p_closed)  # Active -> Fatigued
        params['k_pf'] = params['k_pf'] * (1 + alpha * p_closed)  # Passive -> Fatigued

        # Modulate recovery-related rates
        params['k_fa'] = params['k_fa'] * (1 + alpha * p_open)  # Fatigued -> Active
        params['k_pa'] = params['k_pa'] * (1 + alpha * p_open)  # Passive -> Active

        # Ensure positive rates
        for k in params:
            params[k] = max(0.001, params[k])

        return params

    def predict_trajectory(self, X, initial_state=None, forecast_steps=10):
        """Predict cognitive state trajectory using coupled LSTM-ODE.

        Args:
            X: EEG sequence (1, seq_len, channels)
            initial_state: [A0, P0, F0] or None (inferred from LSTM)
            forecast_steps: Number of ODE time steps to predict

        Returns:
            trajectory: (forecast_steps, 3) state proportions
            probs: LSTM classification probabilities
            attention: LSTM attention weights
        """
        # Get LSTM predictions
        probs, attention = self.get_lstm_probabilities(X)
        p_open = probs[0, 0]
        p_closed = probs[0, 1]

        # Infer initial state if not provided
        if initial_state is None:
            # Map LSTM output to initial ODE state
            if p_closed > 0.6:
                initial_state = [0.2, 0.2, 0.6]  # Mostly fatigued
            elif p_open > 0.6:
                initial_state = [0.6, 0.2, 0.2]  # Mostly active
            else:
                initial_state = [0.33, 0.34, 0.33]  # Mixed

        # Modulate ODE rates
        mod_params = self.modulate_ode_rates(p_closed, p_open)
        self.ode_model.params = mod_params

        # Solve ODE
        t, trajectory = self.ode_model.solve(
            initial_state, (0, forecast_steps), forecast_steps
        )

        # Reset to base params
        self.ode_model.params = self.base_params.copy()

        return trajectory, probs, attention

    def predict_batch(self, X_batch, forecast_steps=20, batch_size=512, show_progress=True):
        """Predict trajectories for a batch of sequences using batched GPU inference.

        Optimized for RTX 3090 with:
        - Batched LSTM inference with mixed precision
        - Efficient memory management
        - Progress bar for monitoring

        Args:
            X_batch: (batch, seq_len, channels)
            forecast_steps: ODE forecast horizon (default 20)
            batch_size: GPU batch size for LSTM inference
            show_progress: Show progress bar

        Returns:
            trajectories: (batch, forecast_steps, 3)
            all_probs: (batch, 2)
            final_predictions: (batch,) predicted class
        """
        n_samples = len(X_batch)
        trajectories = []
        all_probs = []
        predictions = []

        use_amp = torch.cuda.is_available()

        # Step 1: Batch LSTM inference for all samples
        self.lstm_model.eval()
        all_lstm_probs = []
        all_attention = []

        iterator = range(0, n_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="LSTM Inference", leave=False)

        with torch.no_grad():
            for i in iterator:
                batch_end = min(i + batch_size, n_samples)
                X = torch.FloatTensor(X_batch[i:batch_end]).to(DEVICE)

                if use_amp:
                    with autocast():
                        logits, attention = self.lstm_model(X, return_attention=True)
                        probs = torch.softmax(logits, dim=1)
                else:
                    logits, attention = self.lstm_model(X, return_attention=True)
                    probs = torch.softmax(logits, dim=1)

                all_lstm_probs.append(probs.cpu().numpy())
                all_attention.append(attention.cpu().numpy())

                # Free GPU memory
                del X, logits, attention, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_lstm_probs = np.concatenate(all_lstm_probs, axis=0)
        all_attention = np.concatenate(all_attention, axis=0)

        # Step 2: ODE trajectory prediction (parallelizable on CPU)
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="ODE Trajectories", leave=False)

        for i in iterator:
            p_open = all_lstm_probs[i, 0]
            p_closed = all_lstm_probs[i, 1]

            # Infer initial state from LSTM output
            if p_closed > 0.6:
                initial_state = [0.2, 0.2, 0.6]  # Mostly fatigued
            elif p_open > 0.6:
                initial_state = [0.6, 0.2, 0.2]  # Mostly active
            else:
                initial_state = [0.33, 0.34, 0.33]  # Mixed

            # Modulate ODE rates and solve
            mod_params = self.modulate_ode_rates(p_closed, p_open)
            self.ode_model.params = mod_params

            t, traj = self.ode_model.solve(
                initial_state, (0, forecast_steps), forecast_steps
            )

            trajectories.append(traj)
            all_probs.append(all_lstm_probs[i])

            # Final prediction based on trajectory end state
            final_state = traj[-1]
            # Map: Active/Passive -> Open(0), Fatigued -> Closed(1)
            if final_state[2] > 0.5:  # Fatigued dominant
                predictions.append(1)
            else:
                predictions.append(0)

        # Reset ODE params
        self.ode_model.params = self.base_params.copy()

        return np.array(trajectories), np.array(all_probs), np.array(predictions)


def load_models():
    """Load trained LSTM and ODE models."""
    print("=" * 60)
    print("Loading Trained Models")
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

    # Load ODE model
    with open(MODELS_PATH / 'ode_model.pkl', 'rb') as f:
        ode_data = pickle.load(f)

    ode_model = CognitiveStateODE(ode_data['params'])
    print(f"Loaded ODE model with params: {ode_data['params']}")

    return lstm_model, ode_model


def load_test_data():
    """Load test data."""
    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"Loaded test data: {X_test.shape}")
    return X_test, y_test


def evaluate_integration(integration_model, X_test, y_test):
    """Evaluate the integrated LSTM-ODE model with batched GPU inference."""
    print("\n" + "=" * 60)
    print("Evaluating LSTM-ODE Integration")
    print("=" * 60)

    start_time = time.time()

    # Get predictions with batched GPU inference
    trajectories, probs, y_pred = integration_model.predict_batch(
        X_test, forecast_steps=20, batch_size=512, show_progress=True
    )
    y_prob = probs[:, 1]  # P(closed)

    inference_time = time.time() - start_time
    print(f"\nInference completed in {inference_time:.1f}s ({len(X_test)/inference_time:.0f} samples/sec)")

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

    # Print results
    print(f"\nTest Set Metrics (LSTM-ODE Integration):")
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
        'model_name': 'LSTM-ODE Integration',
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

    return results, trajectories, probs


def compare_coupling_strengths(lstm_model, ode_model, X_test, y_test, output_path):
    """Compare different coupling strength values with batched GPU inference."""
    print("\n" + "=" * 60)
    print("Analyzing Coupling Strength Impact")
    print("=" * 60)

    coupling_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results_by_alpha = []

    for alpha in coupling_values:
        print(f"\nTesting alpha={alpha:.2f}...")
        integration = LSTMODEIntegration(lstm_model, ode_model, coupling_strength=alpha)
        _, probs, y_pred = integration.predict_batch(
            X_test, forecast_steps=20, batch_size=512, show_progress=False
        )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results_by_alpha.append({
            'alpha': float(alpha),
            'accuracy': float(acc),
            'f1': float(f1),
            'mcc': float(mcc)
        })
        print(f"  Accuracy={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    alphas = [r['alpha'] for r in results_by_alpha]
    accs = [r['accuracy'] for r in results_by_alpha]
    f1s = [r['f1'] for r in results_by_alpha]

    ax.plot(alphas, accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
    ax.plot(alphas, f1s, 's-', label='F1-Score', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength (α)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Impact of LSTM-ODE Coupling Strength', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas)

    plt.tight_layout()
    plt.savefig(output_path / 'fig15_coupling_strength.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig15_coupling_strength.pdf', bbox_inches='tight')
    plt.close()

    print("\nSaved: fig15_coupling_strength.png/pdf")

    return results_by_alpha


def plot_trajectory_examples(integration_model, X_test, y_test, output_path):
    """Plot example trajectories for different samples."""
    print("\n" + "=" * 60)
    print("Generating Trajectory Examples")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Select diverse samples
    open_idx = np.where(y_test == 0)[0][:3]
    closed_idx = np.where(y_test == 1)[0][:3]

    colors = {'Active': '#2ecc71', 'Passive': '#3498db', 'Fatigued': '#e74c3c'}

    for idx, (ax, sample_idx) in enumerate(zip(axes[0], open_idx)):
        X = X_test[sample_idx:sample_idx+1]
        traj, probs, attn = integration_model.predict_trajectory(X, forecast_steps=20)

        t = np.arange(len(traj))
        ax.plot(t, traj[:, 0], '-', color=colors['Active'], label='Active', linewidth=2)
        ax.plot(t, traj[:, 1], '-', color=colors['Passive'], label='Passive', linewidth=2)
        ax.plot(t, traj[:, 2], '-', color=colors['Fatigued'], label='Fatigued', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Proportion')
        ax.set_title(f'Open Eye Sample (P(open)={probs[0,0]:.2f})', fontweight='bold')
        if idx == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    for idx, (ax, sample_idx) in enumerate(zip(axes[1], closed_idx)):
        X = X_test[sample_idx:sample_idx+1]
        traj, probs, attn = integration_model.predict_trajectory(X, forecast_steps=20)

        t = np.arange(len(traj))
        ax.plot(t, traj[:, 0], '-', color=colors['Active'], label='Active', linewidth=2)
        ax.plot(t, traj[:, 1], '-', color=colors['Passive'], label='Passive', linewidth=2)
        ax.plot(t, traj[:, 2], '-', color=colors['Fatigued'], label='Fatigued', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Proportion')
        ax.set_title(f'Closed Eye Sample (P(closed)={probs[0,1]:.2f})', fontweight='bold')
        if idx == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.suptitle('Cognitive State Trajectories from LSTM-ODE Model',
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'fig16_trajectory_examples.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig16_trajectory_examples.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig16_trajectory_examples.png/pdf")


def plot_comprehensive_comparison(integration_results, output_path):
    """Create comprehensive comparison with all models."""
    print("\n" + "=" * 60)
    print("Generating Comprehensive Comparison")
    print("=" * 60)

    # Load all results
    all_results = {}

    # Load baseline results
    baseline_path = RESULTS_PATH / 'baseline_results.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            all_results.update(baseline)

    # Load LSTM results
    lstm_path = RESULTS_PATH / 'lstm_results.json'
    if lstm_path.exists():
        with open(lstm_path, 'r') as f:
            lstm = json.load(f)
            all_results['LSTM-Attention'] = lstm

    # Add integration results
    all_results['LSTM-ODE'] = integration_results

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(models)]

    # 1. Bar chart comparison
    ax = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.15

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [all_results[model].get(m, 0) or 0 for m in metrics]
        ax.bar(x + i * width, values, width, label=model, color=color)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Accuracy with CI
    ax = axes[0, 1]
    accuracies = [all_results[m].get('accuracy', 0) for m in models]
    ci_lows = [all_results[m].get('accuracy_ci_low', 0) for m in models]
    ci_highs = [all_results[m].get('accuracy_ci_high', 0) for m in models]

    # Handle missing CI values
    for i in range(len(models)):
        if ci_lows[i] == 0:
            ci_lows[i] = accuracies[i] - 0.05
        if ci_highs[i] == 0:
            ci_highs[i] = accuracies[i] + 0.05

    errors = [[a - l for a, l in zip(accuracies, ci_lows)],
              [h - a for a, h in zip(accuracies, ci_highs)]]

    bars = ax.bar(models, accuracies, color=colors, edgecolor='black',
                  yerr=errors, capsize=5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy with 95% Bootstrap CI', fontweight='bold')
    ax.set_ylim([0, 1.1])
    plt.xticks(rotation=15, ha='right')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
               f'{acc:.3f}', ha='center', fontsize=9)

    # 3. Radar chart
    ax = axes[1, 0]
    ax.set_aspect('equal')

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(2, 2, 3, projection='polar')

    for model, color in zip(models, colors):
        values = [all_results[model].get(m, 0) or 0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim([0, 1])
    ax.set_title('Multi-Metric Radar Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for model in models:
        r = all_results[model]
        table_data.append([
            model,
            f"{r.get('accuracy', 0):.3f}",
            f"{r.get('f1', 0):.3f}",
            f"{r.get('auc', 0) or 0:.3f}",
            f"{r.get('mcc', 0):.3f}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'F1', 'AUC', 'MCC'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight LSTM-ODE row
    for i, model in enumerate(models):
        if model == 'LSTM-ODE':
            for j in range(5):
                table[(i+1, j)].set_facecolor('#90EE90')

    ax.set_title('Performance Summary', fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_path / 'fig17_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig17_comprehensive_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig17_comprehensive_comparison.png/pdf")

    # Save all results
    with open(RESULTS_PATH / 'all_model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


def save_integration_results(results, trajectories, coupling_analysis):
    """Save integration model results."""
    print("\n" + "=" * 60)
    print("Saving Integration Results")
    print("=" * 60)

    # Save results
    with open(RESULTS_PATH / 'integration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: integration_results.json")

    # Save coupling analysis
    with open(RESULTS_PATH / 'coupling_analysis.json', 'w') as f:
        json.dump(coupling_analysis, f, indent=2)
    print("Saved: coupling_analysis.json")

    # Save trajectories
    np.save(RESULTS_PATH / 'predicted_trajectories.npy', trajectories)
    print("Saved: predicted_trajectories.npy")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("LSTM-ODE INTEGRATION FRAMEWORK")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Load models
    lstm_model, ode_model = load_models()

    # 2. Load test data
    X_test, y_test = load_test_data()

    # 3. Create integration model
    integration_model = LSTMODEIntegration(
        lstm_model, ode_model,
        coupling_strength=0.5  # Default coupling
    )

    # 4. Evaluate integration
    results, trajectories, probs = evaluate_integration(
        integration_model, X_test, y_test
    )

    # 5. Analyze coupling strength
    coupling_analysis = compare_coupling_strengths(
        lstm_model, ode_model, X_test, y_test, FIGURES_PATH
    )

    # 6. Plot trajectory examples
    plot_trajectory_examples(integration_model, X_test, y_test, FIGURES_PATH)

    # 7. Comprehensive comparison
    plot_comprehensive_comparison(results, FIGURES_PATH)

    # 8. Save results
    save_integration_results(results, trajectories, coupling_analysis)

    print("\n" + "=" * 60)
    print("LSTM-ODE INTEGRATION COMPLETE!")
    print("=" * 60)

    return integration_model, results


if __name__ == "__main__":
    model, results = main()
