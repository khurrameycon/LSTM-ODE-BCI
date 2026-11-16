"""
08_forecasting.py
==================
Multi-Step Trajectory Forecasting with LSTM-ODE Framework

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

This module implements multi-step ahead prediction using:
1. LSTM probability sequences as input
2. ODE dynamics for state evolution forecasting
3. Hybrid LSTM-ODE trajectory prediction

Key Features:
- Multi-horizon forecasting (5, 10, 20 steps ahead)
- Rolling prediction evaluation
- Trajectory uncertainty quantification
- Clinical state transition prediction
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

from revision_common import initial_state_from_probs

warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / 'outputs'
PROCESSED_DATA_PATH = OUTPUT_PATH / 'processed_data'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'


# ============================================================
# Model Architecture (must match training)
# ============================================================
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
    """Enhanced Bidirectional LSTM with Attention."""
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
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context, attn_weights = self.attention(lstm_out)
        output = self.classifier(context)
        if return_attention:
            return output, attn_weights
        return output


# ============================================================
# ODE System for State Evolution
# ============================================================
def cognitive_state_ode(y, t, params):
    """Three-compartment ODE for cognitive state dynamics."""
    A, P, F = y
    k_ap = params['k_ap']
    k_af = params['k_af']
    k_pa = params['k_pa']
    k_pf = params['k_pf']
    k_fa = params['k_fa']
    k_fp = params['k_fp']

    dA = -k_ap * A - k_af * A + k_pa * P + k_fa * F
    dP = k_ap * A - k_pa * P - k_pf * P + k_fp * F
    dF = k_af * A + k_pf * P - k_fa * F - k_fp * F

    return [dA, dP, dF]


def predict_trajectory(initial_state, params, n_steps, dt=1.0):
    """Predict ODE trajectory from initial state."""
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    trajectory = odeint(cognitive_state_ode, initial_state, t, args=(params,))
    return trajectory


# ============================================================
# Forecasting Functions
# ============================================================
def load_resources():
    """Load trained models and data."""
    print("=" * 60)
    print("Loading Models and Data for Forecasting")
    print("=" * 60)

    # Load LSTM model
    checkpoint = torch.load(MODELS_PATH / 'lstm_attention_model.pt',
                           map_location=DEVICE, weights_only=False)
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
    ode_params = {k: float(v) for k, v in ode_data['params'].items()}
    print("Loaded ODE parameters")

    # Load test data
    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    recording_ids = data['recording_id_test'] if 'recording_id_test' in data.files else None
    print(f"Loaded test data: {X_test.shape}")

    return lstm_model, ode_params, X_test, y_test, recording_ids


def get_lstm_probabilities(lstm_model, X_data, batch_size=256):
    """Get LSTM probability predictions for all samples."""
    lstm_model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch = torch.FloatTensor(X_data[i:i+batch_size]).to(DEVICE)
            outputs = lstm_model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            del batch
            torch.cuda.empty_cache()

    return np.vstack(all_probs)


def prob_to_ode_state(prob_closed, prob_open=None):
    """Initial state used by every other script.

    The forecast target is defined in `multistep_forecast`: future
    P(eyes closed), and the eye-state label of that same future window.
    """
    if prob_open is None:
        prob_open = 1.0 - prob_closed
    return initial_state_from_probs(prob_open, prob_closed)


def create_sequences_for_forecasting(probs, y_labels, window_size=10):
    """Create sequences of LSTM probabilities for trajectory prediction."""
    sequences = []
    labels = []
    future_labels = []

    for i in range(len(probs) - window_size):
        seq = probs[i:i+window_size, 1]  # P(closed) sequence
        sequences.append(seq)
        labels.append(y_labels[i:i+window_size])
        future_labels.append(y_labels[i+window_size] if i + window_size < len(y_labels) else y_labels[-1])

    return np.array(sequences), np.array(labels), np.array(future_labels)


# One window step is sequence_length * (1 - overlap) / sampling_rate.
# The preprocessing metadata records the exact value; 0.256 s is the default
# 256-sample, 50% overlap, 500 Hz window.
DEFAULT_STEP_SECONDS = 0.256
FORECAST_HORIZONS_S = (2.5, 5.0, 10.0)


def _step_seconds():
    meta_path = PROCESSED_DATA_PATH / 'preprocessing_metadata.json'
    if meta_path.exists():
        with open(meta_path) as handle:
            return float(json.load(handle).get('step_seconds', DEFAULT_STEP_SECONDS))
    return DEFAULT_STEP_SECONDS


def _horizon_steps(step_seconds):
    return [max(1, int(round(seconds / step_seconds))) for seconds in FORECAST_HORIZONS_S]


def multistep_forecast(probs, y_labels, ode_params, horizons, recording_ids=None,
                       step_seconds=DEFAULT_STEP_SECONDS):
    """Forecast future P(eyes closed) and the future eye-state label.

    Target, for a horizon of h windows that stay inside one recording:
      - probability target: LSTM P(eyes closed) at the future window
      - label target: the eye-state label of that window (0 open, 1 closed)
    Pairs that cross a recording boundary are excluded. The persistence
    baseline carries the current probability and the current label forward.
    """
    print("\n" + "=" * 60)
    print("Multi-Step Trajectory Forecasting")
    print("=" * 60)
    print("Target: future P(eyes closed) and the eye-state label of that window")
    print(f"Horizons (windows): {horizons}")

    results = {
        h: {
            'predictions': [], 'actual_probs': [], 'actual_labels': [],
            'persistence_probs': [], 'persistence_labels': [], 'current_labels': [],
        }
        for h in horizons
    }
    max_horizon = max(horizons)
    n_excluded = 0
    n_scored = 0

    for i in tqdm(range(len(probs) - max_horizon), desc="Forecasting"):
        initial_state = prob_to_ode_state(probs[i, 1], probs[i, 0])
        trajectory = predict_trajectory(
            initial_state, ode_params, max_horizon, dt=step_seconds
        )
        for h in horizons:
            if recording_ids is not None and recording_ids[i] != recording_ids[i + h]:
                n_excluded += 1
                continue
            n_scored += 1
            # Fatigued carries the closed observation; Passive is not a class label.
            predicted_closed = float(np.clip(trajectory[h, 2], 0.0, 1.0))
            bucket = results[h]
            bucket['predictions'].append(predicted_closed)
            bucket['actual_probs'].append(float(probs[i + h, 1]))
            bucket['actual_labels'].append(int(y_labels[i + h]))
            bucket['persistence_probs'].append(float(probs[i, 1]))
            bucket['persistence_labels'].append(int(y_labels[i]))
            bucket['current_labels'].append(int(y_labels[i]))

    for h in horizons:
        for key, value in results[h].items():
            results[h][key] = np.asarray(value)
    results['_counts'] = {'scored_pairs': n_scored, 'excluded_cross_recording_pairs': n_excluded}
    print(f"Scored pairs: {n_scored}; excluded because they cross a recording: {n_excluded}")
    return results


def _score_block(pred_probs, actual_probs, pred_labels, actual_labels):
    pred_labels = np.asarray(pred_labels).astype(int)
    actual_labels = np.asarray(actual_labels).astype(int)
    accuracy = float(np.mean(pred_labels == actual_labels)) if len(actual_labels) else float('nan')
    mae = float(np.mean(np.abs(pred_probs - actual_probs))) if len(actual_probs) else float('nan')
    if len(actual_probs) > 2 and np.std(pred_probs) > 0 and np.std(actual_probs) > 0:
        corr, _ = spearmanr(pred_probs, actual_probs)
        corr = float(corr)
    else:
        corr = float('nan')
    return {'accuracy_vs_eye_state_label': accuracy, 'mae_vs_future_probability': mae,
            'spearman_vs_future_probability': corr, 'n_samples': int(len(actual_labels))}


def evaluate_forecasts(forecast_results, horizons, step_seconds):
    """Score the ODE forecast and the persistence baseline on the same pairs."""
    print("\n" + "=" * 60)
    print("Forecasting Evaluation")
    print("=" * 60)
    print("Accuracy is agreement with the future eye-state label.")
    print("MAE and Spearman compare probabilities with future P(eyes closed).")
    print("Persistence repeats the current probability and the current label.")

    metrics = {}
    for index, h in enumerate(horizons):
        block = forecast_results[h]
        ode = _score_block(
            block['predictions'], block['actual_probs'],
            (block['predictions'] >= 0.5).astype(int), block['actual_labels'],
        )
        persistence = _score_block(
            block['persistence_probs'], block['actual_probs'],
            block['persistence_labels'], block['actual_labels'],
        )
        seconds = FORECAST_HORIZONS_S[index]
        metrics[str(h)] = {
            'horizon_windows': int(h),
            'horizon_seconds': float(seconds),
            'step_seconds': float(step_seconds),
            'ode': ode,
            'persistence': persistence,
            'ode_minus_persistence_accuracy': ode['accuracy_vs_eye_state_label'] - persistence['accuracy_vs_eye_state_label'],
        }
        print(f"\n{seconds:.1f}s ({h} windows):")
        print(f"  ODE accuracy:         {ode['accuracy_vs_eye_state_label']:.4f}")
        print(f"  Persistence accuracy: {persistence['accuracy_vs_eye_state_label']:.4f}")
        print(f"  ODE MAE:              {ode['mae_vs_future_probability']:.4f}")
        print(f"  Persistence MAE:      {persistence['mae_vs_future_probability']:.4f}")
    return metrics


def rolling_forecast_evaluation(probs, ode_params, window_size=50, horizon=10):
    """Evaluate forecasting with rolling windows."""
    print("\n" + "=" * 60)
    print("Rolling Window Forecast Evaluation")
    print("=" * 60)

    n_windows = (len(probs) - window_size - horizon) // window_size
    print(f"Number of rolling windows: {n_windows}")

    window_metrics = []

    for w in tqdm(range(n_windows), desc="Rolling windows"):
        start_idx = w * window_size
        end_idx = start_idx + window_size

        # Get initial states for window
        window_predictions = []
        window_actuals = []

        for i in range(start_idx, end_idx):
            if i + horizon >= len(probs):
                break

            prob_closed = probs[i, 1]
            initial_state = prob_to_ode_state(prob_closed)
            trajectory = predict_trajectory(initial_state, ode_params, horizon)

            predicted_closed = trajectory[horizon, 2] + trajectory[horizon, 1] * 0.5
            predicted_closed = np.clip(predicted_closed, 0, 1)

            window_predictions.append(predicted_closed)
            window_actuals.append(probs[i + horizon, 1])

        if len(window_predictions) > 0:
            preds = np.array(window_predictions)
            actuals = np.array(window_actuals)

            accuracy = np.mean((preds > 0.5) == (actuals > 0.5))
            mae = np.mean(np.abs(preds - actuals))

            window_metrics.append({
                'window': w,
                'accuracy': accuracy,
                'mae': mae
            })

    return pd.DataFrame(window_metrics)


def plot_forecasting_results(forecast_results, metrics, horizons, output_path):
    """Generate forecasting visualization plots."""
    print("\n" + "=" * 60)
    print("Generating Forecasting Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 14))

    # 1. Metrics comparison across horizons
    ax1 = fig.add_subplot(2, 3, 1)
    metric_names = ['accuracy', 'mae', 'correlation']
    x = np.arange(len(horizons))
    width = 0.25

    def metric_value(horizon, name):
        block = metrics[horizon] if horizon in metrics else metrics[str(horizon)]
        ode = block['ode']
        return {
            'accuracy': ode['accuracy_vs_eye_state_label'],
            'mae': ode['mae_vs_future_probability'],
            'correlation': ode['spearman_vs_future_probability'],
        }[name]

    for i, metric in enumerate(metric_names):
        values = [metric_value(h, metric) for h in horizons]
        ax1.bar(x + i * width, values, width, label=metric.upper())

    ax1.set_xlabel('Forecast Horizon')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Forecasting Metrics by Horizon', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{h} steps' for h in horizons])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy vs Horizon trend
    ax2 = fig.add_subplot(2, 3, 2)
    accuracies = [metric_value(h, 'accuracy') for h in horizons]
    ax2.plot(horizons, accuracies, 'bo-', linewidth=2, markersize=10)
    ax2.fill_between(horizons, accuracies, alpha=0.3)
    ax2.set_xlabel('Forecast Horizon (steps)')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy Degradation with Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])

    # 3. Predicted vs Actual scatter for shortest horizon
    ax3 = fig.add_subplot(2, 3, 3)
    h = horizons[0]
    preds = forecast_results[h]['predictions'][:1000]
    actuals = forecast_results[h]['actual_probs'][:1000]

    ax3.scatter(actuals, preds, alpha=0.4, s=10, c='steelblue')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax3.set_xlabel('Actual P(Closed)')
    ax3.set_ylabel('Predicted P(Closed)')
    ax3.set_title(f'Predicted vs Actual (h={h})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Example trajectory predictions
    ax4 = fig.add_subplot(2, 3, 4)
    n_show = 100
    t = np.arange(n_show)

    h = horizons[1]  # Medium horizon
    preds = forecast_results[h]['predictions'][:n_show]
    actuals = forecast_results[h]['actual_probs'][:n_show]

    ax4.plot(t, actuals, 'b-', label='Actual', linewidth=2)
    ax4.plot(t, preds, 'r--', label=f'Predicted (h={h})', linewidth=2, alpha=0.8)
    ax4.fill_between(t, preds - 0.1, preds + 0.1, alpha=0.2, color='red')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('P(Closed Eyes)')
    ax4.set_title('Trajectory Prediction Example', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error distribution by horizon
    ax5 = fig.add_subplot(2, 3, 5)
    errors_data = []
    for h in horizons:
        preds = forecast_results[h]['predictions']
        actuals = forecast_results[h]['actual_probs']
        errors = preds - actuals
        errors_data.append(errors)

    bp = ax5.boxplot(errors_data, patch_artist=True)
    ax5.set_xticks(range(1, len(horizons) + 1))
    ax5.set_xticklabels([f'h={h}' for h in horizons])
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(horizons)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax5.set_xlabel('Forecast Horizon')
    ax5.set_ylabel('Prediction Error')
    ax5.set_title('Error Distribution by Horizon', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Correlation heatmap
    ax6 = fig.add_subplot(2, 3, 6)
    corr_data = []
    for h in horizons:
        row = []
        for h2 in horizons:
            if h <= h2:
                row.append(metric_value(h, 'correlation'))
            else:
                row.append(metric_value(h2, 'correlation'))
        corr_data.append(row)

    im = ax6.imshow(corr_data, cmap='YlGnBu', vmin=0, vmax=1)
    ax6.set_xticks(range(len(horizons)))
    ax6.set_yticks(range(len(horizons)))
    ax6.set_xticklabels([f'h={h}' for h in horizons])
    ax6.set_yticklabels([f'h={h}' for h in horizons])
    ax6.set_title('Prediction Correlation Matrix', fontweight='bold')

    for i in range(len(horizons)):
        for j in range(len(horizons)):
            ax6.text(j, i, f'{corr_data[i][j]:.2f}',
                    ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax6, label='Correlation')

    plt.tight_layout()
    plt.savefig(output_path / 'fig23_forecasting_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig23_forecasting_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig23_forecasting_analysis.png/pdf")


def plot_trajectory_examples(probs, ode_params, output_path, n_examples=4):
    """Plot detailed trajectory examples."""
    print("\n" + "=" * 60)
    print("Generating Trajectory Example Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Select diverse starting points
    start_indices = np.linspace(0, len(probs) - 50, n_examples, dtype=int)
    horizon = 30

    for idx, (ax, start_idx) in enumerate(zip(axes, start_indices)):
        # Get actual trajectory
        actual_probs = probs[start_idx:start_idx + horizon, 1]

        # Predict trajectory from initial state
        initial_prob = probs[start_idx, 1]
        initial_state = prob_to_ode_state(initial_prob)
        trajectory = predict_trajectory(initial_state, ode_params, horizon - 1)

        # Convert trajectory to probability
        pred_probs = np.clip(trajectory[:, 2], 0, 1)

        t = np.arange(len(actual_probs))

        ax.plot(t, actual_probs, 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
        ax.plot(t, pred_probs, 'r--', linewidth=2, label='ODE Predicted', marker='s', markersize=4)
        ax.fill_between(t, pred_probs - 0.1, pred_probs + 0.1, alpha=0.2, color='red')

        # Mark initial state
        ax.axhline(y=initial_prob, color='green', linestyle=':', alpha=0.5, label='Initial')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('P(Closed Eyes)')
        ax.set_title(f'Trajectory {idx+1} (start: {start_idx})', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path / 'fig24_trajectory_examples.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig24_trajectory_examples.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig24_trajectory_examples.png/pdf")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("MULTI-STEP TRAJECTORY FORECASTING")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("Device:", DEVICE)
    print("=" * 60)

    start_time = time.time()

    # 1. Load resources
    lstm_model, ode_params, X_test, y_test, recording_ids = load_resources()

    print("\n" + "=" * 60)
    print("Extracting LSTM Probabilities")
    print("=" * 60)
    probs = get_lstm_probabilities(lstm_model, X_test)
    print(f"Probability shape: {probs.shape}")

    step_seconds = _step_seconds()
    horizons = _horizon_steps(step_seconds)
    forecast_results = multistep_forecast(
        probs, y_test, ode_params, horizons, recording_ids, step_seconds
    )
    counts = forecast_results.pop('_counts')
    metrics = evaluate_forecasts(forecast_results, horizons, step_seconds)

    rolling_metrics = rolling_forecast_evaluation(probs, ode_params)
    plot_forecasting_results(forecast_results, metrics, horizons, FIGURES_PATH)
    plot_trajectory_examples(probs, ode_params, FIGURES_PATH)

    rolling_summary = {}
    if len(rolling_metrics):
        rolling_summary = {
            'mean_accuracy': float(rolling_metrics['accuracy'].mean()),
            'std_accuracy': float(rolling_metrics['accuracy'].std()),
            'mean_mae': float(rolling_metrics['mae'].mean()),
            'std_mae': float(rolling_metrics['mae'].std()),
        }

    results = {
        'target': {
            'probability': 'LSTM P(eyes closed) at a future window inside the same recording',
            'label': 'eye-state label of that future window (0 open, 1 closed)',
            'step_seconds': step_seconds,
            'horizons_seconds': list(FORECAST_HORIZONS_S),
        },
        'pair_counts': counts,
        'horizons': horizons,
        'metrics': metrics,
        'rolling_metrics': rolling_summary,
        'ode_params': ode_params,
        'baseline': 'persistence of the current probability and the current label',
    }

    with open(RESULTS_PATH / 'forecasting_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved: forecasting_results.json")
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("FORECASTING ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    for h in horizons:
        block = metrics[str(h)] if str(h) in metrics else metrics[h]
        print(f"\n{block['horizon_seconds']:.1f}s:")
        print(f"  ODE accuracy:         {block['ode']['accuracy_vs_eye_state_label']:.4f}")
        print(f"  Persistence accuracy: {block['persistence']['accuracy_vs_eye_state_label']:.4f}")
    return results


if __name__ == "__main__":
    results = main()
