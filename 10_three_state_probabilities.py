"""
10_three_state_probabilities.py
===============================
Extract 3-State Cognitive Probabilities for Each Participant

States:
- Eyes Open (Alert/Active)
- Drowsy/Tired (Passive/Transitional)
- Eyes Closed (Fatigued)

This script generates Excel files with probabilities for each state
per participant, suitable for downstream severity analysis.

Authors: Ayesha, Muhammad Khurram Umair
"""

import numpy as np
import pandas as pd
import pickle
import json
import torch
import torch.nn as nn
from pathlib import Path
from scipy.integrate import odeint
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
RESULTS_PATH = OUTPUT_PATH / 'results'
THREE_STATE_PATH = OUTPUT_PATH / 'three_state_results'

THREE_STATE_PATH.mkdir(parents=True, exist_ok=True)


# Model Definitions (same as integration script)
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
        attention_weights = attention_weights.squeeze(-1)
        return context, attention_weights


class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, num_layers=3,
                 num_classes=2, dropout=0.4, bidirectional=True, num_heads=4):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

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
        lstm_output, _ = self.lstm(x)
        lstm_output = self.layer_norm(lstm_output)
        context, attention_weights = self.attention(lstm_output)
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


def load_models():
    """Load trained LSTM and ODE models."""
    print("Loading models...")

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
    print(f"  LSTM model loaded (input_size={config['input_size']})")

    # Load ODE model
    with open(MODELS_PATH / 'ode_model.pkl', 'rb') as f:
        ode_data = pickle.load(f)

    ode_model = CognitiveStateODE(ode_data['params'])
    print(f"  ODE model loaded")

    return lstm_model, ode_model


def load_data():
    """Load all data (train, val, test)."""
    print("Loading data...")
    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_three_state_probabilities(lstm_model, ode_model, X, batch_size=512):
    """
    Get 3-state probabilities for each sample.

    Returns:
        DataFrame with columns:
        - Sample_ID
        - Prob_EyesOpen (Active state)
        - Prob_Drowsy (Passive state)
        - Prob_EyesClosed (Fatigued state)
        - LSTM_P_Open
        - LSTM_P_Closed
        - Predicted_Class
        - Ground_Truth (if available)
    """
    n_samples = len(X)

    # Step 1: Get LSTM probabilities
    print("  Running LSTM inference...")
    lstm_model.eval()
    all_lstm_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="LSTM Inference"):
            batch_end = min(i + batch_size, n_samples)
            X_batch = torch.FloatTensor(X[i:batch_end]).to(DEVICE)
            logits, _ = lstm_model(X_batch, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            all_lstm_probs.append(probs.cpu().numpy())
            del X_batch, logits, probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_lstm_probs = np.concatenate(all_lstm_probs, axis=0)  # (n_samples, 2)

    # Step 2: Get ODE 3-state probabilities
    print("  Running ODE trajectory predictions...")
    three_state_probs = []
    base_params = ode_model.params.copy()
    coupling_strength = 0.5

    for i in tqdm(range(n_samples), desc="ODE Trajectories"):
        p_open = all_lstm_probs[i, 0]
        p_closed = all_lstm_probs[i, 1]

        # Determine initial state based on LSTM output
        if p_closed > 0.6:
            initial_state = [0.2, 0.2, 0.6]  # Mostly fatigued
        elif p_open > 0.6:
            initial_state = [0.6, 0.2, 0.2]  # Mostly active
        else:
            initial_state = [0.33, 0.34, 0.33]  # Mixed/transitional

        # Modulate ODE rates
        params = base_params.copy()
        params['k_af'] = params['k_af'] * (1 + coupling_strength * p_closed)
        params['k_pf'] = params['k_pf'] * (1 + coupling_strength * p_closed)
        params['k_fa'] = params['k_fa'] * (1 + coupling_strength * p_open)
        params['k_pa'] = params['k_pa'] * (1 + coupling_strength * p_open)

        for k in params:
            params[k] = max(0.001, params[k])

        ode_model.params = params

        # Solve ODE to get final state
        t, trajectory = ode_model.solve(initial_state, (0, 20), 20)
        final_state = trajectory[-1]  # [Active, Passive, Fatigued]

        three_state_probs.append(final_state)

    # Reset ODE params
    ode_model.params = base_params

    three_state_probs = np.array(three_state_probs)  # (n_samples, 3)

    # Determine predicted class
    predictions = []
    for i in range(n_samples):
        if three_state_probs[i, 2] > 0.5:  # Fatigued dominant
            predictions.append(2)  # Eyes Closed
        elif three_state_probs[i, 0] > 0.5:  # Active dominant
            predictions.append(0)  # Eyes Open
        else:
            predictions.append(1)  # Drowsy

    return all_lstm_probs, three_state_probs, np.array(predictions)


def create_sample_dataframe(lstm_probs, three_state_probs, predictions, y_true, prefix=""):
    """Create DataFrame with sample-level results."""
    n_samples = len(lstm_probs)

    df = pd.DataFrame({
        'Sample_ID': [f'{prefix}S{i+1:05d}' for i in range(n_samples)],
        'Prob_EyesOpen': three_state_probs[:, 0],
        'Prob_Drowsy': three_state_probs[:, 1],
        'Prob_EyesClosed': three_state_probs[:, 2],
        'LSTM_P_Open': lstm_probs[:, 0],
        'LSTM_P_Closed': lstm_probs[:, 1],
        'Predicted_State': predictions,
        'Ground_Truth': y_true
    })

    # Map numeric to labels
    state_map = {0: 'Eyes Open', 1: 'Drowsy', 2: 'Eyes Closed'}
    gt_map = {0: 'Eyes Open', 1: 'Eyes Closed'}

    df['Predicted_State_Label'] = df['Predicted_State'].map(state_map)
    df['Ground_Truth_Label'] = df['Ground_Truth'].map(gt_map)

    return df


def create_participant_dataframe(sample_df, n_participants=30):
    """
    Aggregate sample-level results to participant-level.

    Distributes samples evenly across participants and computes
    mean probabilities per participant.
    """
    n_samples = len(sample_df)
    samples_per_participant = n_samples // n_participants

    participant_data = []

    for p in range(n_participants):
        start_idx = p * samples_per_participant
        end_idx = start_idx + samples_per_participant if p < n_participants - 1 else n_samples

        subset = sample_df.iloc[start_idx:end_idx]

        participant_data.append({
            'Participant_ID': f'P{p+1:03d}',
            'N_Samples': len(subset),
            'Prob_EyesOpen': subset['Prob_EyesOpen'].mean(),
            'Prob_Drowsy': subset['Prob_Drowsy'].mean(),
            'Prob_EyesClosed': subset['Prob_EyesClosed'].mean(),
            'Prob_EyesOpen_Std': subset['Prob_EyesOpen'].std(),
            'Prob_Drowsy_Std': subset['Prob_Drowsy'].std(),
            'Prob_EyesClosed_Std': subset['Prob_EyesClosed'].std(),
            'Mean_LSTM_P_Open': subset['LSTM_P_Open'].mean(),
            'Mean_LSTM_P_Closed': subset['LSTM_P_Closed'].mean(),
            'Pct_EyesOpen': (subset['Predicted_State'] == 0).mean() * 100,
            'Pct_Drowsy': (subset['Predicted_State'] == 1).mean() * 100,
            'Pct_EyesClosed': (subset['Predicted_State'] == 2).mean() * 100,
        })

    return pd.DataFrame(participant_data)


def main():
    print("=" * 60)
    print("THREE-STATE COGNITIVE PROBABILITY EXTRACTION")
    print("States: Eyes Open | Drowsy | Eyes Closed")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Output: {THREE_STATE_PATH}")
    print("=" * 60)

    # Load models
    lstm_model, ode_model = load_models()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Process each split
    results = {}

    for split_name, X, y in [('train', X_train, y_train),
                              ('val', X_val, y_val),
                              ('test', X_test, y_test)]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} set ({len(X)} samples)")
        print("="*60)

        lstm_probs, three_state_probs, predictions = get_three_state_probabilities(
            lstm_model, ode_model, X
        )

        # Create sample-level DataFrame
        sample_df = create_sample_dataframe(
            lstm_probs, three_state_probs, predictions, y, prefix=f"{split_name}_"
        )

        # Save sample-level results
        sample_file = THREE_STATE_PATH / f'{split_name}_sample_probabilities.xlsx'
        sample_df.to_excel(sample_file, index=False)
        print(f"  Saved: {sample_file.name}")

        # Also save as CSV
        sample_df.to_csv(THREE_STATE_PATH / f'{split_name}_sample_probabilities.csv', index=False)

        results[split_name] = {
            'sample_df': sample_df,
            'n_samples': len(X)
        }

    # Create participant-level aggregation for test set
    print(f"\n{'='*60}")
    print("Creating Participant-Level Aggregation (Test Set)")
    print("="*60)

    # Approximate number of test participants (15% of 30 = ~5)
    n_test_participants = 5
    participant_df = create_participant_dataframe(
        results['test']['sample_df'],
        n_participants=n_test_participants
    )

    # Save participant-level results
    participant_file = THREE_STATE_PATH / 'participant_probabilities.xlsx'
    participant_df.to_excel(participant_file, index=False)
    print(f"  Saved: {participant_file.name}")

    participant_df.to_csv(THREE_STATE_PATH / 'participant_probabilities.csv', index=False)

    # Create summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print("="*60)

    summary = {
        'Dataset': 'OpenNeuro ds004148',
        'Total_Participants': 30,
        'Test_Participants': n_test_participants,
        'Train_Samples': results['train']['n_samples'],
        'Val_Samples': results['val']['n_samples'],
        'Test_Samples': results['test']['n_samples'],
        'States': ['Eyes Open (Active)', 'Drowsy (Passive)', 'Eyes Closed (Fatigued)']
    }

    # Test set statistics
    test_df = results['test']['sample_df']
    summary['Test_Mean_Prob_EyesOpen'] = test_df['Prob_EyesOpen'].mean()
    summary['Test_Mean_Prob_Drowsy'] = test_df['Prob_Drowsy'].mean()
    summary['Test_Mean_Prob_EyesClosed'] = test_df['Prob_EyesClosed'].mean()

    with open(THREE_STATE_PATH / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: summary.json")

    # Print participant table
    print(f"\n{'='*60}")
    print("Participant Probabilities (Test Set)")
    print("="*60)
    print(participant_df[['Participant_ID', 'N_Samples',
                          'Prob_EyesOpen', 'Prob_Drowsy', 'Prob_EyesClosed']].to_string(index=False))

    print(f"\n{'='*60}")
    print("THREE-STATE EXTRACTION COMPLETE!")
    print(f"Results saved to: {THREE_STATE_PATH}")
    print("="*60)

    # List output files
    print("\nOutput Files:")
    for f in sorted(THREE_STATE_PATH.glob('*')):
        print(f"  - {f.name}")

    return results, participant_df


if __name__ == "__main__":
    results, participants = main()
