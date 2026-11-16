"""
13_temporal_baselines.py
========================
Compare the ODE trajectory with ordinary temporal smoothers.

Within each recording, score:
- the raw LSTM label
- a moving average of P(eyes closed)
- a majority vote
- a two-state HMM
- the ODE state after one window step

Accuracy is agreement with the eye-state label. Oscillation rate is
label flips per second. Windows are not pooled across recordings.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from revision_common import initial_state_from_probs


BASE_PATH = Path(__file__).parent
PROCESSED = BASE_PATH / 'outputs' / 'processed_data'
MODELS = BASE_PATH / 'outputs' / 'models'
RESULTS = BASE_PATH / 'outputs' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)


def _load_lstm():
    import importlib.util
    path = BASE_PATH / '04_lstm_model.py'
    spec = importlib.util.spec_from_file_location('lstm_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(MODELS / 'lstm_attention_model.pt', map_location=module.DEVICE, weights_only=False)
    config = checkpoint['model_config']
    model = module.EnhancedLSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        num_heads=config.get('num_heads', 4),
    ).to(module.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, module.DEVICE


def _probabilities(model, device, X, batch_size=256):
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[start:start + batch_size]).to(device)
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


def _moving_average(probabilities, width=5):
    kernel = np.ones(width) / width
    smoothed = np.convolve(probabilities, kernel, mode='same')
    return (smoothed >= 0.5).astype(int)


def _majority(labels, width=5):
    half = width // 2
    output = np.empty_like(labels)
    for index in range(len(labels)):
        window = labels[max(0, index - half):index + half + 1]
        output[index] = int(np.mean(window) >= 0.5)
    return output


def _viterbi(probabilities, stay=0.9):
    closed = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_emit = np.column_stack([np.log(1 - closed), np.log(closed)])
    switch = 1.0 - stay
    log_trans = np.log(np.array([[stay, switch], [switch, stay]]))
    n_steps = len(probabilities)
    scores = np.zeros((n_steps, 2))
    pointer = np.zeros((n_steps, 2), dtype=int)
    scores[0] = log_emit[0]
    for time in range(1, n_steps):
        for state in range(2):
            candidates = scores[time - 1] + log_trans[:, state]
            pointer[time, state] = int(np.argmax(candidates))
            scores[time, state] = candidates[pointer[time, state]] + log_emit[time, state]
    path = np.zeros(n_steps, dtype=int)
    path[-1] = int(np.argmax(scores[-1]))
    for time in range(n_steps - 2, -1, -1):
        path[time] = pointer[time + 1, path[time + 1]]
    return path


def _ode_labels(probabilities, params, step_seconds):
    from scipy.integrate import odeint

    def system(state, _time, rates):
        active, passive, fatigued = state
        d_active = -(rates['k_ap'] + rates['k_af']) * active + rates['k_pa'] * passive + rates['k_fa'] * fatigued
        d_passive = rates['k_ap'] * active - (rates['k_pa'] + rates['k_pf']) * passive + rates['k_fp'] * fatigued
        d_fatigued = rates['k_af'] * active + rates['k_pf'] * passive - (rates['k_fa'] + rates['k_fp']) * fatigued
        return [d_active, d_passive, d_fatigued]

    labels = []
    for open_p, closed_p in probabilities:
        initial = initial_state_from_probs(open_p, closed_p)
        trajectory = odeint(system, initial, [0.0, step_seconds], args=(params,))
        labels.append(int(trajectory[-1, 2] >= 0.5))
    return np.asarray(labels, dtype=int)


def _flip_hz(labels, step_seconds):
    if len(labels) < 2:
        return 0.0
    return float(np.sum(labels[1:] != labels[:-1]) / ((len(labels) - 1) * step_seconds))


def main():
    data = np.load(PROCESSED / 'processed_sequences.npz')
    if 'recording_id_test' not in data.files:
        raise RuntimeError('Re-run 02_preprocessing.py so test windows keep recording ids.')
    with open(PROCESSED / 'preprocessing_metadata.json') as handle:
        step_seconds = float(json.load(handle)['step_seconds'])

    model, device = _load_lstm()
    probabilities = _probabilities(model, device, data['X_test'])
    labels = data['y_test']
    recording_ids = data['recording_id_test']

    ode_params = None
    ode_path = MODELS / 'ode_model.pkl'
    if ode_path.exists():
        with open(ode_path, 'rb') as handle:
            ode_params = {key: float(value) for key, value in pickle.load(handle)['params'].items()}

    pooled = {name: [] for name in ('lstm', 'moving_average', 'majority_vote', 'hmm', 'ode')}
    truth = []
    flips = {name: [] for name in pooled}
    for recording_id in dict.fromkeys(recording_ids.tolist()):
        mask = recording_ids == recording_id
        window_probs = probabilities[mask]
        window_labels = labels[mask]
        closed = window_probs[:, 1]
        raw = (closed >= 0.5).astype(int)
        methods = {
            'lstm': raw,
            'moving_average': _moving_average(closed),
            'majority_vote': _majority(raw),
            'hmm': _viterbi(closed),
        }
        if ode_params is not None:
            methods['ode'] = _ode_labels(window_probs, ode_params, step_seconds)
        truth.append(window_labels)
        for name, prediction in methods.items():
            pooled[name].append(prediction)
            flips[name].append(_flip_hz(prediction, step_seconds))

    truth = np.concatenate(truth)
    summary = {}
    for name, parts in pooled.items():
        if not parts:
            continue
        prediction = np.concatenate(parts)
        summary[name] = {
            'accuracy_vs_eye_state_label': float(accuracy_score(truth, prediction)),
            'mean_label_flips_per_second': float(np.mean(flips[name])),
        }
        print(f"{name:16} accuracy={summary[name]['accuracy_vs_eye_state_label']:.4f}  "
              f"flips/s={summary[name]['mean_label_flips_per_second']:.4f}")

    report = {
        'unit': 'windows that stay inside one recording',
        'step_seconds': step_seconds,
        'methods': summary,
        'note': (
            'A lower flip rate without a gain in label accuracy is smoothing, '
            'not a better classifier. The ODE column is that comparison.'
        ),
    }
    with open(RESULTS / 'temporal_baselines.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print('Saved outputs/results/temporal_baselines.json')


if __name__ == '__main__':
    main()
