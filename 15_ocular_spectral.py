"""
15_ocular_spectral.py
=====================
Two checks requested by the revision:

1. Zero the frontopolar channels and, separately, the occipital channels,
   and measure how test accuracy changes. A large drop only at AF/Fp
   sites is consistent with an ocular cue.
2. Compare relative theta and alpha power of windows whose LSTM
   probability is uncertain (0.4 to 0.6) with windows the model calls
   confidently open or closed. This is the spectral table for the
   intermediate-state claim. It does not create a Passive label.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.signal import welch
from sklearn.metrics import accuracy_score

BASE_PATH = Path(__file__).parent
PROCESSED = BASE_PATH / 'outputs' / 'processed_data'
MODELS = BASE_PATH / 'outputs' / 'models'
RESULTS = BASE_PATH / 'outputs' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

FRONTAL = {'FP1', 'FP2', 'FPZ', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8'}
OCCIPITAL = {'O1', 'O2', 'OZ', 'PO3', 'POZ', 'PO4', 'PO7', 'PO8'}


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


def _predict(model, device, X, batch_size=256):
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[start:start + batch_size]).to(device)
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities, axis=0)
    return probabilities, (probabilities[:, 1] >= 0.5).astype(int)


def _indices(channel_names, wanted):
    return [index for index, name in enumerate(channel_names) if name.upper() in wanted]


def _relative_band(window, fs, low, high):
    """Relative Welch power. A 256-sample window has one bin in theta and alpha,
    so a trapezoid over that single bin is zero. Sum the bin densities times df.
    """
    frequencies, power = welch(window, fs=fs, nperseg=min(128, window.shape[-1]), axis=-1)
    if frequencies.size < 2:
        return np.zeros(power.shape[:-1], dtype=float)
    df = float(frequencies[1] - frequencies[0])
    band_mask = (frequencies >= low) & (frequencies < high)
    full_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    band_power = np.sum(power[..., band_mask], axis=-1) * df
    full_power = np.sum(power[..., full_mask], axis=-1) * df
    return band_power / np.maximum(full_power, 1e-12)


def _group_summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {'n': 0, 'mean': None, 'ci95': None}
    if len(values) == 1:
        return {'n': 1, 'mean': float(values[0]), 'ci95': [float(values[0]), float(values[0])]}
    se = stats.sem(values)
    half = 1.96 * se
    mean = float(values.mean())
    return {'n': int(len(values)), 'mean': mean, 'ci95': [mean - half, mean + half]}


def main():
    data = np.load(PROCESSED / 'processed_sequences.npz')
    with open(PROCESSED / 'preprocessing_metadata.json') as handle:
        metadata = json.load(handle)
    channel_names = metadata['channel_names']
    fs = int(metadata['sampling_rate'])
    X_test = data['X_test']
    y_test = data['y_test']

    model, device = _load_lstm()
    full_probs, full_pred = _predict(model, device, X_test)
    report = {
        'full_model_accuracy': float(accuracy_score(y_test, full_pred)),
        'channel_ablation': {},
        'spectral_by_confidence': {},
    }

    for group_name, wanted in (('frontopolar', FRONTAL), ('occipital', OCCIPITAL)):
        indexes = _indices(channel_names, wanted)
        if not indexes:
            report['channel_ablation'][group_name] = {'status': 'channels not in this montage'}
            continue
        ablated = X_test.copy()
        ablated[:, :, indexes] = 0.0
        _, prediction = _predict(model, device, ablated)
        accuracy = float(accuracy_score(y_test, prediction))
        report['channel_ablation'][group_name] = {
            'channels': [channel_names[index] for index in indexes],
            'accuracy': accuracy,
            'accuracy_drop': report['full_model_accuracy'] - accuracy,
        }
        print(f"{group_name:12} drop={report['channel_ablation'][group_name]['accuracy_drop']:.4f}")

    occipital_idx = _indices(channel_names, OCCIPITAL)
    if occipital_idx:
        occipital = X_test[:, :, occipital_idx].mean(axis=2)
        theta = _relative_band(occipital, fs, 4.0, 8.0)
        alpha = _relative_band(occipital, fs, 8.0, 13.0)
        closed = full_probs[:, 1]
        groups = {
            'confident_open': closed < 0.4,
            'uncertain': (closed >= 0.4) & (closed <= 0.6),
            'confident_closed': closed > 0.6,
        }
        for name, mask in groups.items():
            report['spectral_by_confidence'][name] = {
                'theta_relative': _group_summary(theta[mask]),
                'alpha_relative': _group_summary(alpha[mask]),
            }
        uncertain = groups['uncertain']
        for other in ('confident_open', 'confident_closed'):
            for band_name, band in (('theta', theta), ('alpha', alpha)):
                a = band[uncertain]
                b = band[groups[other]]
                if len(a) > 0 and len(b) > 0:
                    statistic, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
                    report['spectral_by_confidence'][f'uncertain_vs_{other}_{band_name}'] = {
                        'u': float(statistic),
                        'p_value': float(p_value),
                    }

    frontal_drop = report['channel_ablation'].get('frontopolar', {}).get('accuracy_drop')
    occipital_drop = report['channel_ablation'].get('occipital', {}).get('accuracy_drop')
    report['reading'] = (
        'If zeroing frontopolar channels removes more accuracy than zeroing '
        'occipital channels, the classifier is using the sites where blinks '
        'and eye movements are strongest. That does not by itself prove an '
        'ocular artifact, and it does not support a higher-order cognitive claim.'
    )
    report['drops'] = {'frontopolar': frontal_drop, 'occipital': occipital_drop}
    with open(RESULTS / 'ocular_spectral.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print('Saved outputs/results/ocular_spectral.json')


if __name__ == '__main__':
    main()
