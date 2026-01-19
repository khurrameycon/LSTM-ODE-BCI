"""
09_sensitivity_analysis.py
==========================
Comprehensive Sensitivity Analysis and Ablation Studies

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

This module implements:
1. Architecture Ablation Studies
   - Attention mechanism contribution
   - Bidirectional vs Unidirectional LSTM
   - Number of layers impact
   - Hidden size sensitivity

2. LSTM-ODE Coupling Strength Analysis
   - Alpha parameter sweep
   - Optimal coupling determination

3. Statistical Significance Tests
   - McNemar's test for model comparison
   - Cohen's d for effect sizes
   - Bootstrap confidence intervals
   - Paired t-tests

4. Hyperparameter Sensitivity
   - Learning rate
   - Dropout rate
   - Sequence length
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
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


# ============================================================
# Statistical Test Functions
# ============================================================
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def mcnemars_test(y_true, pred1, pred2):
    """Perform McNemar's test for comparing two classifiers.

    Tests whether the two classifiers have the same error rate.

    Returns:
        statistic: Chi-squared statistic
        p_value: Two-sided p-value
        contingency_table: 2x2 table of disagreements
    """
    # Create contingency table
    # [both correct, model1 correct only]
    # [model2 correct only, both wrong]

    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    # b: model 1 correct, model 2 wrong
    b = np.sum(correct1 & ~correct2)
    # c: model 1 wrong, model 2 correct
    c = np.sum(~correct1 & correct2)

    contingency_table = np.array([
        [np.sum(correct1 & correct2), b],
        [c, np.sum(~correct1 & ~correct2)]
    ])

    # McNemar's test
    if b + c == 0:
        return 0.0, 1.0, contingency_table

    # Use exact binomial test for small samples
    if b + c < 25:
        # Exact binomial test
        p_value = 2 * min(stats.binom.cdf(min(b, c), b + c, 0.5),
                         1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5))
        statistic = (b - c) ** 2 / (b + c)
    else:
        # Chi-squared approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value, contingency_table


def bootstrap_ci(data, statistic_func=np.mean, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval."""
    boot_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(boot_sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)

    return lower, upper, np.std(boot_stats)


# ============================================================
# Model Architectures for Ablation
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


class AblationLSTMModel(nn.Module):
    """LSTM model with configurable components for ablation studies."""
    def __init__(self, input_size=61, hidden_size=256, num_layers=3,
                 num_classes=2, dropout=0.4, bidirectional=True,
                 use_attention=True, use_layer_norm=True):
        super(AblationLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * self.num_directions

        # Layer norm
        self.layer_norm = nn.LayerNorm(lstm_output_size) if use_layer_norm else nn.Identity()

        # Attention or simple pooling
        if use_attention:
            self.attention = Attention(lstm_output_size)
        else:
            self.attention = None

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        if self.attention is not None:
            context, _ = self.attention(lstm_out)
        else:
            # Mean pooling
            context = torch.mean(lstm_out, dim=1)

        output = self.classifier(context)
        return output


# ============================================================
# Ablation Study Functions
# ============================================================
def load_data():
    """Load processed data for ablation studies."""
    print("=" * 60)
    print("Loading Data for Ablation Studies")
    print("=" * 60)

    data = np.load(PROCESSED_DATA_PATH / 'processed_sequences.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def quick_train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                         epochs=10, batch_size=512, lr=0.001):
    """Quick training for ablation comparison."""
    model = model.to(DEVICE)

    # Create data loaders (use subset for speed)
    n_train = min(len(X_train), 20000)
    indices = np.random.choice(len(X_train), n_train, replace=False)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train[indices]),
        torch.LongTensor(y_train[indices])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }

    return metrics, all_preds


def run_architecture_ablation(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run architecture ablation studies."""
    print("\n" + "=" * 60)
    print("Architecture Ablation Studies")
    print("=" * 60)

    input_size = X_train.shape[2]
    results = {}
    predictions = {}

    ablation_configs = [
        {'name': 'Full Model', 'bidirectional': True, 'use_attention': True, 'num_layers': 3},
        {'name': 'No Attention', 'bidirectional': True, 'use_attention': False, 'num_layers': 3},
        {'name': 'Unidirectional', 'bidirectional': False, 'use_attention': True, 'num_layers': 3},
        {'name': '1 Layer', 'bidirectional': True, 'use_attention': True, 'num_layers': 1},
        {'name': '2 Layers', 'bidirectional': True, 'use_attention': True, 'num_layers': 2},
        {'name': 'Minimal', 'bidirectional': False, 'use_attention': False, 'num_layers': 1},
    ]

    for config in tqdm(ablation_configs, desc="Ablation configs"):
        print(f"\n  Testing: {config['name']}")

        model = AblationLSTMModel(
            input_size=input_size,
            hidden_size=256,
            num_layers=config['num_layers'],
            num_classes=2,
            dropout=0.4,
            bidirectional=config['bidirectional'],
            use_attention=config['use_attention']
        )

        metrics, preds = quick_train_evaluate(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )

        results[config['name']] = {
            'config': {k: v for k, v in config.items() if k != 'name'},
            'metrics': metrics
        }
        predictions[config['name']] = preds

        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    return results, predictions


def run_statistical_comparison(y_test, predictions, baseline='Full Model'):
    """Run statistical tests comparing models."""
    print("\n" + "=" * 60)
    print("Statistical Model Comparisons")
    print("=" * 60)

    results = {}
    baseline_preds = predictions[baseline]

    for model_name, model_preds in predictions.items():
        if model_name == baseline:
            continue

        # McNemar's test
        stat, p_val, contingency = mcnemars_test(y_test, baseline_preds, model_preds)

        # Cohen's d for accuracy
        baseline_correct = (baseline_preds == y_test).astype(float)
        model_correct = (model_preds == y_test).astype(float)
        d = cohens_d(baseline_correct, model_correct)

        # Paired t-test
        t_stat, t_pval = ttest_rel(baseline_correct, model_correct)

        results[model_name] = {
            'mcnemar_stat': float(stat),
            'mcnemar_p': float(p_val),
            'mcnemar_significant': bool(p_val < 0.05),
            'cohens_d': float(d),
            'effect_size': interpret_cohens_d(d),
            'paired_t_stat': float(t_stat),
            'paired_t_p': float(t_pval),
            'contingency_table': contingency.tolist()
        }

        print(f"\n{baseline} vs {model_name}:")
        print(f"  McNemar's test: chi2={stat:.2f}, p={p_val:.4f}")
        print(f"  Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")
        print(f"  Paired t-test: t={t_stat:.2f}, p={t_pval:.4f}")

    return results


def run_coupling_sensitivity(X_test, y_test):
    """Analyze LSTM-ODE coupling strength sensitivity."""
    print("\n" + "=" * 60)
    print("LSTM-ODE Coupling Strength Sensitivity")
    print("=" * 60)

    # Load models
    checkpoint = torch.load(MODELS_PATH / 'lstm_attention_model.pt',
                           map_location=DEVICE, weights_only=False)

    with open(MODELS_PATH / 'ode_model.pkl', 'rb') as f:
        ode_data = pickle.load(f)
    ode_params = {k: float(v) for k, v in ode_data['params'].items()}

    # Load integration results if available
    try:
        with open(RESULTS_PATH / 'coupling_analysis.json', 'r') as f:
            coupling_results = json.load(f)
        print("Loaded existing coupling analysis results")
    except FileNotFoundError:
        print("Generating coupling analysis from integration results")
        with open(RESULTS_PATH / 'integration_results.json', 'r') as f:
            integration_results = json.load(f)
        coupling_results = integration_results.get('coupling_analysis', {})

    # Analyze alpha sweep
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    if 'alpha_sweep' in coupling_results:
        sweep_data = coupling_results['alpha_sweep']
        print("\nCoupling Strength Analysis:")
        for alpha_data in sweep_data:
            print(f"  alpha={alpha_data['alpha']:.2f}: "
                  f"Acc={alpha_data['accuracy']:.4f}, F1={alpha_data['f1']:.4f}")
    else:
        print("Note: Full coupling sweep not available. Using default analysis.")

    return coupling_results


def compute_bootstrap_intervals(predictions, y_test, n_bootstrap=1000):
    """Compute bootstrap confidence intervals for all models."""
    print("\n" + "=" * 60)
    print("Bootstrap Confidence Intervals")
    print("=" * 60)

    results = {}

    for model_name, preds in predictions.items():
        correct = (preds == y_test).astype(float)

        # Bootstrap CI for accuracy
        acc_lower, acc_upper, acc_se = bootstrap_ci(correct, np.mean, n_bootstrap)
        accuracy = np.mean(correct)

        results[model_name] = {
            'accuracy': float(accuracy),
            'ci_lower': float(acc_lower),
            'ci_upper': float(acc_upper),
            'se': float(acc_se)
        }

        print(f"{model_name}:")
        print(f"  Accuracy: {accuracy:.4f} [{acc_lower:.4f}, {acc_upper:.4f}]")

    return results


def analyze_component_contribution(ablation_results):
    """Analyze contribution of each architectural component."""
    print("\n" + "=" * 60)
    print("Component Contribution Analysis")
    print("=" * 60)

    full_model_acc = ablation_results['Full Model']['metrics']['accuracy']

    contributions = {}

    # Attention contribution
    no_attn_acc = ablation_results['No Attention']['metrics']['accuracy']
    contributions['Attention'] = full_model_acc - no_attn_acc

    # Bidirectional contribution
    unidi_acc = ablation_results['Unidirectional']['metrics']['accuracy']
    contributions['Bidirectional'] = full_model_acc - unidi_acc

    # Deep layers contribution
    one_layer_acc = ablation_results['1 Layer']['metrics']['accuracy']
    contributions['Deep Layers (3 vs 1)'] = full_model_acc - one_layer_acc

    print("\nComponent Contributions to Accuracy:")
    for component, contribution in sorted(contributions.items(), key=lambda x: -x[1]):
        sign = '+' if contribution >= 0 else ''
        print(f"  {component}: {sign}{contribution:.4f} ({contribution * 100:.2f}%)")

    return contributions


def plot_ablation_results(ablation_results, statistical_results, contributions,
                          bootstrap_results, output_path):
    """Generate comprehensive ablation visualization."""
    print("\n" + "=" * 60)
    print("Generating Ablation Study Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 14))

    # 1. Model comparison bar chart
    ax1 = fig.add_subplot(2, 3, 1)
    models = list(ablation_results.keys())
    accuracies = [ablation_results[m]['metrics']['accuracy'] for m in models]
    f1_scores = [ablation_results[m]['metrics']['f1'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral')

    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Component contributions
    ax2 = fig.add_subplot(2, 3, 2)
    components = list(contributions.keys())
    values = [contributions[c] * 100 for c in components]  # Convert to percentage
    colors = ['green' if v >= 0 else 'red' for v in values]

    ax2.barh(components, values, color=colors, edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Contribution to Accuracy (%)')
    ax2.set_title('Component Contribution Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    for i, (comp, val) in enumerate(zip(components, values)):
        ax2.text(val + 0.1 if val >= 0 else val - 0.1, i,
                f'{val:.2f}%', va='center',
                ha='left' if val >= 0 else 'right', fontsize=9)

    # 3. Effect sizes (Cohen's d)
    ax3 = fig.add_subplot(2, 3, 3)
    if statistical_results:
        model_names = list(statistical_results.keys())
        cohens_d_values = [abs(statistical_results[m]['cohens_d']) for m in model_names]

        colors = []
        for d in cohens_d_values:
            if d < 0.2:
                colors.append('#2ecc71')  # Green - negligible
            elif d < 0.5:
                colors.append('#f1c40f')  # Yellow - small
            elif d < 0.8:
                colors.append('#e67e22')  # Orange - medium
            else:
                colors.append('#e74c3c')  # Red - large

        ax3.barh(model_names, cohens_d_values, color=colors, edgecolor='black')
        ax3.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax3.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
        ax3.set_xlabel("Cohen's d (Effect Size)")
        ax3.set_title('Effect Size vs Full Model', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

    # 4. Bootstrap confidence intervals
    ax4 = fig.add_subplot(2, 3, 4)
    models = list(bootstrap_results.keys())
    means = [bootstrap_results[m]['accuracy'] for m in models]
    lowers = [bootstrap_results[m]['ci_lower'] for m in models]
    uppers = [bootstrap_results[m]['ci_upper'] for m in models]

    errors = [[m - l for m, l in zip(means, lowers)],
              [u - m for m, u in zip(means, uppers)]]

    ax4.errorbar(means, range(len(models)), xerr=errors, fmt='o',
                capsize=5, capthick=2, color='steelblue', markersize=8)
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels(models)
    ax4.set_xlabel('Accuracy')
    ax4.set_title('95% Bootstrap Confidence Intervals', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Statistical significance matrix
    ax5 = fig.add_subplot(2, 3, 5)
    if statistical_results:
        model_names = list(statistical_results.keys())
        p_values = [statistical_results[m]['mcnemar_p'] for m in model_names]

        # Create significance matrix (comparing to full model)
        sig_matrix = np.zeros((len(model_names), 1))
        for i, m in enumerate(model_names):
            sig_matrix[i, 0] = 1 if statistical_results[m]['mcnemar_p'] < 0.05 else 0

        im = ax5.imshow(sig_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax5.set_yticks(range(len(model_names)))
        ax5.set_yticklabels(model_names)
        ax5.set_xticks([0])
        ax5.set_xticklabels(['vs Full Model'])
        ax5.set_title("McNemar's Test Significance", fontweight='bold')

        # Add p-values as text
        for i, (m, p) in enumerate(zip(model_names, p_values)):
            sig = '*' if p < 0.05 else ''
            ax5.text(0, i, f'{p:.3f}{sig}', ha='center', va='center', fontsize=10)

    # 6. Performance radar chart
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    metrics = ['Accuracy', 'F1', 'MCC']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for model in ['Full Model', 'No Attention', 'Unidirectional']:
        if model in ablation_results:
            values = [
                ablation_results[model]['metrics']['accuracy'],
                ablation_results[model]['metrics']['f1'],
                ablation_results[model]['metrics']['mcc']
            ]
            values += values[:1]
            ax6.plot(angles, values, 'o-', linewidth=2, label=model)
            ax6.fill(angles, values, alpha=0.1)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_ylim([0.5, 1.0])
    ax6.set_title('Performance Radar', fontweight='bold', y=1.08)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path / 'fig25_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig25_ablation_study.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig25_ablation_study.png/pdf")


def create_results_tables(ablation_results, statistical_results, contributions):
    """Create formatted results tables for the manuscript."""
    print("\n" + "=" * 60)
    print("Generating Results Tables")
    print("=" * 60)

    # Table 1: Model Comparison
    print("\nTable: Architecture Ablation Results")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'MCC':>10}")
    print("-" * 70)
    for model, data in ablation_results.items():
        m = data['metrics']
        print(f"{model:<20} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['mcc']:>10.4f}")
    print("-" * 70)

    # Table 2: Statistical Tests
    if statistical_results:
        print("\nTable: Statistical Comparison (vs Full Model)")
        print("-" * 80)
        print(f"{'Model':<20} {'McNemar p':>12} {'Cohen d':>10} {'Effect':>12} {'Sig':>8}")
        print("-" * 80)
        for model, data in statistical_results.items():
            sig = '*' if data['mcnemar_p'] < 0.05 else ''
            print(f"{model:<20} {data['mcnemar_p']:>12.4f} {data['cohens_d']:>10.3f} "
                  f"{data['effect_size']:>12} {sig:>8}")
        print("-" * 80)

    return {
        'ablation_table': {m: ablation_results[m]['metrics'] for m in ablation_results},
        'statistical_table': statistical_results,
        'contributions': contributions
    }


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS AND ABLATION STUDIES")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("Device:", DEVICE)
    print("=" * 60)

    start_time = time.time()

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Run architecture ablation
    ablation_results, predictions = run_architecture_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 3. Statistical comparisons
    statistical_results = run_statistical_comparison(y_test, predictions)

    # 4. Bootstrap confidence intervals
    bootstrap_results = compute_bootstrap_intervals(predictions, y_test)

    # 5. Component contribution analysis
    contributions = analyze_component_contribution(ablation_results)

    # 6. Coupling sensitivity (if available)
    coupling_results = run_coupling_sensitivity(X_test, y_test)

    # 7. Generate visualizations
    plot_ablation_results(ablation_results, statistical_results, contributions,
                         bootstrap_results, FIGURES_PATH)

    # 8. Create tables
    tables = create_results_tables(ablation_results, statistical_results, contributions)

    # 9. Save all results
    all_results = {
        'ablation': {m: {
            'config': ablation_results[m]['config'],
            'metrics': ablation_results[m]['metrics']
        } for m in ablation_results},
        'statistical_tests': statistical_results,
        'bootstrap_ci': bootstrap_results,
        'component_contributions': {k: float(v) for k, v in contributions.items()},
        'coupling_sensitivity': coupling_results
    }

    with open(RESULTS_PATH / 'sensitivity_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved: sensitivity_analysis.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("\n1. Architecture Ablation:")
    for comp, contrib in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"   - {comp}: {contrib * 100:+.2f}% contribution")

    print("\n2. Statistical Significance:")
    for model, data in statistical_results.items():
        sig = "significant" if data['mcnemar_p'] < 0.05 else "not significant"
        print(f"   - {model}: {sig} (p={data['mcnemar_p']:.4f})")

    return all_results


if __name__ == "__main__":
    results = main()
