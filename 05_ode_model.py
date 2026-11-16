"""
05_ode_model.py
===============
Three-State Compartmental ODE Model for Cognitive State Evolution

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Model: Active-Passive-Fatigued (APF) Compartmental Model

States:
- A(t): Active/Focused state proportion (eyes typically open, high engagement)
- P(t): Passive/Relaxed state proportion (eyes may alternate)
- F(t): Fatigued state proportion (eyes typically closed, drowsy)

Constraint: A(t) + P(t) + F(t) = 1 for all t

The ODE system captures:
1. Natural state transitions based on time evolution
2. Recovery dynamics (fatigue -> passive -> active)
3. Fatigue accumulation dynamics
4. Probabilistic coupling points for LSTM integration

References:
- Kermack-McKendrick (1927) SIR compartmental model principles
- Neural ODEs (Chen et al., 2018)
- Vigilance decrement literature (Warm et al., 2008)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Paths
BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / 'outputs'
PROCESSED_DATA_PATH = OUTPUT_PATH / 'processed_data'
MODELS_PATH = OUTPUT_PATH / 'models'
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'

for path in [MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


class CognitiveStateODE:
    """Three-State Compartmental ODE Model for Cognitive Dynamics.

    The APF (Active-Passive-Fatigued) model describes cognitive state
    evolution using a system of ordinary differential equations:

    dA/dt = -k_ap*A - k_af*A + k_pa*P + k_fa*F
    dP/dt = k_ap*A - k_pa*P - k_pf*P + k_fp*F
    dF/dt = k_af*A + k_pf*P - k_fa*F - k_fp*F

    where k_ij represents transition rate from state i to state j.

    Conservation: A + P + F = 1 (reduces system to 2 independent equations)

    This model can be enhanced with LSTM outputs to modulate transition rates
    based on real-time EEG patterns.
    """

    def __init__(self, params=None):
        """Initialize ODE model with transition rate parameters.

        Default parameters based on cognitive fatigue literature
        (approximate time constants in minutes^-1 for typical vigilance tasks).

        Args:
            params: dict with keys ['k_ap', 'k_af', 'k_pa', 'k_pf', 'k_fa', 'k_fp']
        """
        if params is None:
            # Default parameters (interpretable ranges)
            self.params = {
                'k_ap': 0.1,   # Active -> Passive (attention waning)
                'k_af': 0.02,  # Active -> Fatigued (direct fatigue)
                'k_pa': 0.15,  # Passive -> Active (re-engagement)
                'k_pf': 0.08,  # Passive -> Fatigued (fatigue buildup)
                'k_fa': 0.05,  # Fatigued -> Active (recovery)
                'k_fp': 0.1    # Fatigued -> Passive (partial recovery)
            }
        else:
            self.params = params

        self.state_names = ['Active', 'Passive', 'Fatigued']
        self.state_labels = ['A', 'P', 'F']

    def ode_system(self, y, t, params=None):
        """Define the ODE system.

        Args:
            y: State vector [A, P, F]
            t: Time
            params: Optional parameter dict (uses self.params if None)

        Returns:
            dydt: Derivative vector [dA/dt, dP/dt, dF/dt]
        """
        if params is None:
            params = self.params

        A, P, F = y

        # Ensure non-negativity
        A = max(0, A)
        P = max(0, P)
        F = max(0, F)

        # Transition rates
        k_ap = params['k_ap']
        k_af = params['k_af']
        k_pa = params['k_pa']
        k_pf = params['k_pf']
        k_fa = params['k_fa']
        k_fp = params['k_fp']

        # ODE equations
        dA_dt = -k_ap * A - k_af * A + k_pa * P + k_fa * F
        dP_dt = k_ap * A - k_pa * P - k_pf * P + k_fp * F
        dF_dt = k_af * A + k_pf * P - k_fa * F - k_fp * F

        return [dA_dt, dP_dt, dF_dt]

    def solve(self, initial_state, t_span, n_points=100, method='odeint'):
        """Solve the ODE system.

        Args:
            initial_state: [A0, P0, F0] - initial proportions
            t_span: (t_start, t_end) time span
            n_points: Number of time points
            method: 'odeint' or 'solve_ivp'

        Returns:
            t: Time array
            solution: State trajectories [n_points, 3]
        """
        t = np.linspace(t_span[0], t_span[1], n_points)

        # Normalize initial state
        initial_state = np.array(initial_state) / np.sum(initial_state)

        if method == 'odeint':
            solution = odeint(self.ode_system, initial_state, t)
        else:
            sol = solve_ivp(
                lambda t, y: self.ode_system(y, t),
                t_span, initial_state,
                t_eval=t, method='RK45'
            )
            solution = sol.y.T

        # Ensure solution is normalized and non-negative
        solution = np.clip(solution, 0, 1)
        solution = solution / solution.sum(axis=1, keepdims=True)

        return t, solution

    def solve_with_modulation(self, initial_state, t_span, modulation_func,
                              n_points=100):
        """Solve ODE with time-varying rate modulation (for LSTM coupling).

        Args:
            initial_state: [A0, P0, F0]
            t_span: (t_start, t_end)
            modulation_func: Function that takes (t, params) and returns modified params
            n_points: Number of time points

        Returns:
            t: Time array
            solution: State trajectories
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        initial_state = np.array(initial_state) / np.sum(initial_state)

        def modulated_system(y, t):
            mod_params = modulation_func(t, self.params.copy())
            return self.ode_system(y, t, mod_params)

        solution = odeint(modulated_system, initial_state, t)
        solution = np.clip(solution, 0, 1)
        solution = solution / solution.sum(axis=1, keepdims=True)

        return t, solution

    def get_steady_state(self):
        """Compute the steady-state distribution analytically.

        At equilibrium: dA/dt = dP/dt = dF/dt = 0

        Returns:
            dict: Steady state proportions
        """
        k_ap = self.params['k_ap']
        k_af = self.params['k_af']
        k_pa = self.params['k_pa']
        k_pf = self.params['k_pf']
        k_fa = self.params['k_fa']
        k_fp = self.params['k_fp']

        # Solve numerically for steady state
        t, solution = self.solve([0.33, 0.33, 0.34], (0, 1000), 1000)
        steady = solution[-1]

        return {
            'Active': steady[0],
            'Passive': steady[1],
            'Fatigued': steady[2]
        }

    def get_transition_matrix(self):
        """Get the continuous-time transition rate matrix (Q-matrix).

        Returns:
            Q: 3x3 transition rate matrix
        """
        k_ap = self.params['k_ap']
        k_af = self.params['k_af']
        k_pa = self.params['k_pa']
        k_pf = self.params['k_pf']
        k_fa = self.params['k_fa']
        k_fp = self.params['k_fp']

        Q = np.array([
            [-(k_ap + k_af), k_ap, k_af],        # From Active
            [k_pa, -(k_pa + k_pf), k_pf],         # From Passive
            [k_fa, k_fp, -(k_fa + k_fp)]          # From Fatigued
        ])

        return Q

    def fit_to_data(self, observed_proportions, time_points, method='differential_evolution'):
        """Fit ODE parameters to observed state proportions.

        Optimized for 30-subject EEG data with physiologically plausible bounds.

        Args:
            observed_proportions: Array of shape (n_times, 3) - [A, P, F] at each time
            time_points: Array of time points
            method: 'differential_evolution' or 'minimize'

        Returns:
            fitted_params: Optimized parameter dictionary
            loss: Final loss value
        """

        def loss_function(param_array):
            params = {
                'k_ap': param_array[0],
                'k_af': param_array[1],
                'k_pa': param_array[2],
                'k_pf': param_array[3],
                'k_fa': param_array[4],
                'k_fp': param_array[5]
            }

            # Solve ODE
            self.params = params
            t, solution = self.solve(
                observed_proportions[0],
                (time_points[0], time_points[-1]),
                len(time_points)
            )

            # Compute MSE with regularization for stability
            mse = np.mean((solution - observed_proportions) ** 2)

            # Add regularization to prevent extreme rate values
            reg = 0.001 * np.sum(param_array ** 2)

            return mse + reg

        # Physiologically plausible parameter bounds
        # Based on cognitive fatigue literature (time constants ~seconds to minutes)
        bounds = [
            (0.01, 0.5),   # k_ap: Active->Passive (attention waning, moderate)
            (0.001, 0.2),  # k_af: Active->Fatigued (direct fatigue, slower)
            (0.02, 0.5),   # k_pa: Passive->Active (re-engagement, can be fast)
            (0.01, 0.3),   # k_pf: Passive->Fatigued (fatigue buildup, moderate)
            (0.01, 0.3),   # k_fa: Fatigued->Active (recovery, moderate)
            (0.02, 0.4)    # k_fp: Fatigued->Passive (partial recovery, moderate-fast)
        ]

        if method == 'differential_evolution':
            result = differential_evolution(
                loss_function, bounds, seed=42,
                maxiter=1000,  # Increased from 500 for better convergence
                tol=1e-7,
                polish=True,  # Use L-BFGS-B to polish final result
                workers=1  # Use single worker for compatibility
            )
        else:
            x0 = [0.1, 0.02, 0.15, 0.08, 0.05, 0.1]
            result = minimize(loss_function, x0, bounds=bounds, method='L-BFGS-B',
                            options={'maxiter': 1000})

        fitted_params = {
            'k_ap': result.x[0],
            'k_af': result.x[1],
            'k_pa': result.x[2],
            'k_pf': result.x[3],
            'k_fa': result.x[4],
            'k_fp': result.x[5]
        }

        # Validate physiological plausibility
        self._validate_params(fitted_params)

        self.params = fitted_params
        return fitted_params, result.fun

    def _validate_params(self, params, identifiability=None):
        """Print which rates the binary labels can and cannot constrain."""
        print("\nParameter report:")
        identifiable = {}
        if identifiability:
            identifiable = identifiability.get('parameters', {})
        for name, value in params.items():
            flag = identifiable.get(name, {}).get('identifiable')
            status = 'untested' if flag is None else ('constrained' if flag else 'not identified')
            time_constant = (1.0 / value) if value > 0 else float('inf')
            print(f"  {name}={value:.4f}  ({status})")
            if flag:
                print(f"    time constant {time_constant:.2f} s means the observed block stayed in that compartment")
            elif flag is False:
                print("    no time constant is reported; the labels do not constrain this rate")


def map_eye_state_to_cognitive(eye_states, window_size=20):
    """Deprecated label invention. Not used to fit the reported model.

    Sliding a window over binary eyes-open/eyes-closed labels and calling
    the mixed windows Passive manufactures a third state that the
    recordings do not contain. The fitting routine below does not call
    this function.
    """
    n = len(eye_states)
    cognitive_states = np.zeros(n)
    proportions_list = []

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)
        window = eye_states[start:end]

        closed_ratio = np.mean(window)
        variance = np.var(window)

        # Classification logic
        if closed_ratio < 0.3 and variance < 0.15:
            cognitive_states[i] = 0  # Active
        elif closed_ratio > 0.7:
            cognitive_states[i] = 2  # Fatigued
        else:
            cognitive_states[i] = 1  # Passive

    # Compute proportions over time windows
    step = window_size
    for i in range(0, n - step, step):
        window = cognitive_states[i:i+step]
        proportions = [
            np.mean(window == 0),  # Active
            np.mean(window == 1),  # Passive
            np.mean(window == 2)   # Fatigued
        ]
        proportions_list.append(proportions)

    return cognitive_states, np.array(proportions_list)


def load_and_prepare_data(max_blocks_per_class=16, max_duration_s=30.0):
    """Build constant-label blocks from the training recordings only.

    Test recordings are not used. Recordings are not stitched into one
    series: each eyes-open or eyes-closed file is its own block, and the
    label is constant inside it. Passive is not assigned a target.
    """
    print("=" * 60)
    print("Loading training blocks for the observable-margin ODE fit")
    print("=" * 60)

    data_path = PROCESSED_DATA_PATH / 'processed_sequences.npz'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            "Please run 02_preprocessing.py first."
        )

    data = np.load(data_path)
    y_train = data['y_train']
    if 'recording_id_train' not in data.files:
        raise RuntimeError(
            "processed_sequences.npz has no recording_id_train. "
            "Re-run 02_preprocessing.py so each window keeps its recording id."
        )
    recording_ids = data['recording_id_train']
    meta_path = PROCESSED_DATA_PATH / 'preprocessing_metadata.json'
    step_seconds = 0.256
    if meta_path.exists():
        with open(meta_path) as handle:
            step_seconds = float(json.load(handle).get('step_seconds', step_seconds))

    blocks = []
    for recording_id in _stable_unique(recording_ids):
        mask = recording_ids == recording_id
        labels = y_train[mask]
        if len(np.unique(labels)) != 1:
            raise RuntimeError(
                f"{recording_id} contains more than one label. "
                "Windows must not cross a condition boundary."
            )
        duration = min(max_duration_s, max(step_seconds, len(labels) * step_seconds))
        blocks.append({'recording_id': str(recording_id), 'label': int(labels[0]),
                       'duration_s': float(duration), 'n_windows': int(len(labels))})

    rng = np.random.RandomState(42)
    selected = []
    for label in (0, 1):
        pool = [block for block in blocks if block['label'] == label]
        rng.shuffle(pool)
        selected.extend(pool[:max_blocks_per_class])
    if not selected:
        raise RuntimeError("No training blocks available for the ODE fit")

    print(f"Training recordings: {len(blocks)}")
    print(f"Blocks used in the fit: {len(selected)} "
          f"(up to {max_blocks_per_class} per class, duration capped at {max_duration_s:.0f}s)")
    print("Within each block the eye-state label is constant, so Passive is unobserved.")

    # Display series for the figure only. P is 0 because it was not observed.
    display = []
    for block in selected[:8]:
        row = [1.0, 0.0, 0.0] if block['label'] == 0 else [0.0, 0.0, 1.0]
        display.extend([row] * 5)
    return selected, np.asarray(display, dtype=float), {
        'n_training_recordings': len(blocks),
        'n_blocks_fit': len(selected),
        'step_seconds': step_seconds,
        'within_recording_label_changes': 0,
        'conclusion': (
            'Eye-state labels are constant inside each recording. A rate fitted '
            'to these blocks describes the experimental block, not a within-block '
            'fatigue or recovery process. Stitching recordings together would '
            'create transitions only at the file boundaries.'
        ),
    }


def _stable_unique(values):
    seen = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen


def _observable_loss(model, param_array, blocks, regularize=True):
    """Squared error on the observed compartment only.

    Eyes-open blocks observe Active. Eyes-closed blocks observe Fatigued.
    Passive is not in the loss, so a rate that only moves Passive is not
    constrained. The penalty is used while fitting and is left out of the
    identifiability profile.
    """
    params = {
        'k_ap': param_array[0], 'k_af': param_array[1], 'k_pa': param_array[2],
        'k_pf': param_array[3], 'k_fa': param_array[4], 'k_fp': param_array[5],
    }
    model.params = params
    errors = []
    for block in blocks:
        initial = [1.0, 0.0, 0.0] if block['label'] == 0 else [0.0, 0.0, 1.0]
        _, solution = model.solve(initial, (0.0, block['duration_s']), 20)
        observed_index = 0 if block['label'] == 0 else 2
        errors.append(np.mean((solution[:, observed_index] - 1.0) ** 2))
    mse = float(np.mean(errors))
    if not regularize:
        return mse
    return mse + 1e-4 * float(np.sum(np.square(param_array)))


def profile_identifiability(model, blocks, reference_loss, relative_tolerance=0.05):
    """A rate is identified when scaling it by 0.5 or 1.5 changes the loss."""
    names = ['k_ap', 'k_af', 'k_pa', 'k_pf', 'k_fa', 'k_fp']
    base = np.array([model.params[name] for name in names], dtype=float)
    report = {}
    for index, name in enumerate(names):
        changes = []
        absolute = []
        for factor in (0.5, 1.5):
            trial = base.copy()
            trial[index] = np.clip(base[index] * factor, 1e-4, 1.0)
            trial_loss = _observable_loss(model, trial, blocks, regularize=False)
            absolute.append(abs(trial_loss - reference_loss))
            denom = max(reference_loss, 1e-8)
            changes.append(absolute[-1] / denom)
        model.params = {key: float(value) for key, value in zip(names, base)}
        identifiable = bool(max(changes) >= relative_tolerance and max(absolute) >= 1e-4)
        entry = {
            'identifiable': identifiable,
            'max_relative_loss_change': float(max(changes)),
        }
        if identifiable and base[index] > 0:
            entry['time_constant_s'] = float(1.0 / base[index])
            entry['time_constant_meaning'] = (
                'How long the fitted model keeps the observed compartment '
                'occupied during a constant-label block. This is not a '
                'fatigue-onset or recovery time.'
            )
        else:
            entry['time_constant_s'] = None
            entry['time_constant_meaning'] = (
                'Not reported. Changing this rate does not change the fit to '
                'the eyes-open and eyes-closed labels.'
            )
        report[name] = entry
    return {
        'relative_tolerance': relative_tolerance,
        'parameters': report,
        'passive_to_fatigued_identified': report['k_pf']['identifiable'],
        'interpretation': (
            'The observation is binary. Eyes-open constrains Active and '
            'eyes-closed constrains Fatigued. Passive -> Fatigued is reported '
            'only when the profile says it changes the loss. A summed fatigue '
            'accumulation ratio is not computed.'
        ),
    }


def fit_ode_to_data(blocks, seeds=(42, 0, 1, 7, 123)):
    """Fit the six rates to the observable margin, then repeat across seeds."""
    print("\n" + "=" * 60)
    print("Fitting ODE rates to the observable margin")
    print("=" * 60)

    bounds = [
        (0.001, 0.5), (0.001, 0.5), (0.001, 0.5),
        (0.001, 0.5), (0.001, 0.5), (0.001, 0.5),
    ]
    names = ['k_ap', 'k_af', 'k_pa', 'k_pf', 'k_fa', 'k_fp']
    seed_rows = []
    best = None
    for seed in seeds:
        model = CognitiveStateODE()

        def loss_function(param_array, _model=model):
            return _observable_loss(_model, param_array, blocks)

        result = differential_evolution(
            loss_function, bounds, seed=seed, maxiter=40, popsize=6,
            tol=1e-5, polish=True, workers=1, updating='immediate',
        )
        params = {name: float(value) for name, value in zip(names, result.x)}
        seed_rows.append({'seed': int(seed), 'loss': float(result.fun), 'params': params})
        print(f"  seed {seed}: loss={result.fun:.6f}")
        if best is None or result.fun < best[0]:
            best = (float(result.fun), params, model)

    loss, fitted_params, ode_model = best
    ode_model.params = fitted_params
    stacked = {name: [row['params'][name] for row in seed_rows] for name in names}
    stability = {
        name: {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
        for name, values in stacked.items()
    }
    data_loss = _observable_loss(
        ode_model, np.array([fitted_params[name] for name in names]), blocks, regularize=False
    )
    identifiability = profile_identifiability(ode_model, blocks, data_loss)
    ode_model.params = fitted_params
    ode_model._validate_params(fitted_params, identifiability)
    print("\nSeed stability (std across differential-evolution seeds):")
    for name, summary in stability.items():
        print(f"  {name}: mean={summary['mean']:.4f} std={summary['std']:.4f}")
    return ode_model, fitted_params, {
        'loss': loss,
        'seeds': seed_rows,
        'stability': stability,
        'identifiability': identifiability,
    }


def analyze_dynamics(ode_model):
    """Analyze ODE dynamics and stability."""
    print("\n" + "=" * 60)
    print("Analyzing ODE Dynamics")
    print("=" * 60)

    # Steady state
    steady_state = ode_model.get_steady_state()
    print(f"\nSteady State Distribution:")
    for state, value in steady_state.items():
        print(f"  {state}: {value:.4f}")

    # Transition matrix
    Q = ode_model.get_transition_matrix()
    print(f"\nTransition Rate Matrix (Q):")
    print(Q)

    # Eigenvalue analysis for stability
    eigenvalues = np.linalg.eigvals(Q)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"System is stable: {np.all(np.real(eigenvalues) <= 0)}")

    # Dominant time constant
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]
    if len(nonzero_eigs) > 0:
        dominant_timescale = 1 / np.abs(np.min(np.real(nonzero_eigs)))
        print(f"Dominant time constant: {dominant_timescale:.2f} time units")

    return steady_state, Q


def plot_ode_analysis(ode_model, proportions, output_path):
    """Generate comprehensive ODE analysis plots."""
    print("\n" + "=" * 60)
    print("Generating ODE Analysis Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. State trajectories from different initial conditions
    ax = axes[0, 0]
    initial_conditions = [
        [0.8, 0.1, 0.1],   # Starting Active
        [0.1, 0.8, 0.1],   # Starting Passive
        [0.1, 0.1, 0.8]    # Starting Fatigued
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    linestyles = ['-', '--', ':']

    for ic, ls in zip(initial_conditions, linestyles):
        t, sol = ode_model.solve(ic, (0, 50), 200)
        for i, (state, color) in enumerate(zip(['A', 'P', 'F'], colors)):
            label = f'{state} (IC: {ic})' if ic == initial_conditions[0] else None
            ax.plot(t, sol[:, i], color=color, linestyle=ls,
                   alpha=0.8, label=state if ls == '-' else None)

    ax.set_xlabel('Time')
    ax.set_ylabel('State Proportion')
    ax.set_title('State Evolution from Different Initial Conditions', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Phase portrait (A vs P)
    ax = axes[0, 1]
    for ic in initial_conditions:
        t, sol = ode_model.solve(ic, (0, 100), 500)
        ax.plot(sol[:, 0], sol[:, 1], alpha=0.7)
        ax.scatter(sol[0, 0], sol[0, 1], marker='o', s=100, zorder=5)
        ax.scatter(sol[-1, 0], sol[-1, 1], marker='*', s=150, zorder=5)

    ax.set_xlabel('Active (A)')
    ax.set_ylabel('Passive (P)')
    ax.set_title('Phase Portrait: Active vs Passive', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Phase portrait (P vs F)
    ax = axes[0, 2]
    for ic in initial_conditions:
        t, sol = ode_model.solve(ic, (0, 100), 500)
        ax.plot(sol[:, 1], sol[:, 2], alpha=0.7)
        ax.scatter(sol[0, 1], sol[0, 2], marker='o', s=100, zorder=5)
        ax.scatter(sol[-1, 1], sol[-1, 2], marker='*', s=150, zorder=5)

    ax.set_xlabel('Passive (P)')
    ax.set_ylabel('Fatigued (F)')
    ax.set_title('Phase Portrait: Passive vs Fatigued', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Observed vs Fitted trajectories
    ax = axes[1, 0]
    time_obs = np.arange(len(proportions))

    # Fit and predict
    t, sol_fitted = ode_model.solve(proportions[0], (0, len(proportions)-1), len(proportions))

    for i, (state, color) in enumerate(zip(['Active', 'Passive', 'Fatigued'], colors)):
        ax.plot(time_obs, proportions[:, i], 'o', color=color,
               alpha=0.5, markersize=4, label=f'{state} (Observed)')
        ax.plot(t, sol_fitted[:, i], '-', color=color,
               linewidth=2, label=f'{state} (Fitted)')

    ax.set_xlabel('Time Window')
    ax.set_ylabel('State Proportion')
    ax.set_title('Observable margin vs fitted trajectory (Passive has no label)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Transition rate visualization
    ax = axes[1, 1]
    rates = list(ode_model.params.values())
    rate_names = [r'$k_{AP}$', r'$k_{AF}$', r'$k_{PA}$',
                 r'$k_{PF}$', r'$k_{FA}$', r'$k_{FP}$']
    rate_colors = ['#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#3498db']

    bars = ax.bar(rate_names, rates, color=rate_colors, edgecolor='black')
    ax.set_ylabel('Rate Value')
    ax.set_title('Fitted Transition Rates', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{rate:.4f}', ha='center', fontsize=9)

    # 6. Steady state distribution
    ax = axes[1, 2]
    steady = ode_model.get_steady_state()
    states = list(steady.keys())
    values = list(steady.values())

    wedges, texts, autotexts = ax.pie(
        values, labels=states, autopct='%1.1f%%',
        colors=colors, startangle=90,
        explode=(0.05, 0.05, 0.05)
    )
    ax.set_title('Steady State Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / 'fig12_ode_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig12_ode_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig12_ode_analysis.png/pdf")


def plot_state_diagram(ode_model, output_path):
    """Create a state transition diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # State positions (arranged in triangle)
    positions = {
        'Active': (0.5, 0.85),
        'Passive': (0.15, 0.25),
        'Fatigued': (0.85, 0.25)
    }

    colors = {'Active': '#2ecc71', 'Passive': '#3498db', 'Fatigued': '#e74c3c'}

    # Draw states as circles
    for state, pos in positions.items():
        circle = plt.Circle(pos, 0.12, color=colors[state], ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    # Draw arrows for transitions
    arrows = [
        ('Active', 'Passive', 'k_ap'),
        ('Active', 'Fatigued', 'k_af'),
        ('Passive', 'Active', 'k_pa'),
        ('Passive', 'Fatigued', 'k_pf'),
        ('Fatigued', 'Active', 'k_fa'),
        ('Fatigued', 'Passive', 'k_fp')
    ]

    for start, end, rate_name in arrows:
        start_pos = np.array(positions[start])
        end_pos = np.array(positions[end])

        # Calculate arrow direction
        direction = end_pos - start_pos
        direction = direction / np.linalg.norm(direction)

        # Offset for curved arrows
        midpoint = (start_pos + end_pos) / 2
        perpendicular = np.array([-direction[1], direction[0]])
        offset = 0.08 if rate_name in ['k_ap', 'k_pf', 'k_fa'] else -0.08
        control = midpoint + perpendicular * offset

        # Adjust start and end points to be on circle edges
        start_point = start_pos + direction * 0.13
        end_point = end_pos - direction * 0.13

        rate_value = ode_model.params[rate_name]

        ax.annotate('', xy=end_point, xytext=start_point,
                   arrowprops=dict(arrowstyle='->', color='gray',
                                  connectionstyle=f'arc3,rad={offset}',
                                  lw=1.5 + rate_value * 10))

        # Add rate label
        label_pos = midpoint + perpendicular * (offset * 2.5)
        ax.text(label_pos[0], label_pos[1],
               f'{rate_name}={rate_value:.3f}',
               fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Cognitive State Transition Diagram (APF Model)',
                fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path / 'fig13_state_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig13_state_diagram.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig13_state_diagram.png/pdf")


def sensitivity_analysis(ode_model, output_path):
    """Perform sensitivity analysis on ODE parameters."""
    print("\n" + "=" * 60)
    print("Sensitivity Analysis")
    print("=" * 60)

    base_params = ode_model.params.copy()
    param_names = list(base_params.keys())
    perturbation = 0.2  # 20% perturbation

    # Compute sensitivity of steady state to each parameter
    sensitivity_results = []

    for param in param_names:
        sensitivities = []
        for factor in [1 - perturbation, 1 + perturbation]:
            test_params = base_params.copy()
            test_params[param] = base_params[param] * factor
            test_model = CognitiveStateODE(test_params)
            steady = test_model.get_steady_state()
            sensitivities.append([steady['Active'], steady['Passive'], steady['Fatigued']])

        sens_array = np.array(sensitivities)
        delta_steady = (sens_array[1] - sens_array[0]) / (2 * perturbation * base_params[param])

        sensitivity_results.append({
            'parameter': param,
            'sens_Active': delta_steady[0],
            'sens_Passive': delta_steady[1],
            'sens_Fatigued': delta_steady[2]
        })

        print(f"{param}: dA={delta_steady[0]:.4f}, dP={delta_steady[1]:.4f}, dF={delta_steady[2]:.4f}")

    # Plot sensitivity heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sens_df = pd.DataFrame(sensitivity_results)
    sens_matrix = sens_df[['sens_Active', 'sens_Passive', 'sens_Fatigued']].values

    im = ax.imshow(sens_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Active', 'Passive', 'Fatigued'])
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([f'$k_{{{n[2:]}}}$' for n in param_names])

    # Add text annotations
    for i in range(len(param_names)):
        for j in range(3):
            ax.text(j, i, f'{sens_matrix[i, j]:.3f}',
                   ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Parameter Sensitivity Analysis\n(Change in Steady State per Unit Parameter Change)',
                fontweight='bold')
    plt.colorbar(im, ax=ax, label='Sensitivity')

    plt.tight_layout()
    plt.savefig(output_path / 'fig14_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig14_sensitivity_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("\nSaved: fig14_sensitivity_analysis.png/pdf")

    return sensitivity_results


def save_ode_model(ode_model, fitted_params, steady_state, sensitivity_results, fit_report, block_report):
    """Save ODE model and results."""
    print("\n" + "=" * 60)
    print("Saving ODE Model and Results")
    print("=" * 60)

    ode_results = {
        'parameters': fitted_params,
        'steady_state': {key: float(value) for key, value in steady_state.items()},
        'transition_matrix': ode_model.get_transition_matrix().tolist(),
        'sensitivity': sensitivity_results,
        'fit': {
            'loss': fit_report['loss'],
            'stability_across_seeds': fit_report['stability'],
            'identifiability': fit_report['identifiability'],
            'block_design': block_report,
        },
        'fatigue_accumulation_ratio': None,
        'fatigue_accumulation_ratio_note': (
            'Not reported. Adding rates that act on different compartments '
            'is not a proportion of transitions.'
        ),
    }

    with open(RESULTS_PATH / 'ode_results.json', 'w') as f:
        json.dump(ode_results, f, indent=2)

    print("Saved: ode_results.json")

    import pickle
    with open(MODELS_PATH / 'ode_model.pkl', 'wb') as f:
        pickle.dump({
            'params': fitted_params,
            'model_class': 'CognitiveStateODE',
            'initial_condition_rule': 'revision_common.initial_state_from_probs',
            'identifiability': fit_report['identifiability'],
        }, f)

    print("Saved: ode_model.pkl")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("ODE MODEL FOR COGNITIVE STATE EVOLUTION")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("Fit target: observable eyes-open / eyes-closed margin")
    print("=" * 60)

    blocks, proportions, block_report = load_and_prepare_data()
    ode_model, fitted_params, fit_report = fit_ode_to_data(blocks)
    steady_state, Q = analyze_dynamics(ode_model)
    plot_ode_analysis(ode_model, proportions, FIGURES_PATH)
    plot_state_diagram(ode_model, FIGURES_PATH)
    sensitivity_results = sensitivity_analysis(ode_model, FIGURES_PATH)
    save_ode_model(
        ode_model, fitted_params, steady_state, sensitivity_results,
        fit_report, block_report,
    )

    print("\n" + "=" * 60)
    print("ODE MODEL ANALYSIS COMPLETE!")
    print("=" * 60)
    k_pf = fit_report['identifiability']['parameters']['k_pf']
    print(f"Passive->Fatigued identified: {k_pf['identifiable']}")
    return ode_model, fitted_params


if __name__ == "__main__":
    ode_model, params = main()
