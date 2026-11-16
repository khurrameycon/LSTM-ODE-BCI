"""Shared definitions for the revision experiments.

One initial-condition rule, subject-level statistics, and the design
power calculation. Window-level resampling is not used: consecutive
windows overlap by 50 percent, so they are not independent trials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats


def initial_state_from_probs(p_open, p_closed):
    """Map class probabilities to (Active, Passive, Fatigued).

    This is the only initial-condition rule in the revision. Passive
    occupancy is the uncertainty heuristic

        P_passive = 0.5 * min(P_open, P_closed)

    and the remaining probability is split between Active and Fatigued
    in proportion to the two class probabilities. The training-set mean,
    the piecewise threshold, and the 0.6 threshold are not used.
    """
    p_open = np.asarray(p_open, dtype=float)
    p_closed = np.asarray(p_closed, dtype=float)
    scalar = p_open.ndim == 0
    p_open = np.atleast_1d(p_open)
    p_closed = np.atleast_1d(p_closed)
    total = p_open + p_closed
    total = np.where(total <= 0, 1.0, total)
    p_open = p_open / total
    p_closed = p_closed / total
    p_passive = 0.5 * np.minimum(p_open, p_closed)
    remainder = 1.0 - p_passive
    state = np.stack(
        [remainder * p_open, p_passive, remainder * p_closed],
        axis=-1,
    )
    state = np.clip(state, 0.0, None)
    state = state / np.maximum(state.sum(axis=-1, keepdims=True), 1e-12)
    if scalar:
        return state[0]
    return state


def subject_accuracy(correct, subject_ids):
    """Mean accuracy of each subject. `correct` is 1 where the prediction matches."""
    correct = np.asarray(correct, dtype=float)
    subject_ids = np.asarray(subject_ids)
    subjects = []
    accuracies = []
    for subject in _unique_stable(subject_ids):
        mask = subject_ids == subject
        subjects.append(subject)
        accuracies.append(float(correct[mask].mean()))
    return np.asarray(subjects), np.asarray(accuracies, dtype=float)


def subject_bootstrap_ci(correct, subject_ids, n_bootstrap=1000, ci=0.95, seed=42):
    """CI for mean subject-level accuracy.

    Each draw resamples subjects with replacement and averages their
    accuracies. Windows inside a subject are not resampled.
    """
    _, accuracies = subject_accuracy(correct, subject_ids)
    if len(accuracies) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(accuracies, size=(n_bootstrap, len(accuracies)), replace=True)
    means = draws.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    return lower, upper, float(means.std(ddof=1))


def paired_cohens_dz(correct_a, correct_b, subject_ids):
    """Cohen's d_z for a paired classifier comparison.

    The paired observation is the difference of the two models'
    accuracies on the same subject. d_z = mean(difference) / sd(difference).
    It is not computed on overlapping windows.
    """
    correct_a = np.asarray(correct_a, dtype=float)
    correct_b = np.asarray(correct_b, dtype=float)
    subject_ids = np.asarray(subject_ids)
    diffs = []
    for subject in _unique_stable(subject_ids):
        mask = subject_ids == subject
        diffs.append(correct_a[mask].mean() - correct_b[mask].mean())
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return 0.0
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(diffs.mean() / sd)


def subject_sign_test(correct_a, correct_b, subject_ids):
    """Two-sided sign test on which model has higher subject-level accuracy.

    Ties are dropped. This replaces a window-level McNemar test, which
    treats overlapping windows as independent.
    """
    correct_a = np.asarray(correct_a, dtype=float)
    correct_b = np.asarray(correct_b, dtype=float)
    subject_ids = np.asarray(subject_ids)
    wins_a = wins_b = ties = 0
    for subject in _unique_stable(subject_ids):
        mask = subject_ids == subject
        diff = correct_a[mask].mean() - correct_b[mask].mean()
        if diff > 1e-12:
            wins_a += 1
        elif diff < -1e-12:
            wins_b += 1
        else:
            ties += 1
    n_decisive = wins_a + wins_b
    if n_decisive == 0:
        p_value = 1.0
    else:
        p_value = float(
            stats.binomtest(wins_a, n_decisive, 0.5, alternative='two-sided').pvalue
        )
    return {
        'p_value': p_value,
        'subjects_a_better': int(wins_a),
        'subjects_b_better': int(wins_b),
        'ties': int(ties),
        'unit': 'subject',
    }


def design_power(n_test_subjects, subject_sd=0.10, alpha=0.05, power=0.80,
                 target_delta=0.05):
    """Sample-size statement for a paired subject-level accuracy comparison."""
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    n_test_subjects = int(n_test_subjects)
    if n_test_subjects < 2:
        mdd = None
    else:
        mdd = (z_alpha + z_power) * subject_sd / np.sqrt(n_test_subjects)
    n_required = int(np.ceil(((z_alpha + z_power) * subject_sd / target_delta) ** 2))
    return {
        'unit': 'subject-level accuracy, paired two-sided z test',
        'alpha': alpha,
        'target_power': power,
        'assumed_subject_sd': subject_sd,
        'n_test_subjects': n_test_subjects,
        'minimum_detectable_accuracy_difference': None if mdd is None else float(mdd),
        'n_test_subjects_required_for_delta': n_required,
        'target_delta': target_delta,
        'note': (
            'The sample size is the number of test subjects. Overlapping '
            'windows are not independent observations. assumed_subject_sd is '
            'a design value; replace it with the empirical SD of '
            'subject-level accuracy after evaluation. A detectable difference '
            'larger than the gains being claimed means the test set cannot '
            'support that claim.'
        ),
    }


def claim_or_wait(lock_path, ready, poll_seconds=20):
    """Claim a lock, or wait while another live process holds it.

    Returns True when this process should do the work. Returns False when
    `ready()` is already true, so a later pipeline step does not train the
    same model again.
    """
    import os
    import time

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if ready():
            return False
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _lock_pid(lock_path)
            if pid is not None and _pid_alive(pid):
                print(f"Waiting for pid {pid} ({lock_path.name})", flush=True)
                time.sleep(poll_seconds)
                continue
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True


def release_lock(lock_path):
    """Remove the lock when this process still owns it."""
    import os

    lock_path = Path(lock_path)
    try:
        if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
            lock_path.unlink()
    except OSError:
        pass


def _lock_pid(lock_path):
    try:
        return int(Path(lock_path).read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _unique_stable(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
