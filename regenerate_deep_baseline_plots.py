"""
regenerate_deep_baseline_plots.py
=================================
Regenerate fig26-fig33 from already-saved JSON results.

Workaround for environments where matplotlib's default Qt backend cannot
initialize (xcb plugin missing).  This script forces the Agg backend
*before* importing the plotting modules, so the same plot functions used
in 11_deep_baselines.py / 12_deep_baselines_bciiv2a.py work headlessly.

Run after training has completed (results JSONs must exist in
outputs/results/).
"""

import sys
import json
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

REPO = Path(__file__).parent


def import_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regenerate_ds004148():
    print("=" * 60)
    print("Regenerating ds004148 plots (fig26-fig29)")
    print("=" * 60)
    m = import_module('m11', '11_deep_baselines.py')

    results_dir = m.RESULTS_PATH
    figs_dir = m.FIGURES_PATH
    proc = m.PROCESSED_DATA_PATH / 'processed_sequences.npz'

    needed = ['eegnet_results.json', 'deepconvnet_results.json']
    missing = [n for n in needed if not (results_dir / n).exists()]
    if missing:
        print(f"  Skipping — missing: {missing}")
        return False

    results, histories = {}, {}
    name_map = {'EEGNet': 'eegnet_results.json',
                'DeepConvNet': 'deepconvnet_results.json'}
    for pretty, fname in name_map.items():
        with open(results_dir / fname) as f:
            blob = json.load(f)
        results[pretty] = {k: v for k, v in blob.items() if k != 'history'}
        histories[pretty] = blob['history']

    data = np.load(proc)
    y_test = data['y_test']

    m.plot_training_histories(histories, figs_dir)
    m.plot_confusion_matrices(results, figs_dir)
    m.plot_roc_curves(results, y_test, figs_dir)
    m.plot_full_comparison(results, figs_dir)
    print("  Done.")
    return True


def regenerate_bciiv2a():
    print("=" * 60)
    print("Regenerating BCI-IV-2a plots (fig30-fig33)")
    print("=" * 60)
    m = import_module('m12', '12_deep_baselines_bciiv2a.py')

    results_dir = m.RESULTS_PATH
    figs_dir = m.FIGURES_PATH
    proc = m.PROCESSED_DATA_PATH / 'bciiv2a_sequences.npz'

    needed = ['bciiv2a_eegnet_results.json', 'bciiv2a_deepconvnet_results.json']
    missing = [n for n in needed if not (results_dir / n).exists()]
    if missing:
        print(f"  Skipping — missing: {missing}")
        return False

    results, histories = {}, {}
    name_map = {'EEGNet': 'bciiv2a_eegnet_results.json',
                'DeepConvNet': 'bciiv2a_deepconvnet_results.json'}
    for pretty, fname in name_map.items():
        with open(results_dir / fname) as f:
            blob = json.load(f)
        results[pretty] = {k: v for k, v in blob.items() if k != 'history'}
        histories[pretty] = blob['history']

    data = np.load(proc)
    y_test = data['y_test']

    m.plot_training_histories(histories, figs_dir)
    m.plot_confusion_matrices(results, figs_dir)
    m.plot_roc_curves(results, y_test, figs_dir)
    m.plot_comparison(results, figs_dir)
    print("  Done.")
    return True


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('ds004148', 'both'):
        regenerate_ds004148()
    if which in ('bciiv2a', 'both'):
        regenerate_bciiv2a()
