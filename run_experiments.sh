#!/usr/bin/env bash
# Revision pipeline for ds004148. Run from the repository root.
# BCI-IV-2a is separate: it is not mapped onto the cognitive ODE.
set -euo pipefail
cd "$(dirname "$0")"

python download_dataset.py --yes
python 02_preprocessing.py
python 02_preprocessing.py --sequence-length 128 --output-tag win128
python 02_preprocessing.py --sequence-length 512 --output-tag win512
python 03_baseline_models.py
python 04_lstm_model.py
python 05_ode_model.py
python 06_lstm_ode_integration.py
python 07_explainability.py
python 08_forecasting.py
python 09_sensitivity_analysis.py
python 11_deep_baselines.py
python 13_temporal_baselines.py
python 15_ocular_spectral.py
python 17_paired_statistics.py
python 18_proposition_checks.py
python 16_revision_figures.py

echo
echo "ds004148 revision run finished."
echo "Motor imagery (Supplementary Note 1): bash download_bci_iv.sh, then"
echo "  python 02b_preprocessing_bciiv2a.py && python 12c_within_subject_mi.py && python 12d_bciiv2b.py"
