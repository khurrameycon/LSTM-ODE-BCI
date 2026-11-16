# LSTM-ODE-BCI: Deep Learning and Dynamical Modeling for EEG-Based Cognitive State Evolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and saved results for a framework that couples a bidirectional LSTM-Attention classifier with a three-state
(Active, Passive, Fatigued) compartmental ordinary differential equation (ODE) model, together with the subject-level
evaluation used to test what each component contributes.

## Paper

> **Deep Learning and Dynamical Modeling Framework for EEG-Based Cognitive State Evolution in Brain-Computer Interfaces**
>
> Muhammad Khurram Umair, Ayesha Arif, Muhammad Faisal Abrar, Sikandar Ali, It Ee Lee, Salman Jan
>
> Manuscript under review at *Scientific Reports*.

Every number in the manuscript's tables and figures is read from the files in `outputs/results/`, which are
included in this repository (see [Reproducing the paper](#reproducing-the-paper)).

## Main findings

- On eight held-out test subjects, LSTM-Attention and LSTM-ODE reach an accuracy of 0.888 (subject-level 95% CI
  0.835–0.950), within the range of three feature-based and four deep-learning baselines (0.885–0.922). No
  subject-level comparison is significant; with eight test subjects only differences of about 0.107 can be detected.
- The coupled ODE does not change any classification decision: its read-out is the classifier probability with the
  threshold shifted by less than 0.001 (Proposition 1 of the paper).
- The ODE does not improve temporal consistency (a moving average or majority vote does) or forecasting (persistence
  is more accurate).
- Fitted to eyes-open/eyes-closed recordings, four rates converge to the lower bound and the two rates leaving the
  unobserved Passive state are not identifiable (Proposition 2). The rates are therefore not interpreted
  physiologically.
- Zeroing seven frontopolar channels reduces accuracy to 0.583, zeroing eight occipital channels only to 0.849: the
  classifier relies substantially on ocular activity.

## Architecture

```
EEG window (256 samples x 61 channels, 0.512 s)
        |
Linear projection 61 -> 256, LayerNorm, GELU
        |
3-layer bidirectional LSTM (256 units per direction) + LayerNorm
        |
Attention pooling over 256 time steps
        |
MLP 512 -> 256 -> 128 -> 2  ->  P(open), P(closed)
        |
Initial state  P0 = min(P_open, P_closed)/2,  A0 = (1-P0) P_open,  F0 = (1-P0) P_closed
Coupling       k'_AF, k'_PF scaled by (1 + alpha P_closed);  k'_FA, k'_PA scaled by (1 + alpha P_open)
        |
Three-state ODE  dA/dt, dP/dt, dF/dt  (A + P + F = 1), integrated for 20 s
        |
Eyes closed if F(20 s) > 0.5
```

## Datasets

The datasets are not redistributed with this repository.

**OpenNeuro ds004148** (main analysis): 60 healthy participants (18–28 years), three sessions, 61 EEG channels at
500 Hz; eyes-open and eyes-closed resting recordings of 300 s. Download the resting recordings (about 14 GB) to
`Dataset/II` with

```bash
python download_dataset.py --yes
```

**BCI Competition IV 2a and 2b** (classification-only results in Supplementary Note 1; the ODE is not applied to
motor imagery):

```bash
bash download_bci_iv.sh    # A01T-A09T.mat into "archive (1)/", B01T-B09E.mat into bciiv2b/
```

## Installation

```bash
git clone https://github.com/khurrameycon/LSTM-ODE-BCI.git
cd LSTM-ODE-BCI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The experiments were run with Python 3 and PyTorch on an NVIDIA GeForce RTX 2080 SUPER GPU (8 GB).
`requirements.txt` lists the minimum package versions.

## Reproducing the paper

Run the ds004148 pipeline from the repository root:

```bash
bash run_experiments.sh
```

| Script | Produces | Paper |
|---|---|---|
| `01_data_exploration.py` | class distribution | Fig. 4 |
| `02_preprocessing.py` | filtering, segment-wise artifact rejection, windows, subject-level split | Section 4.1, Supplementary Table S1 |
| `03_baseline_models.py` | SVM, Random Forest, XGBoost | Table 1 |
| `04_lstm_model.py` | LSTM-Attention training and evaluation | Table 1, Fig. 6 |
| `05_ode_model.py` | ODE fit (five seeds), identifiability profile, equilibrium | Table 8, Fig. 10, Supplementary Note 2 |
| `06_lstm_ode_integration.py` | coupling-strength sweep | Section 4.4, Fig. 8a |
| `07_explainability.py` | attention, gradient, permutation and KernelSHAP importance | Fig. 7, Fig. 9c, Supplementary Fig. S3 |
| `08_forecasting.py` | ODE forecast and persistence baseline | Table 6, Fig. 8c |
| `09_sensitivity_analysis.py` | ablation and hyperparameter sweep | Tables 3 and 4 |
| `11_deep_baselines.py` | EEGNet, DeepConvNet, CNN-LSTM, EEG-Transformer | Table 1, Supplementary Figs. S1 and S2 |
| `13_temporal_baselines.py` | moving average, majority vote, HMM, one-step ODE | Table 5, Fig. 8b |
| `15_ocular_spectral.py` | channel zeroing and occipital spectra | Table 7, Fig. 9a,b |
| `16_revision_figures.py` | figures drawn from the saved results | Figs. 5, 8, 9 and 10 |
| `17_paired_statistics.py` | paired permutation tests and paired CIs | Table 2 |
| `outputs/results/subject_tests_vs_lstm.json` | per-subject accuracy of every model, sign tests and d_z against LSTM-Attention | Table 2, Fig. 5, Supplementary Table S2 |
| `18_proposition_checks.py` | numerical checks of Propositions 1 and 2 | Supplementary Note 3 |
| `02b_preprocessing_bciiv2a.py`, `12c_within_subject_mi.py` | BCI-IV 2a, within subject | Supplementary Table S3 |
| `12d_bciiv2b.py` | BCI-IV 2b, within subject and session transfer | Supplementary Table S4 |

Other files: `revision_common.py` and `sequence_mmap.py` (shared helpers), `export_mmap_sequences.py` (memory-mapped
sequences for machines with little RAM), `10_three_state_probabilities.py` (per-participant three-state
probabilities), `12_deep_baselines_bciiv2a.py` and `regenerate_deep_baseline_plots.py` (subject-independent
BCI-IV 2a baselines from the previous version; not reported in the current manuscript).

## Results on ds004148 (eight held-out test subjects, 41,331 windows)

| Model | Accuracy | Subject mean [95% CI] | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| SVM | 0.885 | 0.887 [0.864, 0.911] | 0.849 | 0.933 | 0.889 | 0.960 |
| Random Forest | 0.889 | 0.881 [0.842, 0.918] | 0.869 | 0.913 | 0.890 | 0.954 |
| XGBoost | 0.902 | 0.903 [0.873, 0.928] | 0.876 | 0.934 | 0.904 | 0.967 |
| EEGNet | 0.912 | 0.910 [0.856, 0.957] | 0.943 | 0.876 | 0.908 | 0.970 |
| DeepConvNet | 0.904 | 0.912 [0.838, 0.963] | 0.956 | 0.845 | 0.897 | 0.960 |
| CNN-LSTM | 0.922 | 0.922 [0.890, 0.951] | 0.931 | 0.910 | 0.920 | 0.979 |
| EEG-Transformer | 0.911 | 0.910 [0.865, 0.949] | 0.954 | 0.862 | 0.906 | 0.971 |
| LSTM-Attention | 0.888 | 0.895 [0.835, 0.950] | 0.977 | 0.792 | 0.875 | 0.970 |
| LSTM-ODE (alpha = 0–1) | 0.888 | 0.895 [0.835, 0.950] | 0.977 | 0.792 | 0.875 | 0.970 |

Positive class: eyes closed. Confidence intervals are bootstrap intervals over test subjects.

<p align="center">
  <img src="docs/figures/fig27_deep_baselines_confusion_matrices.png" alt="Confusion matrices of the deep baselines" width="85%"/>
</p>
<p align="center">
  <img src="docs/figures/fig28_deep_baselines_roc_curves.png" alt="ROC curves of the deep baselines" width="55%"/>
</p>

## Citation

If you use this code, please cite the paper (citation details will be added on publication) and this archive:

> Umair, M. K., Arif, A., Abrar, M. F., Ali, S., Lee, I. E., & Jan, S. *LSTM-ODE-BCI* (code), Zenodo.

## License

MIT License; see [LICENSE](LICENSE).

## Contact

Muhammad Khurram Umair (corresponding author): umair19@hotmail.com
