# LSTM-ODE-BCI: Deep Learning and Dynamical Modeling for EEG-Based Cognitive State Evolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel unified framework integrating Long Short-Term Memory (LSTM) networks with a three-state compartmental Ordinary Differential Equation (ODE) model for interpretable cognitive state evolution in Brain-Computer Interfaces (BCIs).

## Overview

This repository contains the implementation for the paper:

> **Deep Learning and Dynamical Modeling Framework for EEG-Based Cognitive State Evolution in Brain-Computer Interfaces**
>
> Muhammad Khurram Umair¹*, Ayesha Arif², Muhammad Faisal Abrar³
>
> ¹University of Engineering and Technology, Peshawar, Pakistan
> ²National University of Sciences and Technology (NUST), Islamabad, Pakistan
> ³University of Hail, Saudi Arabia

## Key Features

- **LSTM-ODE Integration**: Probabilistic coupling between deep learning classification and mechanistic ODE dynamics
- **Three-State Cognitive Model**: Active, Passive, and Fatigued states with interpretable transition rates
- **Multi-Method Explainability**: Gradient-based attribution, permutation importance, and SHAP analysis
- **Comprehensive Pipeline**: From raw EEG preprocessing to model evaluation and visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EEG Input Sequence                       │
│                   (T × C) = 256 × 61                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Bidirectional LSTM                         │
│              (3 layers, 128 hidden units)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Self-Attention Pooling                       │
│                  (Temporal weighting)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Classification Head                           │
│            P(open), P(closed) → [0, 1]                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │   Probabilistic Coupling      │
          │   k'_AF = k_AF(1 + αP_closed) │
          │   k'_FA = k_FA(1 + αP_open)   │
          └───────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Three-State ODE Model                          │
│         dA/dt, dP/dt, dF/dt (A + P + F = 1)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            Cognitive State Trajectory                       │
│              [Active, Passive, Fatigued]                    │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

This project uses the **OpenNeuro ds004148** dataset:

- **Source**: [OpenNeuro ds004148](https://openneuro.org/datasets/ds004148)
- **DOI**: 10.18112/openneuro.ds004148.v1.0.0
- **Participants**: 60 subjects
- **Sessions**: 3 per participant
- **Channels**: 61 EEG channels (Brain Products, 64-electrode system)
- **Sampling Rate**: 500 Hz
- **Tasks**: Resting state (eyes open/closed), cognitive tasks

### Download Dataset

```bash
# Using OpenNeuro CLI
openneuro download ds004148

# Or download manually from:
# https://openneuro.org/datasets/ds004148
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/khurrameycon/LSTM-ODE-BCI.git
cd LSTM-ODE-BCI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
mne>=1.5.0
shap>=0.42.0
tqdm>=4.65.0
xgboost>=2.0.0
```

## Project Structure

```
LSTM-ODE-BCI/
├── 01_data_exploration.py      # Dataset exploration and statistics
├── 02_preprocessing.py         # EEG preprocessing pipeline
├── 03_baseline_models.py       # SVM, Random Forest, XGBoost baselines
├── 04_lstm_model.py            # Enhanced LSTM with attention
├── 05_ode_model.py             # Three-state cognitive ODE model
├── 06_lstm_ode_integration.py  # Unified LSTM-ODE framework
├── 07_explainability.py        # Gradient, permutation, SHAP analysis
├── 08_forecasting.py           # State trajectory forecasting
├── 09_sensitivity_analysis.py  # Ablation studies
├── download_dataset.py         # Dataset download utility
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### 1. Data Exploration

```bash
python 01_data_exploration.py
```

Generates dataset statistics, channel distributions, and temporal patterns.

### 2. Preprocessing

```bash
python 02_preprocessing.py
```

Applies bandpass filtering (1-45 Hz), z-score normalization, and sequence creation.

### 3. Baseline Models

```bash
python 03_baseline_models.py
```

Trains and evaluates SVM, Random Forest, and XGBoost classifiers.

### 4. LSTM Training

```bash
python 04_lstm_model.py
```

Trains the enhanced LSTM with attention mechanism.

### 5. ODE Model Fitting

```bash
python 05_ode_model.py
```

Fits the three-state cognitive ODE parameters.

### 6. LSTM-ODE Integration

```bash
python 06_lstm_ode_integration.py
```

Combines LSTM and ODE with probabilistic coupling.

### 7. Explainability Analysis

```bash
python 07_explainability.py
# Or skip SHAP (faster):
python 07_explainability.py --skip-shap
```

Computes channel importance using multiple methods.

## Results

### Model Performance

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| SVM | 38.0% | 0.000 | 0.467 |
| Random Forest | 60.3% | 0.563 | 0.657 |
| XGBoost | **62.0%** | **0.632** | **0.689** |
| LSTM-Attention | 54.9% | 0.603 | 0.596 |
| LSTM-ODE | 54.9% | 0.599 | 0.596 |

### Learned ODE Parameters

| Transition | Rate | Time Constant | Interpretation |
|------------|------|---------------|----------------|
| Passive → Fatigued | 0.626 | 1.6s | Rapid fatigue onset |
| Fatigued → Active | 0.139 | 7.2s | Slower recovery |
| Active → Fatigued | 0.095 | 10.6s | Gradual fatigue |
| Active → Passive | 0.020 | 50.4s | Slow disengagement |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{umair2024lstmode,
  title={Deep Learning and Dynamical Modeling Framework for EEG-Based
         Cognitive State Evolution in Brain-Computer Interfaces},
  author={Umair, Muhammad Khurram and Arif, Ayesha and Abrar, Muhammad Faisal},
  journal={Journal of Healthcare Informatics Research},
  year={2024},
  publisher={Springer}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenNeuro for hosting the ds004148 dataset
- University of Engineering and Technology, Peshawar
- National University of Sciences and Technology (NUST), Islamabad
- University of Hail, Saudi Arabia

## Contact

- **Muhammad Khurram Umair** - khurram.umair@uetpeshawar.edu.pk
- **Ayesha Arif** - ayesha.arif@nust.edu.pk
- **Muhammad Faisal Abrar** - mfabrar@uoh.edu.sa
