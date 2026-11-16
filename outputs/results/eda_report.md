# OpenNeuro ds004148 Dataset - Exploratory Data Analysis Report

## Dataset Overview
- **DOI**: 10.18112/openneuro.ds004148.v1.0.0
- **Total Subjects**: 60
- **Sessions per Subject**: 3.0
- **Total Recordings**: 360
- **EEG Channels**: 62 (Brain Products 64-cap, excluding FCz reference)
- **Sampling Rate**: 500 Hz
- **Recording Duration**: 300 seconds per task

## Task Distribution
- **Eyes Open**: 180 recordings
- **Eyes Closed**: 180 recordings

## Data Quality
- BIDS format with proper electrode positions
- Reference: FCz
- Power line frequency: 50 Hz

## Key Findings for Eye State Classification
1. **Alpha rhythm**: Strong alpha (8-13 Hz) power increase with eyes closed
2. **Multi-subject**: Cross-subject validation possible with 60 subjects
3. **High temporal resolution**: 500 Hz enables fine-grained analysis
4. **Test-retest**: 3 sessions per subject for reliability analysis

## Implications for Modeling
- Subject-wise train/test split required (no temporal leakage)
- 62 channels provide rich spatial information
- Large dataset enables deep learning approaches
- Multiple sessions enable within-subject validation

## Generated Figures
1. fig01_class_distribution.png - Task balance visualization
2. fig02_eeg_time_series.png - Raw EEG signals
3. fig06_spectral_analysis.png - Power spectral density