"""
01_data_exploration.py
======================
EEG Eye State Dataset - Exploratory Data Analysis
OpenNeuro ds004148: Test-Retest Resting and Cognitive State EEG Dataset

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Dataset: OpenNeuro ds004148
- 60 subjects with 3 sessions each
- 64 EEG channels (Brain Products) at 500 Hz
- Tasks: eyes_closed, eyes_open (we use only these two for binary classification)
- Recording Duration: 300 seconds per task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
from pathlib import Path
import mne
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Paths
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / 'Dataset' / 'II'
OUTPUT_PATH = BASE_PATH / 'outputs' / 'figures'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Dataset parameters
SAMPLING_RATE = 500  # Hz
RECORDING_DURATION = 300  # seconds per task

# 62 EEG channel names (Brain Products 64-channel cap, excluding FCz reference and ground)
EEG_CHANNELS = [
    # Frontal
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    # Frontocentral
    'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
    # Central
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    # Centroparietal
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    # Parietal
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    # Parietooccipital
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    # Occipital
    'O1', 'Oz', 'O2'
]

# Brain regions mapping
CHANNEL_REGIONS = {
    'Frontal': ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'Frontocentral': ['FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8'],
    'Central': ['T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8'],
    'Centroparietal': ['TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10'],
    'Parietal': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'Parietooccipital': ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],
    'Occipital': ['O1', 'Oz', 'O2']
}


def get_all_recordings(downloaded_only=False):
    """Get paths to all EEG recordings in the BIDS dataset.

    Args:
        downloaded_only: If True, only return recordings with actual data

    Returns:
        list: List of dicts with subject, session, task, and file path info
    """
    recordings = []

    for subject_dir in sorted(DATASET_PATH.glob('sub-*')):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        for session_dir in sorted(subject_dir.glob('ses-*')):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            eeg_dir = session_dir / 'eeg'

            if not eeg_dir.exists():
                continue

            # Find eyes open and eyes closed recordings
            for task in ['eyesopen', 'eyesclosed']:
                vhdr_files = list(eeg_dir.glob(f'*task-{task}*_eeg.vhdr'))

                for vhdr_file in vhdr_files:
                    has_data = is_real_data(vhdr_file)

                    if downloaded_only and not has_data:
                        continue

                    recordings.append({
                        'subject': subject_id,
                        'session': session_id,
                        'task': task,
                        'vhdr_path': vhdr_file,
                        'label': 0 if task == 'eyesopen' else 1,  # 0=open, 1=closed
                        'has_data': has_data
                    })

    return recordings


def is_real_data(vhdr_path):
    """Check if a .vhdr file contains real data (not git-annex placeholder)."""
    try:
        with open(vhdr_path, 'r') as f:
            content = f.read(200)
        # Git-annex placeholders start with a path reference
        return 'Common Infos' in content or 'BrainVision' in content
    except:
        return False


def load_single_recording(vhdr_path, duration=None):
    """Load a single EEG recording from BIDS format.

    Args:
        vhdr_path: Path to .vhdr file
        duration: Optional duration in seconds to load (None for full)

    Returns:
        raw: MNE Raw object
        data: numpy array (n_channels, n_samples)
        ch_names: list of channel names
    """
    # Check if file is real data
    if not is_real_data(vhdr_path):
        return None, None, None

    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

        if duration:
            raw = raw.crop(tmax=duration)

        # Get EEG channels only
        raw.pick_types(eeg=True, exclude=[])

        data = raw.get_data()
        ch_names = raw.ch_names

        return raw, data, ch_names
    except Exception as e:
        print(f"Error loading {vhdr_path}: {e}")
        return None, None, None


def load_dataset_summary():
    """Load and summarize the entire dataset structure."""
    print("=" * 60)
    print("Loading OpenNeuro ds004148 Dataset")
    print("=" * 60)

    recordings = get_all_recordings(downloaded_only=False)
    recordings_with_data = [r for r in recordings if r.get('has_data', False)]

    print(f"\nTotal recordings found: {len(recordings)}")
    print(f"Recordings with downloaded data: {len(recordings_with_data)}")

    # Create summary dataframe
    df = pd.DataFrame(recordings)

    # Subject/session/task counts
    n_subjects = df['subject'].nunique()
    n_sessions_per_subject = df.groupby('subject')['session'].nunique().mean()

    print(f"\nDataset Structure:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Sessions per subject: {n_sessions_per_subject:.1f}")
    print(f"  Tasks: {df['task'].unique().tolist()}")

    # Class distribution
    task_counts = df['task'].value_counts()
    print(f"\nTask Distribution:")
    print(f"  Eyes Open: {task_counts.get('eyesopen', 0)} recordings")
    print(f"  Eyes Closed: {task_counts.get('eyesclosed', 0)} recordings")

    return recordings, df


def analyze_sample_recordings(recordings, n_samples=5):
    """Analyze a few sample recordings to understand data characteristics."""
    print("\n" + "=" * 60)
    print("Analyzing Sample Recordings")
    print("=" * 60)

    sample_stats = []

    # Only use recordings with actual data
    recordings_with_data = [r for r in recordings if r.get('has_data', False)]

    if not recordings_with_data:
        print("No recordings with downloaded data found!")
        return sample_stats

    # Sample from different subjects with data
    subjects = list(set([r['subject'] for r in recordings_with_data]))
    sample_subjects = subjects[:min(n_samples, len(subjects))]

    for subj in sample_subjects:
        # Get one recording from this subject
        subj_recordings = [r for r in recordings_with_data if r['subject'] == subj]
        rec = subj_recordings[0]

        raw, data, ch_names = load_single_recording(rec['vhdr_path'], duration=60)  # Load 60s

        if data is not None:
            sample_stats.append({
                'subject': rec['subject'],
                'session': rec['session'],
                'task': rec['task'],
                'n_channels': len(ch_names),
                'n_samples': data.shape[1],
                'duration_s': data.shape[1] / SAMPLING_RATE,
                'mean_amplitude': np.mean(np.abs(data)) * 1e6,  # Convert to uV
                'std_amplitude': np.std(data) * 1e6,
                'channels': ch_names[:5]  # First 5 channels
            })

            print(f"\n{rec['subject']} / {rec['session']} / {rec['task']}:")
            print(f"  Channels: {len(ch_names)}")
            print(f"  Samples: {data.shape[1]:,}")
            print(f"  Duration: {data.shape[1] / SAMPLING_RATE:.1f}s")
            print(f"  Mean amplitude: {np.mean(np.abs(data)) * 1e6:.2f} uV")

    return sample_stats


def basic_statistics(recordings, n_subjects_to_analyze=10):
    """Generate basic statistics for the dataset."""
    print("\n" + "=" * 60)
    print("Basic Statistics (Sampling Subjects)")
    print("=" * 60)

    all_means = {'eyesopen': [], 'eyesclosed': []}
    all_stds = {'eyesopen': [], 'eyesclosed': []}
    channel_stats = {task: {ch: [] for ch in EEG_CHANNELS} for task in ['eyesopen', 'eyesclosed']}

    # Sample subjects
    subjects = sorted(list(set([r['subject'] for r in recordings])))
    sample_subjects = subjects[:n_subjects_to_analyze]

    print(f"\nAnalyzing {len(sample_subjects)} subjects...")

    for subj in tqdm(sample_subjects, desc="Loading subjects"):
        subj_recs = [r for r in recordings if r['subject'] == subj]

        for rec in subj_recs:
            raw, data, ch_names = load_single_recording(rec['vhdr_path'], duration=30)

            if data is None:
                continue

            task = rec['task']
            all_means[task].append(np.mean(data) * 1e6)
            all_stds[task].append(np.std(data) * 1e6)

            # Per-channel stats
            for i, ch in enumerate(ch_names):
                if ch in channel_stats[task]:
                    channel_stats[task][ch].append(np.mean(np.abs(data[i])) * 1e6)

    # Print summary
    print(f"\nOverall Statistics:")
    for task in ['eyesopen', 'eyesclosed']:
        if all_means[task]:
            print(f"\n  {task.upper()}:")
            print(f"    Mean amplitude: {np.mean(all_means[task]):.2f} +/- {np.std(all_means[task]):.2f} uV")
            print(f"    Std amplitude: {np.mean(all_stds[task]):.2f} +/- {np.std(all_stds[task]):.2f} uV")

    return channel_stats


def plot_class_distribution(recordings_df):
    """Plot task/class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Task distribution
    task_counts = recordings_df['task'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Eyes Open', 'Eyes Closed']

    axes[0].pie([task_counts.get('eyesopen', 0), task_counts.get('eyesclosed', 0)],
                labels=labels, autopct='%1.1f%%', colors=colors,
                explode=[0.02, 0.02], shadow=True, textprops={'fontsize': 12})
    axes[0].set_title('Task Distribution', fontsize=14, fontweight='bold')

    # Recordings per subject
    subject_counts = recordings_df.groupby('subject').size()
    axes[1].hist(subject_counts.values, bins=20, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Number of Recordings', fontsize=12)
    axes[1].set_ylabel('Number of Subjects', fontsize=12)
    axes[1].set_title('Recordings per Subject Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(x=subject_counts.mean(), color='red', linestyle='--',
                    label=f'Mean: {subject_counts.mean():.1f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'fig01_class_distribution.png', bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / 'fig01_class_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: fig01_class_distribution.png/pdf")


def plot_eeg_time_series(recordings, n_seconds=5):
    """Plot EEG time series for sample recordings."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Get one eyes-open and one eyes-closed recording (with downloaded data)
    tasks = ['eyesopen', 'eyesclosed']
    colors = {'eyesopen': 'green', 'eyesclosed': 'red'}
    titles = {'eyesopen': 'Eyes Open Recording', 'eyesclosed': 'Eyes Closed Recording'}

    for idx, task in enumerate(tasks):
        task_recs = [r for r in recordings if r['task'] == task and r.get('has_data', False)]
        if not task_recs:
            print(f"No downloaded data found for task: {task}")
            continue

        rec = task_recs[0]
        raw, data, ch_names = load_single_recording(rec['vhdr_path'], duration=n_seconds)

        if data is None:
            continue

        ax = axes[idx]
        n_samples = data.shape[1]
        time = np.arange(n_samples) / SAMPLING_RATE

        # Plot first 10 channels
        channels_to_plot = ch_names[:10]
        for i, ch in enumerate(channels_to_plot):
            ch_idx = ch_names.index(ch)
            signal = data[ch_idx] * 1e6  # Convert to uV
            # Offset for visibility
            ax.plot(time, signal + i * 100, linewidth=0.5, alpha=0.8, label=ch)

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude (uV, offset)', fontsize=12)
        ax.set_title(f'{titles[task]} - {rec["subject"]}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlim([0, n_seconds])

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'fig02_eeg_time_series.png', bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / 'fig02_eeg_time_series.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig02_eeg_time_series.png/pdf")


def plot_spectral_analysis(recordings, n_subjects=3):
    """Perform spectral analysis on sample recordings."""
    print("\n" + "=" * 60)
    print("Spectral Analysis")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    band_powers = {'eyesopen': {b: [] for b in bands}, 'eyesclosed': {b: [] for b in bands}}

    # Sample subjects with downloaded data
    recordings_with_data = [r for r in recordings if r.get('has_data', False)]
    subjects = sorted(list(set([r['subject'] for r in recordings_with_data])))[:n_subjects]

    if not subjects:
        print("No subjects with downloaded data found!")
        plt.close()
        return band_powers

    for subj_idx, subj in enumerate(subjects):
        subj_recs = [r for r in recordings_with_data if r['subject'] == subj]

        for task in ['eyesopen', 'eyesclosed']:
            task_recs = [r for r in subj_recs if r['task'] == task]
            if not task_recs:
                continue

            rec = task_recs[0]
            raw, data, ch_names = load_single_recording(rec['vhdr_path'], duration=60)

            if data is None:
                continue

            # Use occipital channel (O1) for spectral analysis
            if 'O1' in ch_names:
                o1_idx = ch_names.index('O1')
                signal = data[o1_idx]

                # Compute PSD
                freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=1024)

                # Plot PSD
                ax = axes[0, subj_idx]
                color = 'green' if task == 'eyesopen' else 'red'
                label = 'Open' if task == 'eyesopen' else 'Closed'
                ax.semilogy(freqs, psd, color=color, linewidth=1.5, alpha=0.8, label=label)
                ax.set_xlim([0, 50])
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD (V²/Hz)')
                ax.set_title(f'{subj} - O1 Channel', fontweight='bold')
                ax.legend()

                # Calculate band powers
                for band_name, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(psd[mask])
                    band_powers[task][band_name].append(band_power)

    # Plot band power comparison
    ax = axes[1, 0]
    x = np.arange(len(bands))
    width = 0.35

    open_powers = [np.mean(band_powers['eyesopen'][b]) if band_powers['eyesopen'][b] else 0 for b in bands]
    closed_powers = [np.mean(band_powers['eyesclosed'][b]) if band_powers['eyesclosed'][b] else 0 for b in bands]

    ax.bar(x - width/2, open_powers, width, label='Eyes Open', color='green', alpha=0.7)
    ax.bar(x + width/2, closed_powers, width, label='Eyes Closed', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(bands.keys()))
    ax.set_ylabel('Band Power')
    ax.set_title('Average Band Power Comparison', fontweight='bold')
    ax.legend()
    ax.set_yscale('log')

    # Alpha power ratio (key biomarker)
    ax = axes[1, 1]
    if band_powers['eyesopen']['Alpha'] and band_powers['eyesclosed']['Alpha']:
        alpha_ratio = np.array(band_powers['eyesclosed']['Alpha']) / (np.array(band_powers['eyesopen']['Alpha']) + 1e-10)
        ax.bar(['Closed/Open'], [np.mean(alpha_ratio)], color='purple', alpha=0.7)
        ax.axhline(y=1, color='gray', linestyle='--')
        ax.set_ylabel('Alpha Power Ratio')
        ax.set_title('Alpha Power: Closed/Open Ratio', fontweight='bold')
        ax.text(0, np.mean(alpha_ratio) + 0.1, f'{np.mean(alpha_ratio):.2f}', ha='center')

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    SPECTRAL ANALYSIS SUMMARY
    =========================

    Dataset: OpenNeuro ds004148
    Sampling Rate: {SAMPLING_RATE} Hz
    Subjects Analyzed: {n_subjects}

    Key Finding:
    Alpha power (8-13 Hz) increases
    with eyes closed, consistent
    with neural alpha rhythm.

    This is the primary biomarker
    for eye state classification.
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'fig06_spectral_analysis.png', bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / 'fig06_spectral_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig06_spectral_analysis.png/pdf")

    return band_powers


def generate_summary_report(recordings_df, sample_stats):
    """Generate a summary report of the EDA findings."""
    print("\n" + "=" * 60)
    print("EDA SUMMARY REPORT")
    print("=" * 60)

    n_subjects = recordings_df['subject'].nunique()
    n_sessions = recordings_df.groupby('subject')['session'].nunique().mean()
    n_recordings = len(recordings_df)

    report = []
    report.append("# OpenNeuro ds004148 Dataset - Exploratory Data Analysis Report")
    report.append(f"\n## Dataset Overview")
    report.append(f"- **DOI**: 10.18112/openneuro.ds004148.v1.0.0")
    report.append(f"- **Total Subjects**: {n_subjects}")
    report.append(f"- **Sessions per Subject**: {n_sessions:.1f}")
    report.append(f"- **Total Recordings**: {n_recordings}")
    report.append(f"- **EEG Channels**: 62 (Brain Products 64-cap, excluding FCz reference)")
    report.append(f"- **Sampling Rate**: {SAMPLING_RATE} Hz")
    report.append(f"- **Recording Duration**: {RECORDING_DURATION} seconds per task")

    report.append(f"\n## Task Distribution")
    task_counts = recordings_df['task'].value_counts()
    report.append(f"- **Eyes Open**: {task_counts.get('eyesopen', 0)} recordings")
    report.append(f"- **Eyes Closed**: {task_counts.get('eyesclosed', 0)} recordings")

    report.append(f"\n## Data Quality")
    report.append(f"- BIDS format with proper electrode positions")
    report.append(f"- Reference: FCz")
    report.append(f"- Power line frequency: 50 Hz")

    report.append(f"\n## Key Findings for Eye State Classification")
    report.append(f"1. **Alpha rhythm**: Strong alpha (8-13 Hz) power increase with eyes closed")
    report.append(f"2. **Multi-subject**: Cross-subject validation possible with {n_subjects} subjects")
    report.append(f"3. **High temporal resolution**: {SAMPLING_RATE} Hz enables fine-grained analysis")
    report.append(f"4. **Test-retest**: {n_sessions:.0f} sessions per subject for reliability analysis")

    report.append(f"\n## Implications for Modeling")
    report.append(f"- Subject-wise train/test split required (no temporal leakage)")
    report.append(f"- 62 channels provide rich spatial information")
    report.append(f"- Large dataset enables deep learning approaches")
    report.append(f"- Multiple sessions enable within-subject validation")

    report.append(f"\n## Generated Figures")
    report.append(f"1. fig01_class_distribution.png - Task balance visualization")
    report.append(f"2. fig02_eeg_time_series.png - Raw EEG signals")
    report.append(f"3. fig06_spectral_analysis.png - Power spectral density")

    # Save report
    report_text = '\n'.join(report)
    report_path = BASE_PATH / 'outputs' / 'results' / 'eda_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nReport saved to: {report_path}")

    return report_text


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("EEG EYE STATE DATASET - EXPLORATORY DATA ANALYSIS")
    print("OpenNeuro ds004148")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("Target: Journal of Healthcare Informatics Research (JHIR)")
    print("=" * 60)

    # 1. Load dataset summary
    recordings, recordings_df = load_dataset_summary()

    if len(recordings) == 0:
        print("\nERROR: No recordings found. Check dataset path.")
        return None

    # 2. Analyze sample recordings
    sample_stats = analyze_sample_recordings(recordings, n_samples=3)

    # 3. Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    plot_class_distribution(recordings_df)
    plot_eeg_time_series(recordings, n_seconds=5)

    # 4. Spectral analysis
    band_powers = plot_spectral_analysis(recordings, n_subjects=3)

    # 5. Generate summary report
    generate_summary_report(recordings_df, sample_stats)

    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_PATH}")

    return recordings, recordings_df


if __name__ == "__main__":
    recordings, df = main()
