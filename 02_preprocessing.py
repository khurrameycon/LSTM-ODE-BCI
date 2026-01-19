"""
02_preprocessing.py
===================
EEG Data Preprocessing Pipeline for Eye State Classification
OpenNeuro ds004148: Test-Retest Resting and Cognitive State EEG Dataset

Authors: Ayesha, Muhammad Khurram Umair
Project: LSTM-ODE Framework for EEG-Based Cognitive State Evolution in BCIs
Target Journal: Journal of Healthcare Informatics Research (JHIR)

Preprocessing Steps:
1. Load raw EEG from BIDS format
2. Bandpass filter (1-45 Hz)
3. Z-score normalization (per channel)
4. Sequence creation with sliding windows
5. Subject-wise train/validation/test split
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
from pathlib import Path
import mne
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# Set random seed
np.random.seed(42)

# Paths
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / 'Dataset' / 'II'
OUTPUT_PATH = BASE_PATH / 'outputs'
PROCESSED_DATA_PATH = OUTPUT_PATH / 'processed_data'
FIGURES_PATH = OUTPUT_PATH / 'figures'

for path in [PROCESSED_DATA_PATH, FIGURES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset parameters
SAMPLING_RATE = 500  # Hz
SEQUENCE_LENGTH = 256  # samples (0.512 seconds at 500 Hz)
SEQUENCE_OVERLAP = 0.5  # 50% overlap
LOWCUT = 1.0  # Hz
HIGHCUT = 45.0  # Hz
FILTER_ORDER = 4

# Limit subjects for quick testing (set to None for all subjects)
MAX_SUBJECTS = 30  # Change to None to use all 60 subjects


def is_real_data(vhdr_path):
    """Check if a .vhdr file contains real data (not git-annex placeholder)."""
    try:
        with open(vhdr_path, 'r') as f:
            content = f.read(200)
        return 'Common Infos' in content or 'BrainVision' in content
    except:
        return False


def get_downloaded_recordings():
    """Get paths to all recordings with downloaded data."""
    recordings = []
    subjects_found = set()

    for subject_dir in sorted(DATASET_PATH.glob('sub-*')):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        # Limit number of subjects if MAX_SUBJECTS is set
        if MAX_SUBJECTS is not None and len(subjects_found) >= MAX_SUBJECTS:
            break

        for session_dir in sorted(subject_dir.glob('ses-*')):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            eeg_dir = session_dir / 'eeg'

            if not eeg_dir.exists():
                continue

            for task in ['eyesopen', 'eyesclosed']:
                vhdr_files = list(eeg_dir.glob(f'*task-{task}*_eeg.vhdr'))

                for vhdr_file in vhdr_files:
                    if is_real_data(vhdr_file):
                        recordings.append({
                            'subject': subject_id,
                            'session': session_id,
                            'task': task,
                            'vhdr_path': vhdr_file,
                            'label': 0 if task == 'eyesopen' else 1
                        })
                        subjects_found.add(subject_id)

    if MAX_SUBJECTS is not None:
        print(f"Limited to {MAX_SUBJECTS} subjects: {sorted(subjects_found)}")

    return recordings


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter.

    Args:
        data: EEG data (n_channels, n_samples)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def normalize_data(data, mean=None, std=None):
    """Z-score normalize data per channel.

    Args:
        data: EEG data (n_channels, n_samples)
        mean: Pre-computed mean (for test data)
        std: Pre-computed std (for test data)

    Returns:
        Normalized data, mean, std
    """
    if mean is None:
        mean = np.mean(data, axis=1, keepdims=True)
    if std is None:
        std = np.std(data, axis=1, keepdims=True)
        std[std < 1e-10] = 1e-10  # Prevent division by zero

    normalized = (data - mean) / std
    return normalized, mean.flatten(), std.flatten()


def create_sequences(data, label, seq_length, overlap):
    """Create overlapping sequences from continuous EEG.

    Args:
        data: EEG data (n_channels, n_samples)
        label: Class label (0 or 1)
        seq_length: Sequence length in samples
        overlap: Overlap fraction (0-1)

    Returns:
        X: Sequences (n_sequences, seq_length, n_channels)
        y: Labels (n_sequences,)
    """
    n_channels, n_samples = data.shape
    step = int(seq_length * (1 - overlap))

    sequences = []
    labels = []

    for start in range(0, n_samples - seq_length + 1, step):
        end = start + seq_length
        seq = data[:, start:end].T  # (seq_length, n_channels)
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def load_and_preprocess_recording(vhdr_path, label, normalization_params=None):
    """Load and preprocess a single EEG recording.

    Args:
        vhdr_path: Path to .vhdr file
        label: Class label (0=eyes open, 1=eyes closed)
        normalization_params: Dict with 'mean' and 'std' for normalization

    Returns:
        X: Sequences (n_sequences, seq_length, n_channels)
        y: Labels
        norm_params: Normalization parameters
    """
    try:
        # Load raw data
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude=[])

        data = raw.get_data()  # (n_channels, n_samples)
        ch_names = raw.ch_names

        # Bandpass filter
        data = bandpass_filter(data, LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)

        # Normalize
        if normalization_params:
            mean = np.array(normalization_params['mean']).reshape(-1, 1)
            std = np.array(normalization_params['std']).reshape(-1, 1)
            data, _, _ = normalize_data(data, mean, std)
        else:
            data, mean, std = normalize_data(data)
            normalization_params = {'mean': mean.tolist(), 'std': std.tolist()}

        # Create sequences
        X, y = create_sequences(data, label, SEQUENCE_LENGTH, SEQUENCE_OVERLAP)

        return X, y, ch_names, normalization_params

    except Exception as e:
        print(f"Error processing {vhdr_path}: {e}")
        return None, None, None, None


def split_subjects(recordings, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split recordings by subject for cross-subject validation.

    Args:
        recordings: List of recording dicts
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing

    Returns:
        train_recordings, val_recordings, test_recordings
    """
    # Get unique subjects
    subjects = sorted(list(set([r['subject'] for r in recordings])))
    n_subjects = len(subjects)

    print(f"Total subjects with data: {n_subjects}")

    if n_subjects < 3:
        # If only 1 subject, split by session
        print("Warning: Less than 3 subjects. Splitting by session instead.")
        sessions = sorted(list(set([r['session'] for r in recordings])))

        if len(sessions) >= 3:
            train_sessions = sessions[:int(len(sessions) * train_ratio)]
            remaining = sessions[int(len(sessions) * train_ratio):]
            val_sessions = remaining[:len(remaining) // 2]
            test_sessions = remaining[len(remaining) // 2:]

            train_recs = [r for r in recordings if r['session'] in train_sessions]
            val_recs = [r for r in recordings if r['session'] in val_sessions]
            test_recs = [r for r in recordings if r['session'] in test_sessions]
        else:
            # Only one session - split by time
            train_recs = recordings[:int(len(recordings) * train_ratio)]
            val_recs = recordings[int(len(recordings) * train_ratio):int(len(recordings) * (train_ratio + val_ratio))]
            test_recs = recordings[int(len(recordings) * (train_ratio + val_ratio)):]

        return train_recs, val_recs, test_recs

    # Normal case: split by subject
    np.random.shuffle(subjects)

    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    n_test = n_subjects - n_train - n_val

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    train_recs = [r for r in recordings if r['subject'] in train_subjects]
    val_recs = [r for r in recordings if r['subject'] in val_subjects]
    test_recs = [r for r in recordings if r['subject'] in test_subjects]

    return train_recs, val_recs, test_recs


def process_all_recordings(train_recs, val_recs, test_recs):
    """Process all recordings and create train/val/test sets.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, metadata
    """
    print("\n" + "=" * 60)
    print("Processing Training Data")
    print("=" * 60)

    X_train_list = []
    y_train_list = []
    normalization_params = None
    ch_names = None

    for rec in tqdm(train_recs, desc="Training"):
        X, y, channels, norm_params = load_and_preprocess_recording(
            rec['vhdr_path'], rec['label'], normalization_params
        )
        if X is not None:
            X_train_list.append(X)
            y_train_list.append(y)
            if normalization_params is None:
                normalization_params = norm_params
            if ch_names is None:
                ch_names = channels

    if not X_train_list:
        print("No training data processed!")
        return None, None, None, None, None, None, None

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    print(f"Training sequences: {X_train.shape}")
    print(f"Training labels: {np.bincount(y_train)}")

    print("\n" + "=" * 60)
    print("Processing Validation Data")
    print("=" * 60)

    X_val_list = []
    y_val_list = []

    for rec in tqdm(val_recs, desc="Validation"):
        X, y, _, _ = load_and_preprocess_recording(
            rec['vhdr_path'], rec['label'], normalization_params
        )
        if X is not None:
            X_val_list.append(X)
            y_val_list.append(y)

    if X_val_list:
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        print(f"Validation sequences: {X_val.shape}")
    else:
        X_val, y_val = None, None
        print("No validation data processed!")

    print("\n" + "=" * 60)
    print("Processing Test Data")
    print("=" * 60)

    X_test_list = []
    y_test_list = []

    for rec in tqdm(test_recs, desc="Test"):
        X, y, _, _ = load_and_preprocess_recording(
            rec['vhdr_path'], rec['label'], normalization_params
        )
        if X is not None:
            X_test_list.append(X)
            y_test_list.append(y)

    if X_test_list:
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        print(f"Test sequences: {X_test.shape}")
    else:
        X_test, y_test = None, None
        print("No test data processed!")

    # Metadata
    metadata = {
        'sampling_rate': SAMPLING_RATE,
        'sequence_length': SEQUENCE_LENGTH,
        'overlap': SEQUENCE_OVERLAP,
        'lowcut_hz': LOWCUT,
        'highcut_hz': HIGHCUT,
        'filter_order': FILTER_ORDER,
        'n_channels': X_train.shape[2] if X_train is not None else 0,
        'channel_names': ch_names if ch_names else [],
        'normalization_params': normalization_params,
        'train_subjects': list(set([r['subject'] for r in train_recs])),
        'val_subjects': list(set([r['subject'] for r in val_recs])) if val_recs else [],
        'test_subjects': list(set([r['subject'] for r in test_recs])) if test_recs else [],
        'n_train_sequences': len(X_train) if X_train is not None else 0,
        'n_val_sequences': len(X_val) if X_val is not None else 0,
        'n_test_sequences': len(X_test) if X_test is not None else 0
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, metadata):
    """Save processed data to files."""
    print("\n" + "=" * 60)
    print("Saving Processed Data")
    print("=" * 60)

    # Save numpy arrays
    np.savez_compressed(
        PROCESSED_DATA_PATH / 'processed_sequences.npz',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if X_val is not None else np.array([]),
        y_val=y_val if y_val is not None else np.array([]),
        X_test=X_test if X_test is not None else np.array([]),
        y_test=y_test if y_test is not None else np.array([])
    )
    print(f"Saved: processed_sequences.npz")

    # Save metadata
    with open(PROCESSED_DATA_PATH / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: preprocessing_metadata.json")


def plot_preprocessing_overview(X_train, y_train, X_val, y_val, X_test, y_test, metadata):
    """Generate preprocessing overview figure."""
    print("\n" + "=" * 60)
    print("Generating Preprocessing Overview")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Class distribution per split
    ax = axes[0, 0]
    splits = ['Train', 'Val', 'Test']
    class0 = [np.sum(y_train == 0) if y_train is not None else 0,
              np.sum(y_val == 0) if y_val is not None else 0,
              np.sum(y_test == 0) if y_test is not None else 0]
    class1 = [np.sum(y_train == 1) if y_train is not None else 0,
              np.sum(y_val == 1) if y_val is not None else 0,
              np.sum(y_test == 1) if y_test is not None else 0]

    x = np.arange(len(splits))
    width = 0.35
    ax.bar(x - width/2, class0, width, label='Open (0)', color='#3498db')
    ax.bar(x + width/2, class1, width, label='Closed (1)', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('Sequences')
    ax.set_title('Class Distribution per Split', fontweight='bold')
    ax.legend()

    # 2. Sample sequence visualization
    ax = axes[0, 1]
    if X_train is not None and len(X_train) > 0:
        sample_idx = np.random.randint(len(X_train))
        n_channels_to_plot = min(5, X_train.shape[2])
        for i in range(n_channels_to_plot):
            ax.plot(X_train[sample_idx, :, i] + i*3, alpha=0.8, linewidth=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude (offset)')
        ax.set_title(f'Sample Sequence (class={y_train[sample_idx]})', fontweight='bold')

    # 3. Class balance pie chart
    ax = axes[0, 2]
    if y_train is not None:
        ax.pie([np.sum(y_train == 0), np.sum(y_train == 1)],
               labels=['Eyes Open', 'Eyes Closed'],
               autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c'],
               explode=(0.05, 0.05))
        ax.set_title('Training Class Balance', fontweight='bold')

    # 4. Channel correlation heatmap
    ax = axes[1, 0]
    if X_train is not None and len(X_train) > 0:
        sample_data = X_train[:min(100, len(X_train))].reshape(-1, X_train.shape[2])
        corr = np.corrcoef(sample_data.T)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Channel Correlation', fontweight='bold')
        plt.colorbar(im, ax=ax)

    # 5. Dataset sizes bar chart
    ax = axes[1, 1]
    if X_train is not None:
        sizes = [len(X_train),
                 len(X_val) if X_val is not None else 0,
                 len(X_test) if X_test is not None else 0]
        bars = ax.bar(['Train', 'Val', 'Test'], sizes, color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_ylabel('Sequences')
        ax.set_title('Dataset Sizes', fontweight='bold')
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{size:,}', ha='center', va='bottom', fontsize=9)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    PREPROCESSING SUMMARY
    =====================

    Dataset: OpenNeuro ds004148
    Sampling Rate: {SAMPLING_RATE} Hz

    Sequence Length: {SEQUENCE_LENGTH} samples
    Duration: {SEQUENCE_LENGTH/SAMPLING_RATE:.3f} seconds
    Overlap: {SEQUENCE_OVERLAP*100:.0f}%

    Bandpass: {LOWCUT}-{HIGHCUT} Hz
    N Channels: {metadata['n_channels']}

    Train: {metadata['n_train_sequences']:,} sequences
    Val: {metadata['n_val_sequences']:,} sequences
    Test: {metadata['n_test_sequences']:,} sequences

    Subjects:
    - Train: {len(metadata['train_subjects'])}
    - Val: {len(metadata['val_subjects'])}
    - Test: {len(metadata['test_subjects'])}
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'fig07_preprocessing_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_PATH / 'fig07_preprocessing_overview.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig07_preprocessing_overview.png/pdf")


def print_summary(X_train, y_train, X_val, y_val, X_test, y_test, metadata):
    """Print preprocessing summary."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"\nDataset Parameters:")
    print(f"  Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"  Sequence Length: {SEQUENCE_LENGTH} samples ({SEQUENCE_LENGTH/SAMPLING_RATE:.3f} seconds)")
    print(f"  Overlap: {SEQUENCE_OVERLAP*100:.0f}%")
    print(f"  Bandpass Filter: {LOWCUT}-{HIGHCUT} Hz")
    print(f"  N Channels: {metadata['n_channels']}")

    print(f"\nData Splits:")
    if X_train is not None:
        print(f"  Training:   {X_train.shape[0]:,} sequences (Class 0: {np.sum(y_train==0):,}, Class 1: {np.sum(y_train==1):,})")
    if X_val is not None and len(X_val) > 0:
        print(f"  Validation: {X_val.shape[0]:,} sequences (Class 0: {np.sum(y_val==0):,}, Class 1: {np.sum(y_val==1):,})")
    if X_test is not None and len(X_test) > 0:
        print(f"  Test:       {X_test.shape[0]:,} sequences (Class 0: {np.sum(y_test==0):,}, Class 1: {np.sum(y_test==1):,})")

    print(f"\nSubject Split:")
    print(f"  Train subjects: {metadata['train_subjects']}")
    print(f"  Val subjects: {metadata['val_subjects']}")
    print(f"  Test subjects: {metadata['test_subjects']}")


def main():
    """Main preprocessing function."""
    print("\n" + "=" * 60)
    print("EEG DATA PREPROCESSING PIPELINE")
    print("OpenNeuro ds004148")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("=" * 60)

    # Get downloaded recordings
    print("\nLoading recordings...")
    recordings = get_downloaded_recordings()
    print(f"Found {len(recordings)} recordings with downloaded data")

    if not recordings:
        print("ERROR: No recordings found with downloaded data!")
        print("Please run download_dataset.py first.")
        return

    # Split by subject
    train_recs, val_recs, test_recs = split_subjects(recordings)

    # Process all data
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = process_all_recordings(
        train_recs, val_recs, test_recs
    )

    if X_train is None:
        print("ERROR: No data processed!")
        return

    # Save data
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, metadata)

    # Generate visualization
    plot_preprocessing_overview(X_train, y_train, X_val, y_val, X_test, y_test, metadata)

    # Print summary
    print_summary(X_train, y_train, X_val, y_val, X_test, y_test, metadata)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


if __name__ == "__main__":
    main()
