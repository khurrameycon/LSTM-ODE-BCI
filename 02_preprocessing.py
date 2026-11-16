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
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from revision_common import design_power

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# Set random seed
np.random.seed(42)

# Paths
BASE_PATH = Path(__file__).parent
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

# None uses every subject that has been downloaded (the paper uses all 60).
MAX_SUBJECTS = None
SPLIT_SEED = 42
# A sample is rejected when any channel exceeds this amplitude. The
# surrounding samples are not spliced together; windowing restarts after
# the gap. See Major Comment 8.
ARTIFACT_THRESHOLD_UV = 100.0


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


def clean_segments(data, threshold_uv=ARTIFACT_THRESHOLD_UV):
    """Return index ranges that stay inside ±threshold on every channel.

    A bad sample ends the current segment. The next clean sample starts a
    new segment. Segments are never concatenated, so a window cannot
    contain the splice that artifact deletion would create.
    """
    threshold_v = threshold_uv * 1e-6
    bad = np.any(np.abs(data) > threshold_v, axis=0)
    segments = []
    start = None
    for index, is_bad in enumerate(bad):
        if not is_bad and start is None:
            start = index
        elif is_bad and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, data.shape[1]))
    return segments, int(bad.sum())


def create_sequences_from_segments(data, label, segments, seq_length, overlap):
    """Window only inside contiguous clean segments of one recording.

    `data` is (n_channels, n_samples). Each recording is a single
    eyes-open or eyes-closed file, so a window cannot cross a condition
    boundary. Windows shorter than `seq_length` are dropped rather than
    padded across a gap.
    """
    step = max(1, int(seq_length * (1 - overlap)))
    sequences = []
    labels = []
    for start, end in segments:
        for cursor in range(start, end - seq_length + 1, step):
            sequences.append(data[:, cursor:cursor + seq_length].T)
            labels.append(label)
    if not sequences:
        return (
            np.empty((0, seq_length, data.shape[0]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_filtered_recording(vhdr_path):
    """Load one BrainVision file and bandpass it. Values stay in volts."""
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude=[])
    data = bandpass_filter(raw.get_data(), LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)
    return data, list(raw.ch_names)


def training_normalization(train_recs):
    """Channel mean and variance from clean training samples only."""
    total = None
    total_sq = None
    count = 0
    channel_names = None
    used = 0
    for rec in tqdm(train_recs, desc="Train normalization"):
        try:
            data, names = load_filtered_recording(rec['vhdr_path'])
        except Exception as exc:
            print(f"Skipping {rec['vhdr_path']}: {exc}")
            continue
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            print(f"Skipping {rec['subject']}: channel montage does not match")
            continue
        segments, _ = clean_segments(data)
        if not segments:
            continue
        clean = np.concatenate([data[:, start:end] for start, end in segments], axis=1)
        if total is None:
            total = clean.sum(axis=1)
            total_sq = np.square(clean).sum(axis=1)
        else:
            total += clean.sum(axis=1)
            total_sq += np.square(clean).sum(axis=1)
        count += clean.shape[1]
        used += 1
    if count == 0:
        raise RuntimeError("No clean training samples available for normalization")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    print(f"Normalization from {used} training recordings, {count:,} clean samples")
    return mean, std, channel_names


def window_recording(vhdr_path, label, mean, std, channel_names):
    """Filter, reject by amplitude, normalize with training stats, window."""
    data, names = load_filtered_recording(vhdr_path)
    if names != channel_names:
        raise ValueError("channel montage does not match the training montage")
    segments, n_rejected = clean_segments(data)
    normalized = (data - mean[:, None]) / std[:, None]
    sequences, labels = create_sequences_from_segments(
        normalized, label, segments, SEQUENCE_LENGTH, SEQUENCE_OVERLAP
    )
    return sequences, labels, n_rejected, int(data.shape[1])


def split_subjects(recordings, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Subject-level split. The same subject never appears in two splits.

    The assignment is the shuffle of the sorted subject list under
    SPLIT_SEED. There is no chronological or session fallback: those
    protocols are a different experiment and were being mixed with this one.
    """
    subjects = sorted({r['subject'] for r in recordings})
    n_subjects = len(subjects)
    print(f"Total subjects with data: {n_subjects}")
    if n_subjects < 3:
        raise RuntimeError(
            "Need at least 3 subjects for a subject-level train/validation/test "
            "split. Download more of ds004148; do not fall back to a session "
            "or chronological split."
        )

    order = np.array(subjects)
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(order)

    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    if n_train + n_val >= n_subjects:
        n_val = max(1, n_subjects - n_train - 1)
    train_subjects = order[:n_train].tolist()
    val_subjects = order[n_train:n_train + n_val].tolist()
    test_subjects = order[n_train + n_val:].tolist()

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

    train_set, val_set, test_set = set(train_subjects), set(val_subjects), set(test_subjects)
    train_recs = [r for r in recordings if r['subject'] in train_set]
    val_recs = [r for r in recordings if r['subject'] in val_set]
    test_recs = [r for r in recordings if r['subject'] in test_set]
    return train_recs, val_recs, test_recs, train_subjects, val_subjects, test_subjects


def _process_split(recordings, mean, std, channel_names, description):
    sequences, labels, subjects, recording_ids = [], [], [], []
    rejected = 0
    total_samples = 0
    for rec in tqdm(recordings, desc=description):
        try:
            x_rec, y_rec, n_rejected, n_samples = window_recording(
                rec['vhdr_path'], rec['label'], mean, std, channel_names
            )
        except Exception as exc:
            print(f"Skipping {rec['vhdr_path']}: {exc}")
            continue
        rejected += n_rejected
        total_samples += n_samples
        if len(x_rec) == 0:
            continue
        recording_key = f"{rec['subject']}_{rec['session']}_{rec['task']}"
        sequences.append(x_rec)
        labels.append(y_rec)
        subjects.append(np.full(len(x_rec), rec['subject']))
        recording_ids.append(np.full(len(x_rec), recording_key))
    if not sequences:
        empty_x = np.empty((0, SEQUENCE_LENGTH, len(channel_names)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_s = np.empty((0,), dtype='<U32')
        return empty_x, empty_y, empty_s, empty_s, rejected, total_samples
    return (
        np.concatenate(sequences, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(subjects, axis=0),
        np.concatenate(recording_ids, axis=0),
        rejected,
        total_samples,
    )


def process_all_recordings(train_recs, val_recs, test_recs,
                           train_subjects, val_subjects, test_subjects):
    """Process every split with training-set normalization and no splicing."""
    print("\n" + "=" * 60)
    print("Estimating training-set normalization")
    print("=" * 60)
    mean, std, channel_names = training_normalization(train_recs)

    print("\n" + "=" * 60)
    print("Processing Training Data")
    print("=" * 60)
    X_train, y_train, sub_train, rec_train, rej_train, n_train = _process_split(
        train_recs, mean, std, channel_names, "Training"
    )
    print(f"Training sequences: {X_train.shape}")

    print("\n" + "=" * 60)
    print("Processing Validation Data")
    print("=" * 60)
    X_val, y_val, sub_val, rec_val, rej_val, n_val = _process_split(
        val_recs, mean, std, channel_names, "Validation"
    )
    print(f"Validation sequences: {X_val.shape}")

    print("\n" + "=" * 60)
    print("Processing Test Data")
    print("=" * 60)
    X_test, y_test, sub_test, rec_test, rej_test, n_test = _process_split(
        test_recs, mean, std, channel_names, "Test"
    )
    print(f"Test sequences: {X_test.shape}")

    if len(X_train) == 0:
        print("No training data processed!")
        return None

    step_samples = max(1, int(SEQUENCE_LENGTH * (1 - SEQUENCE_OVERLAP)))
    metadata = {
        'sampling_rate': SAMPLING_RATE,
        'sequence_length': SEQUENCE_LENGTH,
        'overlap': SEQUENCE_OVERLAP,
        'step_samples': step_samples,
        'step_seconds': step_samples / SAMPLING_RATE,
        'lowcut_hz': LOWCUT,
        'highcut_hz': HIGHCUT,
        'filter_order': FILTER_ORDER,
        'n_channels': int(X_train.shape[2]),
        'channel_names': channel_names,
        'normalization': 'z-score from clean training samples only',
        'normalization_params': {'mean': mean.tolist(), 'std': std.tolist()},
        'split_protocol': 'subject-level',
        'split_seed': SPLIT_SEED,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'n_train_sequences': int(len(X_train)),
        'n_val_sequences': int(len(X_val)),
        'n_test_sequences': int(len(X_test)),
        'artifact_threshold_uv': ARTIFACT_THRESHOLD_UV,
        'artifact_policy': (
            'Samples beyond ±100 µV on any channel end the current segment. '
            'Windowing restarts after the gap. Rejected samples are not deleted '
            'and concatenated.'
        ),
        'rejected_samples': {
            'train': int(rej_train),
            'val': int(rej_val),
            'test': int(rej_test),
        },
        'raw_samples': {
            'train': int(n_train),
            'val': int(n_val),
            'test': int(n_test),
        },
        'label_policy': (
            'Each file is one eyes-open or eyes-closed condition. Every window '
            'inherits that file label. Windows that would span a condition '
            'boundary are not created. Majority vote is not used.'
        ),
        'windows_spanning_condition_boundary': 0,
        'power_analysis': design_power(len(test_subjects)),
    }
    groups = {
        'subject_id_train': sub_train,
        'subject_id_val': sub_val,
        'subject_id_test': sub_test,
        'recording_id_train': rec_train,
        'recording_id_val': rec_val,
        'recording_id_test': rec_test,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata, groups


def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test,
                        metadata, groups, output_name='processed_sequences.npz'):
    """Save processed arrays, subject ids, and the split list."""
    print("\n" + "=" * 60)
    print("Saving Processed Data")
    print("=" * 60)

    np.savez_compressed(
        PROCESSED_DATA_PATH / output_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if X_val is not None else np.array([]),
        y_val=y_val if y_val is not None else np.array([]),
        X_test=X_test if X_test is not None else np.array([]),
        y_test=y_test if y_test is not None else np.array([]),
        **groups,
    )
    print(f"Saved: {output_name}")

    with open(PROCESSED_DATA_PATH / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(PROCESSED_DATA_PATH / 'subject_splits.json', 'w') as f:
        json.dump({
            'seed': SPLIT_SEED,
            'protocol': 'subject-level',
            'train_subjects': metadata['train_subjects'],
            'val_subjects': metadata['val_subjects'],
            'test_subjects': metadata['test_subjects'],
        }, f, indent=2)
    print("Saved: preprocessing_metadata.json, subject_splits.json")


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
    import argparse
    global SEQUENCE_LENGTH, MAX_SUBJECTS

    parser = argparse.ArgumentParser(description='Preprocess ds004148')
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH,
                        help='Window length in samples. Use 128, 256, and 512 for the sensitivity run.')
    parser.add_argument('--output-tag', default='',
                        help='Suffix for the npz name. Empty keeps processed_sequences.npz.')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Optional cap. The paper run leaves this unset.')
    args = parser.parse_args()
    SEQUENCE_LENGTH = args.sequence_length
    if args.max_subjects is not None:
        MAX_SUBJECTS = args.max_subjects

    print("\n" + "=" * 60)
    print("EEG DATA PREPROCESSING PIPELINE")
    print("OpenNeuro ds004148")
    print("=" * 60)
    print("Authors: Ayesha, Muhammad Khurram Umair")
    print("=" * 60)

    print("\nLoading recordings...")
    recordings = get_downloaded_recordings()
    print(f"Found {len(recordings)} recordings with downloaded data")

    if not recordings:
        print("ERROR: No recordings found with downloaded data!")
        print("Please run download_dataset.py first.")
        return

    train_recs, val_recs, test_recs, train_subjects, val_subjects, test_subjects = split_subjects(recordings)

    processed = process_all_recordings(
        train_recs, val_recs, test_recs, train_subjects, val_subjects, test_subjects
    )

    if processed is None:
        print("ERROR: No data processed!")
        return
    X_train, y_train, X_val, y_val, X_test, y_test, metadata, groups = processed

    output_name = 'processed_sequences.npz'
    if args.output_tag:
        output_name = f'processed_sequences_{args.output_tag}.npz'
    save_processed_data(
        X_train, y_train, X_val, y_val, X_test, y_test, metadata, groups, output_name
    )

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
