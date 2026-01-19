"""
download_dataset.py
===================
Download OpenNeuro ds004148 EEG data for Eyes Open/Closed classification

This script downloads the raw EEG data directly from OpenNeuro's S3 storage.
Only downloads eyesopen and eyesclosed tasks (needed for our study).

Dataset: https://openneuro.org/datasets/ds004148
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from urllib.parse import urljoin
import time

# Configuration
DATASET_ID = "ds004148"
VERSION = "1.0.0"
BASE_URL = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/"

# Local paths
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / 'Dataset' / 'II'

# What to download
N_SUBJECTS = 60
SESSIONS = ['session1', 'session2', 'session3']
TASKS = ['eyesopen', 'eyesclosed']  # Only these two tasks
FILE_EXTENSIONS = ['.vhdr', '.vmrk', '.eeg']

# Download settings
MAX_WORKERS = 4
TIMEOUT = 60
MAX_RETRIES = 3


def get_file_urls():
    """Generate URLs for all files to download."""
    files_to_download = []

    for sub_num in range(1, N_SUBJECTS + 1):
        sub_id = f"sub-{sub_num:02d}"

        for session in SESSIONS:
            ses_id = f"ses-{session}"

            for task in TASKS:
                base_name = f"{sub_id}_{ses_id}_task-{task}_eeg"

                for ext in FILE_EXTENSIONS:
                    filename = f"{base_name}{ext}"
                    url = f"{BASE_URL}{sub_id}/{ses_id}/eeg/{filename}"
                    local_path = DATASET_PATH / sub_id / ses_id / 'eeg' / filename

                    files_to_download.append({
                        'url': url,
                        'local_path': local_path,
                        'subject': sub_id,
                        'session': ses_id,
                        'task': task,
                        'extension': ext
                    })

    return files_to_download


def download_file(file_info, progress_bar=None):
    """Download a single file with retries."""
    url = file_info['url']
    local_path = file_info['local_path']

    # Skip if already exists and has content
    if local_path.exists():
        # Check if it's a git-annex placeholder (small text file)
        if local_path.stat().st_size > 200:  # Real EEG files are much larger
            return {'status': 'skipped', 'file': str(local_path)}

    # Create directory if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True)

            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if progress_bar:
                    progress_bar.update(1)

                return {'status': 'success', 'file': str(local_path), 'size': total_size}

            elif response.status_code == 404:
                return {'status': 'not_found', 'file': str(local_path)}

            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return {'status': 'error', 'file': str(local_path), 'error': str(e)}

    return {'status': 'failed', 'file': str(local_path)}


def download_all_parallel(files, max_workers=MAX_WORKERS):
    """Download all files using parallel threads."""
    results = {'success': 0, 'skipped': 0, 'not_found': 0, 'error': 0}

    with tqdm(total=len(files), desc="Downloading") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_file, f, pbar): f for f in files}

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results[result['status']] = results.get(result['status'], 0) + 1

    return results


def download_sequential(files):
    """Download files sequentially (more reliable but slower)."""
    results = {'success': 0, 'skipped': 0, 'not_found': 0, 'error': 0}

    for file_info in tqdm(files, desc="Downloading"):
        result = download_file(file_info)
        if result:
            results[result['status']] = results.get(result['status'], 0) + 1

            if result['status'] == 'error':
                print(f"\n  Error: {result.get('error', 'Unknown')}")

    return results


def estimate_download_size():
    """Estimate total download size."""
    # Each .eeg file is approximately:
    # 300 seconds * 500 Hz * 64 channels * 4 bytes = ~38.4 MB
    n_recordings = N_SUBJECTS * len(SESSIONS) * len(TASKS)
    eeg_size_mb = 38.4
    vhdr_size_kb = 2  # Small header files
    vmrk_size_kb = 1  # Small marker files

    total_gb = (n_recordings * eeg_size_mb +
                n_recordings * (vhdr_size_kb + vmrk_size_kb) / 1024) / 1024

    return n_recordings, total_gb


def main(auto_confirm=False):
    """Main download function."""
    print("=" * 60)
    print("OpenNeuro ds004148 Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset: Test-Retest Resting and Cognitive State EEG")
    print(f"Tasks to download: {', '.join(TASKS)}")
    print(f"Subjects: {N_SUBJECTS}")
    print(f"Sessions per subject: {len(SESSIONS)}")

    n_recordings, total_gb = estimate_download_size()
    print(f"\nEstimated download:")
    print(f"  - Total recordings: {n_recordings}")
    print(f"  - Estimated size: ~{total_gb:.1f} GB")

    # Get file list
    print("\nGenerating file list...")
    files = get_file_urls()
    print(f"Total files to check: {len(files)}")

    # Ask user to confirm (skip if auto_confirm)
    print(f"\nTarget directory: {DATASET_PATH}")
    if auto_confirm:
        print("Auto-confirm enabled, proceeding with download...")
        response = 'y'
    else:
        response = input("\nProceed with download? (y/n): ").strip().lower()

    if response != 'y':
        print("Download cancelled.")
        return

    # Download
    print("\nStarting download (this may take a while)...")
    print("Note: Files already downloaded will be skipped.\n")

    # Use sequential download for reliability
    results = download_sequential(files)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Downloaded: {results.get('success', 0)}")
    print(f"  Skipped (already exists): {results.get('skipped', 0)}")
    print(f"  Not found: {results.get('not_found', 0)}")
    print(f"  Errors: {results.get('error', 0)}")

    return results


def test_single_download():
    """Test downloading a single file."""
    print("Testing download of a single file...")

    # Test URL for sub-01, session1, eyesclosed
    test_file = {
        'url': f"{BASE_URL}sub-01/ses-session1/eeg/sub-01_ses-session1_task-eyesclosed_eeg.vhdr",
        'local_path': DATASET_PATH / 'sub-01' / 'ses-session1' / 'eeg' / 'sub-01_ses-session1_task-eyesclosed_eeg.vhdr.test',
        'subject': 'sub-01',
        'session': 'ses-session1',
        'task': 'eyesclosed',
        'extension': '.vhdr'
    }

    print(f"URL: {test_file['url']}")
    result = download_file(test_file)
    print(f"Result: {result}")

    # Clean up test file
    if test_file['local_path'].exists():
        test_file['local_path'].unlink()

    return result


if __name__ == "__main__":
    import sys
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv

    # First test if downloads work
    print("Testing OpenNeuro S3 access...")
    test_result = test_single_download()

    if test_result and test_result['status'] in ['success', 'skipped']:
        print("\nTest successful! Starting full download...\n")
        main(auto_confirm=auto_confirm)
    else:
        print("\nTest failed. OpenNeuro S3 bucket may not be directly accessible.")
        print("Alternative: Download from https://openneuro.org/datasets/ds004148")
