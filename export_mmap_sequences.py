"""Write uncompressed .npy copies so training can memory-map them.

One array is expanded at a time. The SVM process is left running.
"""

import gc
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
SOURCE = BASE / "outputs" / "processed_data" / "processed_sequences.npz"
DEST = BASE / "outputs" / "processed_data" / "mmap"
KEYS = (
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "subject_id_test",
)
MIN_FREE_KB = 16 * 1024 * 1024


def mem_available_kb():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable"):
                return int(line.split()[1])
    return 0


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ready = DEST / "READY"
    if ready.exists():
        print("mmap export already complete")
        return
    source = np.load(SOURCE, allow_pickle=True)
    for key in KEYS:
        if key not in source.files:
            print(f"skip missing key {key}")
            continue
        target = DEST / f"{key}.npy"
        if target.exists():
            print(f"exists {target.name}")
            continue
        free_kb = mem_available_kb()
        print(f"export {key} (MemAvailable {free_kb / 1024 / 1024:.1f} GB)")
        if free_kb < MIN_FREE_KB:
            raise SystemExit(
                f"Refusing to export {key}: only {free_kb / 1024 / 1024:.1f} GB available"
            )
        array = np.ascontiguousarray(source[key])
        np.save(target, array)
        print(f"wrote {target.name} shape={array.shape} dtype={array.dtype}")
        del array
        gc.collect()
    ready.write_text("ok\n")
    print("mmap export complete")


if __name__ == "__main__":
    main()
