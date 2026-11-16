#!/usr/bin/env bash
# Download BCI Competition IV 2a (training files) and 2b from the BNCI Horizon 2020 database.
set -u
cd "$(dirname "$0")"
mkdir -p "archive (1)" bciiv2b
for s in 01 02 03 04 05 06 07 08 09; do
  [ -f "archive (1)/A${s}T.mat" ] || curl -fL --retry 2 -o "archive (1)/A${s}T.mat" \
    "https://bnci-horizon-2020.eu/database/data-sets/001-2014/A${s}T.mat" || echo "failed: A${s}T.mat"
  for split in T E; do
    [ -f "bciiv2b/B${s}${split}.mat" ] || curl -fL --retry 2 -o "bciiv2b/B${s}${split}.mat" \
      "https://bnci-horizon-2020.eu/database/data-sets/004-2014/B${s}${split}.mat" || echo "failed: B${s}${split}.mat"
  done
done
echo "Then run: python 02b_preprocessing_bciiv2a.py && python 12c_within_subject_mi.py && python 12d_bciiv2b.py"
