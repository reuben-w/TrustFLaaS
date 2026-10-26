"""
PAMAP2 Physical Activity Monitoring dataset loader, for the sensor-time-series
extension of the Split FL / HealSplit track (per-supervisor request to test
generalization beyond image datasets).

Not available via torchvision -- this is raw per-subject accelerometer/
gyroscope data from the UCI PAMAP2 archive, requiring: download+extract,
column selection, per-activity-segment windowing, subject-holdout train/test
split, and per-channel normalization. None of this reuses src/data.py's
torchvision-oriented DATASET_STATS/get_datasets path -- PAMAP2 is structurally
a different kind of dataset (variable-length raw sensor streams, not
fixed-size labeled images), so it gets its own pipeline, called directly from
main.py behind a `dataset == "pamap2"` branch rather than being squeezed into
the image-dataset interface.

Design decisions, made explicit per the plan agreed with the user:

- Minority class: "rope jumping" (activityID 24), left NATURALLY rare rather
  than artificially subsampled. PAMAP2's 12 core "protocol" activities were
  all performed by (most) subjects, but rope jumping sessions are much
  shorter in duration than e.g. lying/walking/sitting, so it ends up with far
  fewer windows without any subsampling step -- a genuine, not constructed,
  minority-class scenario. minority_keep_fraction from the shared CLI config
  is therefore a no-op for this dataset (documented in config.py).

- Train/test split: SUBJECT-HOLDOUT, not random-window split. Sensor
  signals vary a lot person-to-person (gait, stride length, sensor
  placement), so training and testing on windows from the same subjects
  would let the model partly memorize subject-specific signal quirks rather
  than learning activity-general patterns -- the standard, more rigorous
  protocol in the HAR literature. Subject 109's protocol recording is known
  to be incomplete in the official PAMAP2 release (missing several
  activities) and is excluded entirely, matching common practice.
  TRAIN_SUBJECTS / TEST_SUBJECTS below are the resulting split; adjust here
  if a different split is wanted.

- Channel selection: for each of the three IMU units (hand, chest, ankle),
  keep the 3-axis +-16g accelerometer and 3-axis gyroscope (6 channels x 3
  units = 18 channels total). Dropped: the redundant +-6g accelerometer
  (saturates during vigorous activity, +-16g doesn't), the magnetometer
  (documented in PAMAP2's own readme as unreliable/noisy), the 4-column
  orientation fields (PAMAP2's readme states these are invalid/unused), and
  heart rate (sampled at ~9Hz vs the IMUs' 100Hz -- a separate resampling
  problem not worth the complexity for an 18-channel motion-only baseline).

- Windowing: 100 samples (1s @ 100Hz) per window, 50% overlap (stride=50),
  computed WITHIN each contiguous same-activity segment only (never sliding
  across an activity transition). A window is kept only if a single activity
  label covers >=90% of its samples (guards against edge-of-segment
  windows that straddle two activities); the window's label is that
  majority activity.

- Normalization: per-channel z-score, mean/std fit on TRAIN windows only,
  applied to both train and test -- standard practice, avoids leakage from
  the held-out subjects' own statistics.

Setup required before this module can be used -- PAMAP2 isn't auto-fetchable
the way torchvision datasets are (no small canonical mirror), so:
  1. Download the dataset zip from the UCI archive:
     https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
     (the "Download" button gives a .zip; direct static link, if it still
     resolves: https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip)
  2. Extract it so that <data_root>/PAMAP2_Dataset/Protocol/subject10X.dat
     exist for X in 1..9 (this is the zip's own internal layout -- no
     renaming needed, just extract as-is into --data-root).
  3. build_pamap2_datasets() below reads directly from that Protocol/
     folder. If the folder isn't found, this module raises a clear
     FileNotFoundError with the same instructions rather than trying to
     auto-download (this sandbox can't verify the current UCI URL/redirect
     behavior, so auto-download is deliberately NOT attempted here --
     safer to fail loudly than to silently fetch the wrong thing).
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Column layout of PAMAP2's raw .dat files (54 columns per row) ---
# col 0: timestamp (s)      col 1: activityID      col 2: heart rate (bpm, ~9Hz, else NaN)
# cols 3-19:  hand IMU  (17 cols: temp, acc16(3), acc6(3), gyro(3), mag(3), orientation(4, invalid))
# cols 20-36: chest IMU (same 17-col layout)
# cols 37-53: ankle IMU (same 17-col layout)
_IMU_BLOCK_STARTS = {"hand": 3, "chest": 20, "ankle": 37}
# Within each 17-col IMU block: temp=+0, acc16=+1..+3, acc6=+4..+6, gyro=+7..+9, mag=+10..+12, orient=+13..+16
_ACC16_OFFSET = 1
_GYRO_OFFSET = 7


def _selected_channel_indices() -> List[int]:
    """The 18 raw-file column indices this module keeps: acc16(3) + gyro(3)
    per IMU unit, for hand/chest/ankle in that order."""
    idx = []
    for unit in ("hand", "chest", "ankle"):
        base = _IMU_BLOCK_STARTS[unit]
        idx.extend([base + _ACC16_OFFSET + k for k in range(3)])   # acc16 x,y,z
        idx.extend([base + _GYRO_OFFSET + k for k in range(3)])    # gyro x,y,z
    return idx  # length 18


# The 12 "protocol" activities (performed by ~all subjects). Excludes
# activityID 0 (transient/other) and the optional activities {9,10,11,18,19,20}
# some subjects skipped entirely. Sorted ascending; remapped to 0..11 in this
# order below -- rope jumping (24) lands on index 11.
PROTOCOL_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
ACTIVITY_ID_TO_INDEX = {aid: i for i, aid in enumerate(PROTOCOL_ACTIVITY_IDS)}
ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    24: "rope_jumping",
}
MINORITY_ACTIVITY_ID = 24  # rope jumping
MINORITY_CLASS_INDEX = ACTIVITY_ID_TO_INDEX[MINORITY_ACTIVITY_ID]  # 11

# Subject 109's protocol file is known-incomplete in the official release
# (missing several activities) -- excluded. Remaining 8 subjects split
# 6 train / 2 test.
TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106]
TEST_SUBJECTS = [107, 108]

WINDOW_SIZE = 100   # samples (1.0s @ 100Hz)
WINDOW_STRIDE = 50  # 50% overlap
MIN_PURITY = 0.90   # a window must be >=90% one activity to be kept


def _find_protocol_dir(data_root: str) -> str:
    protocol_dir = os.path.join(data_root, "PAMAP2_Dataset", "Protocol")
    if not os.path.isdir(protocol_dir):
        raise FileNotFoundError(
            f"PAMAP2 raw data not found at {protocol_dir}.\n"
            f"Download the dataset from "
            f"https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring "
            f"and extract it so that {protocol_dir}/subject10X.dat exist "
            f"(extract as-is into --data-root -- no renaming needed)."
        )
    return protocol_dir


def _load_subject_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (activity_ids [N], features [N, 18]) for one subject's raw
    .dat file, with NaNs forward/backward-filled per channel (occasional
    dropouts in the IMU stream; rare but present in the raw data)."""
    raw = np.genfromtxt(path)  # "NaN" tokens -> np.nan automatically
    activity_ids = raw[:, 1].astype(np.int64)
    features = raw[:, _selected_channel_indices()].astype(np.float32)

    # forward-fill then backward-fill each channel independently
    for c in range(features.shape[1]):
        col = features[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[:] = 0.0
            continue
        if nan_mask.any():
            valid_idx = np.flatnonzero(~nan_mask)
            # forward-fill
            fill_idx = np.searchsorted(valid_idx, np.arange(len(col)), side="right") - 1
            fill_idx = np.clip(fill_idx, 0, len(valid_idx) - 1)
            filled = col[valid_idx[fill_idx]]
            # anything before the first valid value: backward-fill from it
            before_first = np.arange(len(col)) < valid_idx[0]
            filled[before_first] = col[valid_idx[0]]
            features[:, c] = filled
    return activity_ids, features


def _segment_and_window(activity_ids: np.ndarray, features: np.ndarray,
                         window_size: int, stride: int, min_purity: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Splits into contiguous same-activity runs, slides a window within
    each run only (never across a transition), keeps windows whose
    majority label covers >=min_purity of samples, restricted to the 12
    protocol activities. Returns (X [M, 18, window_size], y [M])."""
    windows_x, windows_y = [], []
    n = len(activity_ids)

    # find contiguous runs of identical activityID
    change_points = np.flatnonzero(np.diff(activity_ids) != 0) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [n]))

    for start, end in zip(run_starts, run_ends):
        aid = int(activity_ids[start])
        if aid not in ACTIVITY_ID_TO_INDEX:
            continue  # not a protocol activity (transient/optional) -- skip
        run_len = end - start
        for w_start in range(start, end - window_size + 1, stride):
            w_end = w_start + window_size
            seg_labels = activity_ids[w_start:w_end]
            # purity check is almost always trivially 1.0 here since we
            # never cross a run boundary, but kept as a defensive check
            purity = (seg_labels == aid).mean()
            if purity < min_purity:
                continue
            windows_x.append(features[w_start:w_end].T)  # -> [18, window_size]
            windows_y.append(ACTIVITY_ID_TO_INDEX[aid])

    if not windows_x:
        return np.zeros((0, 18, window_size), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(windows_x).astype(np.float32), np.array(windows_y, dtype=np.int64)


class Pamap2WindowDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_pamap2_datasets(
    data_root: str,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    train_subjects: List[int] = None,
    test_subjects: List[int] = None,
) -> Tuple[Dataset, Dataset, int]:
    """Returns (train_dataset, test_dataset, num_classes=12).

    Subject-holdout split (see module docstring). No minority subsampling
    is applied -- rope jumping (class index 11) is naturally rare in the
    resulting window counts; --minority-keep-fraction is a no-op here."""
    train_subjects = train_subjects or TRAIN_SUBJECTS
    test_subjects = test_subjects or TEST_SUBJECTS
    protocol_dir = _find_protocol_dir(data_root)

    def load_subjects(subject_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        all_x, all_y = [], []
        for sid in subject_ids:
            path = os.path.join(protocol_dir, f"subject{sid}.dat")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Expected PAMAP2 subject file not found: {path}")
            activity_ids, features = _load_subject_file(path)
            x, y = _segment_and_window(activity_ids, features, window_size, stride, MIN_PURITY)
            if x.shape[0] > 0:
                all_x.append(x)
                all_y.append(y)
        return np.concatenate(all_x), np.concatenate(all_y)

    train_x, train_y = load_subjects(train_subjects)
    test_x, test_y = load_subjects(test_subjects)

    # per-channel z-score, fit on TRAIN only
    mean = train_x.mean(axis=(0, 2), keepdims=True)   # [1, 18, 1]
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    train_ds = Pamap2WindowDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = Pamap2WindowDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    print(f"[pamap2] train windows={len(train_ds)} (subjects {train_subjects}), "
          f"test windows={len(test_ds)} (subjects {test_subjects})")
    for aid in PROTOCOL_ACTIVITY_IDS:
        ci = ACTIVITY_ID_TO_INDEX[aid]
        n_train = int((train_y == ci).sum())
        n_test = int((test_y == ci).sum())
        tag = "  <-- MINORITY" if aid == MINORITY_ACTIVITY_ID else ""
        print(f"[pamap2]   class {ci:2d} ({ACTIVITY_NAMES[aid]:>16s}): "
              f"train={n_train:5d} test={n_test:5d}{tag}")

    return train_ds, test_ds, len(PROTOCOL_ACTIVITY_IDS)
