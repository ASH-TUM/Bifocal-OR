import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from datasets.loo_splits import loo_split
from utils.vitrectomy_constants import CLASS_LABELS


class VitrectomyMultiVideo(Dataset):
    """Holds multiple full video sequences for MS-TCN training (one item = one video)."""

    def __init__(self, pkl_paths, factor_sampling=1):
        self.sequences = []
        for pkl_path in pkl_paths:
            data = pd.read_pickle(pkl_path)
            stems = torch.from_numpy(np.asarray(data[0], dtype=np.float32))
            y_hats = torch.from_numpy(np.asarray(data[1], dtype=np.float32))
            ys = torch.from_numpy(np.asarray(data[2])).long()
            self.sequences.append((
                stems[::factor_sampling],
                y_hats[::factor_sampling],
                ys[::factor_sampling],
            ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class Vitrectomy:
    """
    Dataset wrapper for vitrectomy MS-TCN training (Stage 2).

    Video-level Leave-One-Out split (see datasets/loo_splits.py):
      test_fold      -> held-out test video
      following 2    -> validation videos
      remaining 6    -> training videos
    """

    def __init__(self, hparams):
        self.name = "Vitrectomy"
        self.hparams = hparams
        self.class_labels = CLASS_LABELS
        self.out_features = hparams.out_features
        self.features_per_seconds = hparams.features_per_seconds

        factor = max(1, int(self.features_per_seconds / max(hparams.features_subsampling, 1e-9)))
        hparams.factor_sampling = factor
        print(
            f"Feature subsampling: {self.features_per_seconds}fps -> "
            f"{hparams.features_subsampling}fps (factor: {factor})"
        )

        fps_str = f"{self.features_per_seconds:.1f}fps"
        pkl_dir = Path(hparams.data_root) / fps_str

        all_vids = sorted(
            int(p.stem.split("_")[1]) for p in pkl_dir.glob(f"video_*_{fps_str}.pkl")
        )
        assert all_vids, f"No feature pkls found in {pkl_dir}\nRun Stage 1 first."
        test_fold = int(getattr(hparams, "test_fold", all_vids[-1]))

        pool_str = str(getattr(hparams, "pool", "") or "")
        pool = None
        if pool_str.strip():
            pool = [int(x) for x in pool_str.replace(",", " ").split()]
        val_size = int(getattr(hparams, "val_size", 2))
        train_vids, val_vids, test_vids = loo_split(
            all_vids, test_fold, pool=pool, val_size=val_size
        )
        split_vids = {"train": train_vids, "val": val_vids, "test": test_vids}
        print(
            f"LOO fold: test_fold={test_fold} | pool={pool or 'all'} | "
            f"val_size={val_size} | train={train_vids} val={val_vids} test={test_vids}"
        )

        self.data = {}
        for split, vid_ids in split_vids.items():
            paths = [pkl_dir / f"video_{i}_{fps_str}.pkl" for i in vid_ids]
            self.data[split] = VitrectomyMultiVideo(paths, factor)
            print(f"  {split:5s}: {len(paths)} videos (ids {vid_ids})")

        # Class weights from labeled training frames (ignore unlabeled = -1)
        train_labels = np.concatenate([
            np.asarray(pd.read_pickle(pkl_dir / f"video_{i}_{fps_str}.pkl")[2])
            for i in train_vids
        ])
        train_labels = train_labels[train_labels >= 0]
        counts = np.array(
            [(train_labels == c).sum() for c in range(hparams.out_features)], dtype=float
        )
        freq = counts / counts.sum()
        weights = (np.median(freq) / freq).tolist()
        self.weights = {"train": weights}
        print(f"  class weights: {[round(w, 4) for w in weights]}")

    @staticmethod
    def add_dataset_specific_args(parser):
        vitrectomy_args = parser.add_argument_group(title='vitrectomy TCN dataset args')
        vitrectomy_args.add_argument("--features_per_seconds", default=1.0, type=float)
        vitrectomy_args.add_argument("--features_subsampling", default=1.0, type=float)
        vitrectomy_args.add_argument("--test_fold", default=8, type=int,
                                     help="Held-out video index for LOO cross-validation")
        # Inert pass-through: Stage 2 reads features regardless of which camera
        # produced them, but we tag the run dir with the camera name for clarity.
        vitrectomy_args.add_argument("--camera", default="camera01", type=str,
                                     help="Camera tag for run-name (no functional effect)")
        vitrectomy_args.add_argument("--pool", default="", type=str,
                                     help="Comma- or space-separated list of video indices "
                                          "the LOO rotation operates over. Empty = legacy.")
        vitrectomy_args.add_argument("--val_size", default=2, type=int,
                                     help="Number of cyclic validation videos following "
                                          "test_fold (legacy = 2; Exp A/B = 1).")
        return parser
