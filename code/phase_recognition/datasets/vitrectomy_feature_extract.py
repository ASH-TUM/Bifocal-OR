import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

from datasets.loo_splits import loo_split
from utils.vitrectomy_constants import CLASS_LABELS


class VitrectomyImageDataset(Dataset):
    """Simple image dataset from a vitrectomy dataframe slice."""

    def __init__(self, df, transform, add_label_cols=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.add_label_cols = add_label_cols or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row["image_path"]).convert("RGB")
        img_t = self.transform(img)
        label = torch.tensor(int(row["class"]), dtype=torch.long)
        add_label = []
        for col in self.add_label_cols:
            val = row[col]
            # numeric columns → tensor; string columns (e.g. image_path) → pass through
            if isinstance(val, str):
                add_label.append(val)
            else:
                add_label.append(torch.tensor(int(val), dtype=torch.long))
        return img_t.float(), label, add_label


class VitrectomyFeatureExtract:
    """
    Dataset wrapper for vitrectomy CNN feature extraction (Stage 1).

    Video-level Leave-One-Out split (see datasets/loo_splits.py):
      test_fold      -> held-out test video
      following 2    -> validation videos (early stopping)
      remaining 6    -> training videos

    The held-out video never appears in CNN train/val, so the exported
    features are leakage-free for the Stage 2 LOO evaluation.

    With test_extract=True the *test* split still covers ALL videos, because
    Stage 2 needs the CNN features for every video (incl. train/val ones).
    """

    def __init__(self, hparams):
        self.hparams = hparams
        data_root = Path(hparams.data_root)
        camera = getattr(hparams, "camera", "camera01")
        pkl_path = data_root / "dataframes" / f"vitrectomy_1fps_{camera}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Dataframe for {camera} not found at {pkl_path}. "
                f"Build it first with:  "
                f"python vitrectomy/build_dataframe.py --camera {camera}"
            )
        print(f"Loading dataframe: {pkl_path}")

        df = pd.read_pickle(pkl_path)
        all_vids = sorted(df["video_idx"].unique().tolist())
        test_fold = int(getattr(hparams, "test_fold", all_vids[-1]))

        # Optional pool restriction (Exp A: Runs 1-8 only, Run 9 = OOD)
        pool_str = str(getattr(hparams, "pool", "") or "")
        pool = None
        if pool_str.strip():
            pool = [int(x) for x in pool_str.replace(",", " ").split()]
        val_size = int(getattr(hparams, "val_size", 2))
        train_vids, val_vids, test_vids = loo_split(
            all_vids, test_fold, pool=pool, val_size=val_size
        )
        print(
            f"LOO fold: test_fold={test_fold} | pool={pool or 'all'} | "
            f"val_size={val_size} | train={train_vids} val={val_vids} test={test_vids}"
        )

        self.df = {}
        self.df["all"] = df.reset_index(drop=True)
        # Filter unlabeled (class=-1) from train/val to avoid CrossEntropyLoss errors.
        train_slice = df[df["video_idx"].isin(train_vids) & (df["class"] >= 0)]
        self.df["train"] = train_slice.reset_index(drop=True)
        val_slice = df[df["video_idx"].isin(val_vids) & (df["class"] >= 0)]
        self.df["val"] = val_slice.reset_index(drop=True)
        if hparams.test_extract:
            # all videos → full feature export for Stage 2
            self.df["test"] = df.reset_index(drop=True)
        else:
            self.df["test"] = df[df["video_idx"].isin(test_vids)].reset_index(drop=True)

        self.vids_for_training = train_vids
        self.vids_for_val = val_vids
        self.vids_for_test = test_vids

        # Weights: median_class_freq / class_freq, computed from labeled training frames
        num_classes = hparams.out_features
        train_labeled = self.df["train"]
        counts = np.array([(train_labeled["class"] == c).sum() for c in range(num_classes)], dtype=float)
        freq = counts / counts.sum()
        self.class_weights = np.median(freq) / freq
        self.class_labels = CLASS_LABELS

        train_transform = transforms.Compose([
            transforms.Resize((hparams.input_height, hparams.input_width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((hparams.input_height, hparams.input_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data = {}
        self.data["train"] = VitrectomyImageDataset(self.df["train"], train_transform)
        self.data["val"] = VitrectomyImageDataset(self.df["val"], val_transform)
        # test dataset exports video_idx and image_path for per-video pickle saving
        self.data["test"] = VitrectomyImageDataset(
            self.df["test"], val_transform, add_label_cols=["video_idx", "image_path"]
        )

        print(
            f"Vitrectomy feature extract: "
            f"train={len(self.data['train'])} val={len(self.data['val'])} test={len(self.data['test'])}"
        )

    @staticmethod
    def add_dataset_specific_args(parser):
        vitrectomy_args = parser.add_argument_group(title='vitrectomy feature extract args')
        vitrectomy_args.add_argument("--fps_sampling", type=float, default=1.0)
        vitrectomy_args.add_argument("--fps_sampling_test", type=float, default=1.0)
        vitrectomy_args.add_argument("--dataset_mode", default="img", type=str)
        vitrectomy_args.add_argument("--test_extract", action="store_true")
        vitrectomy_args.add_argument("--test_fold", default=8, type=int,
                                     help="Held-out video index for LOO cross-validation")
        vitrectomy_args.add_argument("--camera", default="camera01", type=str,
                                     help="Which camera dataframe to use "
                                          "(loads vitrectomy_1fps_<camera>.pkl)")
        vitrectomy_args.add_argument("--pool", default="", type=str,
                                     help="Comma- or space-separated list of video indices "
                                          "the LOO rotation operates over. Empty = all "
                                          "available videos (legacy). "
                                          "Exp A/B: '0,1,2,3,4,5,6,7' (Runs 1-8).")
        vitrectomy_args.add_argument("--val_size", default=2, type=int,
                                     help="Number of cyclic validation videos following "
                                          "test_fold (legacy = 2; Exp A/B = 1).")
        return parser
