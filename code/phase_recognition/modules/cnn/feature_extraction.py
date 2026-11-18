import pickle
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule


class FeatureExtraction(LightningModule):
    """
    Stage 1: Train a ResNet50 CNN for surgical phase classification and
    export per-frame CNN features (stem vectors) for Stage 2 (MS-TCN).

    Compatible with any dataset that provides batches of the form:
        train/val:  (img, label, _)
        test:       (img, label, [video_idx, image_path])

    Features are saved to:
        {output_path}/{pickle_export_folder}/{fps:.1f}fps/video_{idx}_{fps:.1f}fps.pkl
    """

    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        # Use self.params to avoid colliding with PL's reserved self.hparams property
        self.params = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.num_tasks = hparams.num_tasks
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.from_numpy(self.dataset.class_weights).float()
        )

        num_classes = hparams.out_features
        self.train_acc_phase = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.val_acc_phase = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_acc_phase = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        self.current_video_idx = self.dataset.df["test"].video_idx.min()
        self.current_stems: list = []
        self.current_phase_labels: list = []
        self.current_p_phases: list = []
        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.test_acc_per_video: dict = {}
        self.pickle_path: Path | None = None

    # ---------------------
    # FORWARD
    # ---------------------

    def forward(self, x):
        stem, phase = self.model.forward(x)
        return stem, phase

    # ---------------------
    # TRAINING
    # ---------------------

    def training_step(self, batch, batch_idx):
        x, y_phase, _ = batch
        _, p_phase = self.forward(x)
        loss = self.ce_loss(p_phase, y_phase)
        self.train_acc_phase(p_phase, y_phase)
        self.log("train_acc_phase", self.train_acc_phase, on_epoch=True, on_step=True)
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_phase, _ = batch
        _, p_phase = self.forward(x)
        loss = self.ce_loss(p_phase, y_phase)
        preds = torch.argmax(p_phase, dim=1)
        self.val_acc_phase.update(preds, y_phase)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        avg_acc = self.val_acc_phase.compute()
        self.log("val_acc_phase", avg_acc, prog_bar=True, on_epoch=True)
        self.val_acc_phase.reset()

    # ---------------------
    # TEST / FEATURE EXPORT
    # ---------------------

    def set_export_pickle_path(self):
        folder = getattr(self.params, "pickle_export_folder", "feature_export")
        self.pickle_path = Path(self.params.output_path) / folder
        self.pickle_path.mkdir(parents=True, exist_ok=True)
        print(f"Feature export path: {self.pickle_path}")

    def save_to_drive(self, vid_index):
        assert self.pickle_path is not None
        fps = self.params.fps_sampling_test
        save_dir = self.pickle_path / f"{fps:.1f}fps"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"video_{vid_index}_{fps:.1f}fps.pkl"

        stems = np.asarray(self.current_stems)
        p_phases = np.asarray(self.current_p_phases)
        labels = np.asarray(self.current_phase_labels)

        with open(save_path, "wb") as f:
            pickle.dump([stems, p_phases, labels], f)

        n = len(labels)
        if n > 0:
            preds = np.argmax(p_phases, axis=1)
            acc = float(np.mean(labels == preds))
        else:
            acc = 0.0
        self.test_acc_per_video[vid_index] = acc
        print(f"video {vid_index:>4d} | {n:>5d} frames | acc: {acc:.4f} | {save_path}")

    def test_step(self, batch, batch_idx):
        x, y_phase, (vid_idx, img_name) = batch
        vid_idx_raw = vid_idx.cpu().numpy()

        with torch.no_grad():
            stem, y_hat = self.forward(x)

        self.test_acc_phase(y_hat, y_phase)

        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(v) for v in vid_idxs]
        index_next = len(vid_idx_raw) if len(vid_idxs) == 1 else indexes[1]

        for i in range(len(vid_idxs)):
            vid_idx_i = vid_idxs[i]
            index = indexes[i]
            if vid_idx_i != self.current_video_idx:
                self.save_to_drive(self.current_video_idx)
                self.current_stems = []
                self.current_phase_labels = []
                self.current_p_phases = []
                index_next = (
                    len(vid_idx_raw) if i + 1 >= len(vid_idxs) else indexes[i + 1]
                )
                self.current_video_idx = vid_idx_i

            y_hat_np = y_hat.cpu().numpy()
            self.current_p_phases.extend(y_hat_np[index:index_next].tolist())
            self.current_stems.extend(
                stem[index:index_next].cpu().detach().numpy().tolist()
            )
            self.current_phase_labels.extend(
                y_phase.cpu().numpy()[index:index_next].tolist()
            )

        if (batch_idx + 1) * self.params.batch_size >= self.len_test_data:
            self.save_to_drive(self.current_video_idx)
            print("Feature extraction complete.")

    def test_epoch_end(self, outputs):
        # flush any frames that didn't trigger the last-batch save (e.g. fast_dev_run)
        if self.current_stems:
            self.save_to_drive(self.current_video_idx)
            self.current_stems = []
            self.current_phase_labels = []
            self.current_p_phases = []

        if self.test_acc_per_video:
            self.log(
                "test_acc_train",
                float(np.mean([self.test_acc_per_video[v] for v in self.dataset.vids_for_training
                               if v in self.test_acc_per_video])),
            )
            self.log(
                "test_acc_val",
                float(np.mean([self.test_acc_per_video[v] for v in self.dataset.vids_for_val
                               if v in self.test_acc_per_video])),
            )
            self.log(
                "test_acc_test",
                float(np.mean([self.test_acc_per_video[v] for v in self.dataset.vids_for_test
                               if v in self.test_acc_per_video])),
            )
        self.log("test_acc", float(self.test_acc_phase.compute()))

    # ---------------------
    # OPTIMIZERS & DATALOADERS
    # ---------------------

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def __dataloader(self, split: str):
        dataset = self.dataset.data[split]
        if self.params.batch_size > self.params.model_specific_batch_size_max:
            print(
                f"Batch size reduced: {self.params.batch_size} → "
                f"{self.params.model_specific_batch_size_max}"
            )
            self.params.batch_size = self.params.model_specific_batch_size_max

        shuffle = split == "train"
        # test must be sequential so per-video accumulation is ordered
        workers = 0 if split == "test" else self.params.num_workers
        print(f"split: {split} | shuffle: {shuffle} | workers: {workers}")
        return DataLoader(
            dataset=dataset,
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        loader = self.__dataloader("train")
        print(f"train loader size: {len(loader.dataset)}")
        return loader

    def val_dataloader(self):
        loader = self.__dataloader("val")
        print(f"val loader size: {len(loader.dataset)}")
        return loader

    def test_dataloader(self):
        self.set_export_pickle_path()
        loader = self.__dataloader("test")
        print(f"test loader size: {len(loader.dataset)} | starting video_idx: {self.current_video_idx}")
        return loader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        group = parser.add_argument_group(title="FeatureExtraction module args")
        group.add_argument("--learning_rate", default=0.0005, type=float)
        group.add_argument("--num_tasks", default=1, type=int, choices=[1, 2])
        group.add_argument("--optimizer_name", default="adam", type=str)
        group.add_argument("--batch_size", default=64, type=int)
        group.add_argument(
            "--pickle_export_folder",
            default="feature_export",
            type=str,
            help="Sub-folder inside output_path where feature PKLs are written",
        )
        return parser
