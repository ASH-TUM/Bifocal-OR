import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from utils.configargparse_arguments import build_configargparser
from datetime import datetime

logging.disable(logging.WARNING)


def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = ModelClass(hparams=hparams)
    dataset = DatasetClass(hparams=hparams)
    module = ModuleClass(hparams, model, dataset)

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        filename=hparams.name + '-{epoch}-{' + hparams.early_stopping_metric + ':.2f}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=getattr(hparams, 'early_stopping_patience', 3),
        mode='max')

    fast_dev_run = hparams.fast_dev_run
    if isinstance(fast_dev_run, str):
        fast_dev_run = fast_dev_run.lower() not in ('false', '0', '')
    trainer = Trainer(
        accelerator=getattr(hparams, 'accelerator', 'auto'),
        logger=logger,
        fast_dev_run=fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(module)
    best_path = checkpoint_callback.best_model_path
    print(
        f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} | path: {best_path}"
        f"\nTesting..."
    )
    if best_path:
        trainer.test(ckpt_path=best_path)
    else:
        trainer.test(module)


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)
    # ------------------------
    # PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()

    exp_name = (hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] + "_" + hparams.model.replace(".", "_"))
    cam = getattr(hparams, "camera", None)
    if cam:
        exp_name = f"{cam}_{exp_name}"
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = str(Path(hparams.output_path).absolute() / hparams.name)

    # ------------------------
    # REPRODUCIBILITY
    # ------------------------
    # Per-fold deterministic seed: every LOO fold is reproducible and
    # distinct, so re-running run_loo_cv.sh yields identical results.
    fold = int(getattr(hparams, "test_fold", 0) or 0)
    hparams.fold_seed = hparams.seed + fold
    seed_everything(hparams.fold_seed, workers=True)
    print(f"Seed: {hparams.fold_seed} (base {hparams.seed} + test_fold {fold})")

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')

    argparse_summary(hparams, parser)
    print('Output path: ', hparams.output_path)

    train(hparams, ModuleClass, ModelClass, DatasetClass, tb_logger)
