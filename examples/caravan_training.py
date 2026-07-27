# Import necessary packages
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import xarray as xr

from hy2dl.datasetzoo import get_dataset
from hy2dl.evaluation import calculate_metrics, get_tester
from hy2dl.modelzoo import get_model
from hy2dl.training.basetrainer import BaseTrainer
from hy2dl.utils.config import Config

os.chdir(sys.path[0])  # Change working directory to the script's location
base_dir = Path.cwd().resolve()


if __name__ == "__main__":
    # Read experiment settings
    path_experiment_settings = "../examples/configs/caravan.yml"
    config = Config(path_experiment_settings, base_dir=base_dir)
    config.init_experiment()

    # Cache the processed training/validation datasets to disk (keyed on the deterministic experiment folder), so
    # that a resumed job (see below) does not have to repeat the ~20 min dataset-processing step from scratch.
    config.path_dataset_training = config.path_save_folder / "dataset_training.zarr"
    config.path_dataset_validation = config.path_save_folder / "dataset_validation.zarr"

    config.dump()

    Dataset = get_dataset(config)
    Tester = get_tester(config)

    # Create training dataset
    training_dataset = Dataset(cfg=config, time_period="training")
    training_dataset.setup_dataset()
    # Initialize training object
    trainer = BaseTrainer(cfg=config, training_dataset=training_dataset)

    validation_dataset = Dataset(cfg=config, time_period="validation")
    validation_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")
    tester_validation = Tester(cfg=config, evaluation_dataset=validation_dataset)

    # ------------------------------------------------------------------
    # Resume support: this script is meant to be re-submitted as a chain of SLURM jobs (each capped by the
    # cluster's max walltime). On each (re)start, pick up from the last epoch that was actually checkpointed rather
    # than always starting at epoch 1. Note: this reconstructs a fresh Adam optimizer at the resumed learning rate
    # (momentum/variance state is not persisted) and only recomputes the learning rate via the custom epoch->lr
    # schedule used in caravan.yml; a StepLR-based config would additionally need its scheduler state fast-forwarded.
    # ------------------------------------------------------------------
    model_dir = config.path_save_folder / "model"
    completed_epochs = sorted(int(p.name.rsplit("_", 1)[1]) for p in model_dir.glob("model_epoch_*"))
    start_epoch = 1
    if completed_epochs:
        last_epoch = completed_epochs[-1]
        trainer.model.load_state_dict(
            torch.load(model_dir / f"model_epoch_{last_epoch}", map_location=config.device)
        )
        trainer.optimizer.update_optimizer_lr(epoch=last_epoch + 1)
        start_epoch = last_epoch + 1
        config.logger.info(f"Resuming training from epoch {start_epoch} (found checkpoint for epoch {last_epoch}).")

    if start_epoch > config.epochs:
        config.logger.info(f"All {config.epochs} epochs already completed. Skipping training loop.")
    else:
        # Training report structure
        validation_headers = "".join([f"{m:^10}|" for m in config.validation_metric])
        config.logger.info("Training model".center(60, "-"))
        config.logger.info(
            f"{'':^16}|{'Training':^21}|{'Validation':^{(11 * len(config.validation_metric)) + 10}}|"
        )
        config.logger.info(f"{'Epoch':^5}|{'LR':^10}|{'Loss':^10}|{'Time':^10}|{validation_headers}{'Time':^10}|")

        # Loop through the remaining epochs
        total_time = time.time()
        for epoch in range(start_epoch, config.epochs + 1):
            trainer.train_model(epoch=epoch)  # Training
            tester_validation.validate_model(model=trainer.model, epoch=epoch)  # Validation
            config.logger.info(trainer.report + tester_validation.validation_report)  # report

        config.logger.info(f"Total training time: {datetime.timedelta(seconds=int(time.time() - total_time))}\n")
        shutil.rmtree(tester_validation.path_zarr, ignore_errors=True)  # delete validation results

    # Reconstruct the model from the last saved epoch and evaluate it on the testing period
    model = get_model(config).to(config.device)
    model.load_state_dict(
        torch.load(config.path_save_folder / "model" / f"model_epoch_{config.epochs}", map_location=config.device)
    )

    testing_dataset = Dataset(cfg=config, time_period="testing")
    testing_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")
    tester_testing = Tester(cfg=config, evaluation_dataset=testing_dataset)

    config.logger.info("Testing model...")
    testing_time = time.time()
    tester_testing.evaluate_model(model=model)
    config.logger.info("Testing completed.")
    config.logger.info(f"Total testing time: {datetime.timedelta(seconds=int(time.time() - testing_time))}\n")

    test_results = xr.open_zarr(tester_testing.path_zarr)
    testing_metrics = calculate_metrics(ds_results=test_results, metric_name=config.testing_metrics)
    testing_metrics.to_zarr(config.path_save_folder / "testing_metrics.zarr", mode="w")
