# Builds the training/validation/testing zarr datasets for the full Caravan basin list, using the cluster's
# already-complete local Caravan_all (no download/extraction needed, no camelsaus gap - see
# notebooks/LSTM_CARAVAN_Colab.ipynb's intro cell for why Colab can't do this itself). Output goes to a
# dedicated folder, separate from results/LSTM_CARAVAN_seed_1 (the actual training job's folder, which may
# still be pending/running - see train_caravan_a100.sbatch), so the two never race on the same files.
#
# Meant to be run once; the resulting three dataset_*.zarr folders get tarred and uploaded to Google Drive
# (see build_caravan_datasets.sbatch), landing at the exact path/filenames the Colab notebook's own
# Drive-cache check (cell 10) already looks for - so the notebook picks them up with no changes needed.
import os
import sys
from pathlib import Path

import yaml

from hy2dl.datasetzoo import get_dataset
from hy2dl.utils.config import Config

os.chdir(sys.path[0])  # Change working directory to the script's location
base_dir = Path.cwd().resolve()

# Everything below must stay inside this guard: dask spawns worker processes that re-import this module in a
# fresh interpreter, and without the guard they would re-run the whole script (including dataset creation)
# recursively instead of just executing as workers.
if __name__ == "__main__":
    with open("configs/caravan.yml") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict["path_save_folder"] = "../results/Caravan_datasets_for_drive"
    cfg_dict["device"] = "cpu"  # this script only builds datasets - no model/GPU involved, runs on the cpu partition
    cfg_dict["dataset_in_ram"] = False  # dataset_in_ram defaults to True, which would skip writing zarr to disk
    # entirely - the whole point here is to produce actual zarr files to tar and upload to Drive.

    config = Config(cfg_dict, base_dir=base_dir)
    config.init_experiment()

    config.path_dataset_training = config.path_save_folder / "dataset_training.zarr"
    config.path_dataset_validation = config.path_save_folder / "dataset_validation.zarr"
    config.path_dataset_testing = config.path_save_folder / "dataset_testing.zarr"

    config.dump()

    Dataset = get_dataset(config)

    print("Creating training dataset...")
    training_dataset = Dataset(cfg=config, time_period="training")
    training_dataset.setup_dataset()

    print("Creating validation dataset...")
    validation_dataset = Dataset(cfg=config, time_period="validation")
    validation_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")

    print("Creating testing dataset...")
    testing_dataset = Dataset(cfg=config, time_period="testing")
    testing_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")

    print(f"Done. Zarr datasets created at {config.path_save_folder}")
