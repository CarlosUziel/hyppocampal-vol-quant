"""
This file contains code that will kick off training and testing processes
"""
import json
import os

from sklearn.model_selection import train_test_split

from data_prep.utils import load_hyppocampus_data
from experiments.UNetExperiment import UNetExperiment
from utils.config import Config

if __name__ == "__main__":
    # 1. Get configuration
    config = Config()

    # 2. Load data
    print("Loading data...")
    data = load_hyppocampus_data(
        config.root_dir, y_shape=config.patch_size, z_shape=config.patch_size
    )

    # 3. Train/test/val split
    ids = range(len(data))
    train_ids, test_ids = train_test_split(ids, test_size=0.2)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2)
    split_indices = {"train": train_ids, "val": val_ids, "test": test_ids}

    # 4. Set up and run experiment
    exp = UNetExperiment(config, split_indices, data)

    # 4.1. Run training
    exp.run()

    # 4.2. Prepare and run testing
    results_json = exp.run_test()

    results_json["config"] = vars(config)

    with open(os.path.join(exp.out_dir, "results.json"), "w") as out_file:
        json.dump(results_json, out_file, indent=2, separators=(",", ": "))
