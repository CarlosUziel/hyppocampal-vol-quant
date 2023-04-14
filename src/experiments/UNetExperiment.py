"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from inference.UNetInferenceAgent import UNetInferenceAgent
from networks.RecursiveUNet import UNet
from utils.config import Config
from utils.utils import log_to_tensorboard
from utils.volume_stats import dice_3d, jaccard_3d, sensitivity


class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet
        (https://arxiv.org/abs/1505.04597).

    The basic life cycle of a UNetExperiment is:
        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """

    def __init__(self, config: Config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # 1. Create output folders
        self.out_dir = config.test_results_dir.joinpath(
            f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        )
        self.out_dir.mkdir(exist_ok=True, parents=True)

        # 2. Create data loaders
        self.train_loader = DataLoader(
            SlicesDataset(dataset[split["train"]]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            SlicesDataset(dataset[split["val"]]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # 3. We will access volumes directly for testing
        self.test_data = dataset[split["test"]]

        # 4. Do we have CUDA available?
        if not torch.cuda.is_available():
            print(
                "WARNING: No CUDA device is found. This may take significantly longer!"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 5. Configure our model and other training implements
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # We are using a standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging
        # to a certain class
        self.loss_function = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

        # Set up Tensorboard. By default it saves data into runs folder.
        tensorboard_path = config.test_results_dir.joinpath("runs")
        tensorboard_path.mkdir(exist_ok=True, parents=True)
        self.tensorboard_train_writer = SummaryWriter(
            log_dir=tensorboard_path, comment="_train"
        )
        self.tensorboard_val_writer = SummaryWriter(
            log_dir=tensorboard_path, comment="_val"
        )

    def train(self):
        """
        This method is executed once per epoch and takes
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # 1. Loop over training minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # 1.1. Get data and target batches
            data = batch["image"].float().to(device=self.device)
            target = batch["seg"].to(device=self.device)

            # 1.2. Get prediction loss
            prediction = self.model(data)

            # We are also getting softmax'd version of prediction to output a
            # probability map so that we can see how the model converges to the
            # solution.
            # We predict class probabilities (one per class) for each voxel in the image.
            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])

            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(
                    f"\nEpoch: {self.epoch} Train loss: {loss},"
                    f" {100*(i+1)/len(self.train_loader):.1f}% complete"
                )

                counter = 100 * self.epoch + 100 * (i / len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter,
                )

            print(".", end="")

        print("\nTraining complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not
        propagate
        """
        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        # 1. Loop through validation minibatches
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # 1.1. Get data and target batches
                data = batch["image"].float().to(device=self.device)
                target = batch["seg"].to(device=self.device)

                # 1.2. Get prediction loss
                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)
                loss = self.loss_function(prediction, target[:, 0, :, :])

                print(f"Batch {i}. Data shape {data.shape}. Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax,
            prediction,
            (self.epoch + 1) * 100,
        )
        print(f"Validation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=""):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset. Note that process and evaluations
            are quite different. Here we are computing a lot more metrics and returning
            a dictionary that could later be persisted as JSON.

        In this method we will be computing metrics that are relevant to the task of 3D
            volume segmentation. Therefore, unlike train and validation methods, we will
            do inferences on full 3D volumes, much like we will be doing it when we
            deploy the model in the clinical environment.
        """
        print("Testing...")
        self.model.eval()

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []

        # 1. Compute metrics for every volume in the test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # 1.1. Compute and report performance metrics
            out_dict["volume_stats"].append(
                {
                    "filename": x["filename"],
                    "dice": dice_3d(x["seg"], pred_label),
                    "jaccard": jaccard_3d(x["seg"], pred_label),
                    "tpr": sensitivity(x["seg"], pred_label),
                    "tnr": sensitivity(x["seg"], pred_label),
                }
            )
            print(
                f"[{100*(i+1)/len(self.test_data):.2f}% complete] "
                f"{x['filename']} Dice {out_dict['volume_stats']['dice']:.4f}."
            )

        # 2. Compute metrics' averages over all test instances
        out_dict["overall"] = {
            "mean_dice": np.mean([e["dice"] for e in out_dict["volume_stats"]]),
            "mean_jaccard": np.mean([e["jaccard"] for e in out_dict["volume_stats"]]),
            "mean_tpr": np.mean([e["tpr"] for e in out_dict["volume_stats"]]),
            "mean_tnr": np.mean([e["tnr"] for e in out_dict["volume_stats"]]),
        }

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(
            "Run complete. Total time:"
            f" {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}"
        )
