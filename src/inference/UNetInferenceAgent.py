"""
Contains class that runs inferencing
"""
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from networks.RecursiveUNet import UNet
from utils.utils import med_reshape


class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """

    def __init__(
        self,
        parameter_file_path: Path = None,
        model: Optional[Any] = None,
        device: str = "cpu",
        patch_size: int = 64,
    ):
        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(
                torch.load(parameter_file_path, map_location=self.device)
            )

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume: np.array) -> np.array:
        """
        Runs inference on a single volume of arbitrary patch size,
            padding it to the conformant size first.

        Args:
            volume: 3D array representing the volume

        Returns:
            3D NumPy array with prediction masks
        """
        return self.single_volume_inference(
            med_reshape(
                volume, new_shape=(volume.shape[0], self.patch_size, self.patch_size)
            )
        )

    def single_volume_inference(self, volume: np.array) -> np.array:
        """
        Runs inference on a single volume of conformant patch size

        Args:
            volume: 3D array representing the volume

        Returns:
            3D NumPy array with prediction masks
        """
        self.model.eval()

        return np.vstack(
            [
                torch.argmax(F.softmax(self.model(vol_slice[None, ...]), dim=1), dim=1)
                for vol_slice in volume
            ]
        )
