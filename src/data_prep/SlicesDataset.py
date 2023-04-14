"""
Module for Pytorch dataset representations
"""

from typing import Any, Dict

from torch import tensor
from torch.utils.data import Dataset


class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """

    def __init__(self, data):
        self.data = data
        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Args:
            idx: id of sample

        Returns:
            Dictionary of sample ID and 2 Torch Tensors of dimensions [1, W, H]
        """
        sample_idx, slice_idx = self.slices[idx]
        return {
            "id": idx,
            "image": tensor(self.data[sample_idx]["image"][slice_idx, :, :][None, :]),
            "seg": tensor(self.data[sample_idx]["seg"][slice_idx, :, :][None, :]),
        }

    def __len__(self) -> int:
        """
        This method is called by PyTorch DataLoader class to return number of samples
            in the dataset.
        """
        return len(self.slices)
