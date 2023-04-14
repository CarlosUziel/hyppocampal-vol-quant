"""
Module loads the hippocampus dataset into RAM
"""
from pathlib import Path

import numpy as np
from medpy.io import load

from utils.utils import med_reshape


def load_hyppocampus_data(data_dir: Path, y_shape: int, z_shape: int):
    """
    This function loads our dataset from disk into memory, reshaping output to common
        size.

    Args:
        data_dir: Directory where images and masks reside.
        y_shape: Coronal reshape size.
        z_shape: Axial reshape size.

    Returns:
        Array of dictionaries with data stored in seg and image fields as
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    """
    # 1. Get subdirectories
    images_dir = data_dir.joinpath("images")
    masks_dir = data_dir.joinpath("masks")

    # 2. Load and reshape NIFTI images and corresponding masks
    out = []
    for img_file, mask_file in zip(
        sorted(images_dir.glob("*.nii.gz")), sorted(masks_dir.glob("*.nii.gz"))
    ):
        # 2.1. Ensure image and label belong to the same sample
        assert (
            img_file.stem == mask_file.stem
        ), "Data directory has mismatched sample IDs"

        # 2.2. Load image and mask. We will ignore header since we will not use it
        (image, _), (mask, _) = (load(img_file), load(mask_file))

        # 2.3. Scale image to [0, 1] range.
        image = image / np.max(image)

        # 2.4. Reshape images by extending coronal and axial
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        # cast mask to int as they represent class labels
        mask = med_reshape(mask, new_shape=(mask.shape[0], y_shape, z_shape)).astype(
            int
        )

        out.append({"image": image, "seg": mask, "filename": str(img_file)})

    # Dataset only takes about 300 Mb, so we can afford to keep it all in RAM
    print(
        f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} "
        "slices."
    )
    return np.array(out)
