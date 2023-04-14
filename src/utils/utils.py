"""
Various utility methods in this module
"""
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")


def mpl_image_grid(images):
    """
    Create an image grid from an array of images. Show up to 16 images in one figure

    Arguments:
        image {Torch tensor} -- NxWxH array of images

    Returns:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16)  # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2 * rows, 2 * cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [
                [
                    [
                        (1 - vol[0, x, y]) * vol[1, x, y],
                        (1 - vol[0, x, y]) * vol[2, x, y],
                        0,
                    ]
                    for y in range(vol.shape[2])
                ]
                for x in range(vol.shape[1])
            ]
            plt.imshow(img)
        else:  # plotting only 1st channel
            plt.imshow((images[i, 0] * 255).int(), cmap="gray")

    return figure


def log_to_tensorboard(
    writer, loss, data, target, prediction_softmax, prediction, counter
):
    """Logs data to Tensorboard

    Arguments:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss", loss, counter)
    writer.add_figure(
        "Image Data", mpl_image_grid(data.float().cpu()), global_step=counter
    )
    writer.add_figure("Mask", mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure(
        "Probability map", mpl_image_grid(prediction_softmax.cpu()), global_step=counter
    )
    writer.add_figure(
        "Prediction",
        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)),
        global_step=counter,
    )


def save_numpy_as_image(arr: np.array, path: Path) -> None:
    """
    This saves image (2D array) as a file using matplotlib

    Arguments:
        arr: 2D array of pixels
        path: path to file
    """
    plt.imshow(arr, cmap="gray")  # Needs to be in row,col order
    plt.savefig(path)


def med_reshape(image: np.array, new_shape: Tuple[int, int, int]) -> np.array:
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Args:
        image: 3D array of pixel data.
        new_shape: Expected output shape.

    Returns:
        3D array of desired shape, padded with zeroes
    """
    reshaped_image = np.zeros(new_shape)
    x_dim, y_dim, z_dim = image.shape
    reshaped_image[0:x_dim, 0:y_dim, 0:z_dim] = image

    return reshaped_image
