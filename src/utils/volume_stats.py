"""
Contains various functions for computing statistics over 3D volumes
"""


import numpy as np


def dice_3d(a: np.array, b: np.array) -> float:
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Args:
        a: 3D array with first volume.
        b: 3D array with second volume.

    Returns:
        Dice Similarity coefficient
    """
    # 1. Sanity checks
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(
            f"Expecting inputs of the same shape, got {a.shape} and {b.shape}"
        )

    # 2. Arrays must be binary
    a, b = (a.astype(bool), b.astype(bool))

    # 3. Compute metric
    intersection = np.sum(a * b)
    union = np.sum(a) + np.sum(b)

    if union == 0:
        return -1

    return 2.0 * float(intersection) / float(union)


def jaccard_3d(a: np.array, b: np.array) -> float:
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Args:
        a: 3D array with first volume.
        b: 3D array with second volume.

    Returns:
        3D Jaccard coefficient.
    """
    # 1. Sanity checks
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(
            f"Expecting inputs of the same shape, got {a.shape} and {b.shape}"
        )

    # 2. Arrays must be binary
    a, b = (a.astype(bool), b.astype(bool))

    # 3. Compute metric
    intersection = np.sum(a * b)
    union = np.sum(a) + np.sum(b)

    if union == 0:
        return -1

    return float(intersection) / float((union - intersection))


def sensitivity(gt: np.array, pred: np.array) -> float:
    """
    This will compute the sensitivity for two 3-dimensional volumes representing ground
        truch and prediction. Volumes are expected to be of the same size. We are
        expecting binary masks - 0's are treated as background and anything else is
        counted as data.

    Sensitivity is TP / (TP + FN)

    Args:
        a: 3D array with first volume.
        b: 3D array with second volume.

    Returns:
        Dice Similarity coefficient
    """
    # Arrays must be binary
    gt, pred = (gt.astype(bool), pred.astype(bool))

    tp = np.sum(gt[gt == pred])
    fn = np.sum(gt[gt != pred])

    if fn + tp == 0:
        return -1

    return tp / (fn + tp)


def specificity(gt: np.array, pred: np.array) -> float:
    """
    This will compute the specificity for two 3-dimensional volumes representing ground
        truch and prediction. Volumes are expected to be of the same size. We are
        expecting binary masks - 0's are treated as background and anything else is
        counted as data.

    Specificity is TN / (TN + FP)

    Args:
        a: 3D array with first volume.
        b: 3D array with second volume.

    Returns:
        Dice Similarity coefficient
    """
    # Arrays must be binary
    gt, pred = (gt.astype(bool), pred.astype(bool))

    tn = np.sum(~gt[gt == pred])
    fp = np.sum(~gt[gt != pred])

    if tn + fp == 0:
        return -1

    return tn / (tn + fp)
