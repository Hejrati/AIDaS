import numpy as np
from surface_distance import (
    compute_surface_distances,
    compute_average_surface_distance,
    compute_robust_hausdorff,
)
import tensorflow as tf
from tensorflow.keras import backend as K
from typeguard import typechecked
from typing import Tuple

from oct_image_segmentation_models.common import (
    TRAINING_MONITOR_METRIC_DICE_MACRO,
    TRAINING_MONITOR_METRIC_DICE_MICRO,
)


@typechecked
def dice_coef_micro(is_y_true_sparse: bool, num_classes: int):
    def _dice_coef_micro(y_true, y_pred):
        """
        c is number of classes
        :param y_pred: b x X x Y( x Z...) x c network output, must sum to 1
        over c channel (such as after softmax)
        :param y_true:
        b x X x Y( x Z...) x c one hot encoding of ground truth
        if is_y_true_sparse == False
        or
        y_true: b x X x Y( x Z...) if is_y_true_sparse == True
        """
        if is_y_true_sparse:
            y_true = tf.one_hot(
                tf.cast(tf.squeeze(y_true), dtype=tf.int32), num_classes
            )
        y_true_f = K.flatten(y_true)
        y_pred = K.cast(y_pred, "float32")
        y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), "float32")
        intersection = y_true_f * y_pred_f
        score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
        return score

    # Change the __name__ attr. so that Keras can find the function when
    # invoking the ModelCheckpoint callback
    _dice_coef_micro.__name__ = "dice_coef_micro"
    return _dice_coef_micro


@typechecked
def dice_coef_macro(is_y_true_sparse: bool, num_classes: int):
    def _dice_coef_macro(y_true, y_pred, eps=1e-05):
        """
        c is number of classes
        :param y_pred: b x X x Y( x Z...) x c network output, must sum to 1
        over c channel (such as after softmax)
        :param y_true:
        b x X x Y( x Z...) x c one hot encoding of ground truth
        if is_y_true_sparse == False
        or
        y_true: b x X x Y( x Z...) if is_y_true_sparse == True
        """
        if is_y_true_sparse:
            y_true = tf.one_hot(
                tf.cast(tf.squeeze(y_true), dtype=tf.int32), num_classes
            )
        y_pred = K.cast(K.greater(y_pred, 0.5), "float32")
        reduce_axis = range(1, len(y_pred.shape) - 1)
        intersection = K.sum(y_true * y_pred, axis=reduce_axis)
        y_true_s = K.sum(y_true, axis=reduce_axis)
        y_pred_s = K.sum(y_pred, axis=reduce_axis)
        denominator = y_true_s + y_pred_s
        score = (2.0 * intersection + eps) / (denominator + eps)
        return K.mean(score)

    # Change the __name__ attr. so that Keras can find the function when
    # invoking the ModelCheckpoint callback
    _dice_coef_macro.__name__ = "dice_coef_macro"
    return _dice_coef_macro


training_monitor_metric_objects = {
    TRAINING_MONITOR_METRIC_DICE_MACRO: dice_coef_macro,
    TRAINING_MONITOR_METRIC_DICE_MICRO: dice_coef_micro,
}


def soft_dice_class(y_true, y_pred, eps=1e-5):
    """
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over
    c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    """
    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    class_dices = ((2.0 * intersect) + eps) / (denom + eps)
    return class_dices


def average_surface_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing: Tuple[float],
) -> dict:
    surface_distances = compute_surface_distances(y_true, y_pred, spacing)
    return compute_average_surface_distance(surface_distances)


def hausdorff_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing: Tuple[float],
    percent: float,
) -> float:
    surface_distances = compute_surface_distances(y_true, y_pred, spacing)
    return compute_robust_hausdorff(surface_distances, percent)
