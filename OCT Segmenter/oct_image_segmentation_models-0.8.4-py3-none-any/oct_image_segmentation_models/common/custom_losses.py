import focal_loss as fl
import numpy as np
from keras.utils import losses_utils
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from typeguard import typechecked
from typing import Any, Optional, Union


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the
        normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


@typechecked
def focal_loss(
    gamma: float = 2, class_weight: Union[np.ndarray, None] = None, **kwargs
):
    return fl.SparseCategoricalFocalLoss(gamma=gamma, class_weight=class_weight)


@typechecked
def dice_loss_micro(*, is_y_true_sparse: bool, num_classes: int, **kwargs):
    def _dice_loss_micro(y_true, y_pred, smooth=1e-05):
        if is_y_true_sparse:
            y_true = tf.one_hot(
                tf.cast(tf.squeeze(y_true), dtype=tf.int32), num_classes
            )
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2.0 * K.sum(intersection) + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )
        return 1.0 - score

    return _dice_loss_micro


@typechecked
def dice_loss_macro(*, is_y_true_sparse: bool, num_classes: int, **kwargs):
    def _dice_loss_macro(y_true, y_pred, smooth=1e-05):
        if is_y_true_sparse:
            y_true = tf.one_hot(
                tf.cast(tf.squeeze(y_true), dtype=tf.int32), num_classes
            )
        reduce_axis = range(1, len(y_pred.shape) - 1)
        intersection = K.sum(y_true * y_pred, axis=reduce_axis)
        y_true_sum = K.sum(y_true, axis=reduce_axis)
        y_pred_sum = K.sum(y_pred, axis=reduce_axis)

        denominator = y_true_sum + y_pred_sum
        score = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - K.mean(score)

    return _dice_loss_macro


@typechecked
def bce_dice_loss(*, num_classes: int, **kwargs):
    dice_loss_fn = dice_loss_micro(is_y_true_sparse=False, num_classes=num_classes)

    def _bce_dice_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss_fn(y_true, y_pred)

    return _bce_dice_loss


def bce_focal_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + focal_loss(y_true, y_pred)


@typechecked
@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalDiceLoss(fl.SparseCategoricalFocalLoss):
    def __init__(
        self,
        num_classes: int,
        focal_loss_weight: float,
        gamma: float,
        class_weight: Optional[Any] = None,
        from_logits: bool = False,
        dice_macro: bool = True,
        reduction=losses_utils.ReductionV2.AUTO,
        name="sparse_categorical_focal_dice_loss",
        **kwargs,
    ):
        super().__init__(
            gamma, class_weight, from_logits, name=name, reduction=reduction
        )
        self.num_classes = num_classes
        self.focal_loss_weight = focal_loss_weight
        if dice_macro:
            self.dice_loss_fn = dice_loss_macro(
                is_y_true_sparse=True,
                num_classes=num_classes,
            )
        else:
            self.dice_loss_fn = dice_loss_micro(
                is_y_true_sparse=True,
                num_classes=num_classes,
            )

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.

        Returns
        -------
        dict
            This layer's config.
        """
        config = super().get_config()
        config.update(
            num_classes=self.num_classes,
            focal_loss_weight=self.focal_loss_weight,
            dice_loss_fn=self.dice_loss_fn,
        )
        return config

    def call(self, y_true, y_pred):
        focal_loss = super().call(y_true, y_pred)
        # The focal loss is averaged accros the local batch.
        # Keras will handle the division by the number of replicas
        focal_loss = K.sum(focal_loss) / tf.cast(tf.size(y_true), dtype=tf.float32)
        dice_loss = self.dice_loss_fn(y_true, y_pred)

        return (
            self.focal_loss_weight * focal_loss
            + (1 - self.focal_loss_weight) * dice_loss
        )


@typechecked
def focal_dice_loss(
    *,
    num_classes: int,
    gamma: float = 2,
    class_weight: Union[np.ndarray, None] = None,
    focal_loss_weight: float = 0.5,
    dice_macro: bool = True,
    **kwargs,
):
    return SparseCategoricalFocalDiceLoss(
        num_classes,
        focal_loss_weight,
        gamma=gamma,
        class_weight=class_weight,
        dice_macro=dice_macro,
    )


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(
        1.0 - dice_loss_micro(y_true, y_pred)
    )


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    logit_y_pred = K.log(y_pred / (1.0 - y_pred))
    loss = weight * (
        logit_y_pred * (1.0 - y_true)
        + K.log(1.0 + K.exp(-K.abs(logit_y_pred)))
        + K.maximum(-logit_y_pred, 0.0)
    )
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.0
    w, m1, m2 = weight, y_true, y_pred
    intersection = m1 * m2
    score = (2.0 * K.sum(w * intersection) + smooth) / (
        K.sum(w * m1) + K.sum(w * m2) + smooth
    )
    loss = 1.0 - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true,
        pool_size=(50, 50),
        strides=(1, 1),
        padding="same",
        pool_mode="avg",
    )
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5.0 * K.exp(-5.0 * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= w0 / w1
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss_micro(y_true, y_pred)
    return loss


custom_loss_objects = {
    "bce_dice_loss": {
        "function": bce_dice_loss,
        "takes_sparse": False,
    },
    "dice_loss_micro": {
        "function": dice_loss_micro,
        "takes_sparse": False,
    },
    "dice_loss_macro": {
        "function": dice_loss_macro,
        "takes_sparse": False,
    },
    "focal_loss": {
        "function": focal_loss,
        "takes_sparse": True,
    },
    "bce_focal_loss": {
        "function": bce_focal_loss,
        "takes_sparse": False,
    },
    "focal_dice_loss": {
        "function": focal_dice_loss,
        "takes_sparse": True,
    },
}
