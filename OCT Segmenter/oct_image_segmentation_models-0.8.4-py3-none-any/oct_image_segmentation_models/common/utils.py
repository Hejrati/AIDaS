import datetime
import hashlib
import json
import keras
import logging as log
import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from pathlib import Path, PurePosixPath
import tensorflow as tf
from typeguard import typechecked
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Union

from oct_image_segmentation_models.common import custom_losses, custom_metrics


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H_%M_%S")

    return timestamp


@typechecked
def load_model_and_config(
    model_path: Union[Path, PurePosixPath], **kwargs
) -> Tuple[keras.engine.functional.Functional, dict]:
    custom_objects = dict(
        list(custom_losses.custom_loss_objects.items())
        + list(custom_metrics.training_monitor_metric_objects.items())
    )
    mlflow_tracking_uri = kwargs.pop("mlflow_tracking_uri", {})
    mlflow_run_uuid = kwargs.pop("mlflow_run_uuid", {})

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        try:
            loaded_model = mlflow.tensorflow.load_model(
                str(model_path),
                keras_model_kwargs={
                    "custom_objects": custom_objects,
                    "compile": False,
                },
            )
            run = mlflow.get_run(mlflow_run_uuid)
            artifact_uri = run.info.artifact_uri
            model_config = mlflow.artifacts.load_dict(
                artifact_uri + "/model/data/model_config.json"
            )
        except MlflowException as exc:
            if exc.get_http_status_code() == 401:
                log.error(
                    "Looks like the MLFLow client is not authorized to "
                    "log into the MLFlow server. Make sure the "
                    " environment variables 'MLFLOW_TRACKING_USERNAME' "
                    "and 'MLFLOW_TRACKING_PASSWORD' are correct"
                )
            log.exception(msg="An error occurred while loading MLflow model")
            exit(1)
    else:
        loaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        with open(model_path.parent / Path("model_config.json"), "r") as config_file:
            model_config = json.load(config_file)
    return loaded_model, model_config


def convert_maps_uint8(prob_maps):
    prob_maps *= 255
    prob_maps = prob_maps.astype("uint8")

    return prob_maps


def perform_argmax(predictions, bin=True):
    """
    Arguments:
        bin: If True 'categorical_pred' will contain 1 or 0s corresponding to
        the pixel belonging to a particular class or not. If 'False',
        'categorical_pred' will contain the
        prediction probabilites for each pixel per class.

    Returns:
        argmax_pred: A matrix of shape (1, image_width, image_height) that
        contains the predicted classes numbered from 0 to num_classes - 1.
        categorical_pred: A matrix of shape (num_classes, image_width,
        image_height) that contains:
        - If 'bin' == True: 1 on pixels that belong the class and 0 otherwise.
        - If 'bin' == False: Prediction probabilities for each pixel per class.
    """
    if K.image_data_format() == "channels_last":
        pass
    else:
        predictions = np.transpose(predictions, (0, 2, 3, 1))

    num_maps = predictions.shape[3]

    if bin:
        argmax_pred = np.argmax(predictions, axis=3)  # TODO: Refactor line

        categorical_pred = to_categorical(argmax_pred, num_maps)
        categorical_pred = np.transpose(categorical_pred, axes=(0, 3, 1, 2))
    else:
        argmax_pred = np.argmax(predictions, axis=3)
        categorical_pred = np.transpose(predictions, axes=(0, 3, 1, 2))

    return [argmax_pred, categorical_pred]


def convert_predictions_to_maps_semantic(categorical_pred, bg_ilm=True, bg_csi=False):
    """
    #TODO: Document functionality
    """
    num_samples = categorical_pred.shape[0]
    num_maps = categorical_pred.shape[1]
    img_height = categorical_pred.shape[2]
    img_width = categorical_pred.shape[3]

    boundary_maps = np.zeros(
        (num_samples, num_maps - 1, img_height, img_width), dtype="uint8"
    )

    for sample_ind in range(num_samples):
        for map_ind in range(1, num_maps):  # don't care about boundary for top region

            if (map_ind == 1 and bg_ilm is True) or (
                map_ind == num_maps - 1 and bg_csi is True
            ):
                cur_map = categorical_pred[sample_ind, map_ind - 1, :, :]

                grad_map = np.gradient(cur_map, axis=0)

                grad_map = -grad_map

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=0)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = convert_maps_uint8(
                    grad_map
                )
            else:
                cur_map = categorical_pred[sample_ind, map_ind, :, :]

                grad_map = np.gradient(cur_map, axis=0)

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=0)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = convert_maps_uint8(
                    grad_map
                )

    return boundary_maps


@typechecked
def md5(file_path: Path) -> str:
    log.info(f"Calculating md5 of file: {file_path}")
    with open(file_path, "rb") as file_to_check:
        data = file_to_check.read()
        return hashlib.md5(data).hexdigest()
