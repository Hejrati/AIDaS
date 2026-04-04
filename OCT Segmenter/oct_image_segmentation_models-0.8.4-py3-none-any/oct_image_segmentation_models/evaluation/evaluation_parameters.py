from __future__ import annotations

import logging as log
from pathlib import Path
from typeguard import typechecked
from typing import List, Optional

from oct_image_segmentation_models.common import EVALUATION_METRICS, utils


@typechecked
class EvaluationSaveParams:
    def __init__(
        self,
        predicted_labels: bool = True,
        categorical_pred: bool = False,
        png_images: bool = True,
        boundary_maps: bool = True,
    ) -> None:
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.png_images = png_images
        self.boundary_maps = boundary_maps


@typechecked
class EvaluationParameters:
    """Parameters for evaluation of trained network.
    _________

    model_path: filename of Keras model to be loaded
    _________

    dataset: Dataset (see 'Dataset' class)
    _________

    save_params: parameters using to determine what to save from the evaluation
    _________
    """

    def __init__(
        self,
        model_path: Path,
        mlflow_tracking_uri: Optional[str],
        mlflow_run_uuid: Optional[str],
        test_dataset_path: Path,
        save_foldername: Path,
        save_params: EvaluationSaveParams,
        graph_search: bool,
        metrics: List[str],
        gsgrad=1,
        dice_errors: bool = True,
        binarize: bool = True,
        bg_ilm: bool = True,
        bg_csi: bool = False,
    ):
        self.model_path = model_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_run_uuid = mlflow_run_uuid
        self.test_dataset_path = test_dataset_path
        self.binarize = binarize

        self.save_params = save_params
        self.graph_search = graph_search
        if not set(metrics).issubset(EVALUATION_METRICS):
            log.error(
                "Some of the provided metrics are invalid. "
                f"Provided metrics: {metrics}."
            )
            exit(1)

        self.metrics = metrics
        self.gsgrad = gsgrad
        self.dice_errors = dice_errors

        self.bg_ilm = bg_ilm
        self.bg_csi = bg_csi

        self.save_foldername = save_foldername
        self.loaded_model, self.model_config = utils.load_model_and_config(
            model_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_uuid=mlflow_run_uuid,
        )
        self.num_classes = self.loaded_model.output.shape[-1]
