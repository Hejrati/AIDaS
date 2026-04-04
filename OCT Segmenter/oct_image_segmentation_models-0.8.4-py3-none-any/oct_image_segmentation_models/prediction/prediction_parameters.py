from __future__ import annotations

from pathlib import Path, PurePosixPath
from typeguard import typechecked
from typing import Union

from oct_image_segmentation_models.common import utils
from oct_image_segmentation_models.common.dataset import Dataset


@typechecked
class PredictionSaveParams:
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
class PredictionParams:
    def __init__(
        self,
        model_path: Union[Path, PurePosixPath],
        mlflow_tracking_uri: Union[str, None],
        mlflow_run_uuid: Union[str, None],
        dataset: Dataset,
        config_output_dir: Path,
        save_params: PredictionSaveParams,
        graph_search: bool = False,
        trim_maps: bool = False,
        trim_ref_ind: int = 0,
        trim_window: tuple = (0, 0),
        col_error_range: tuple = None,
    ) -> None:
        self.model_path = model_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_run_uuid = mlflow_run_uuid
        self.dataset = dataset

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.loaded_model, self.model_config = utils.load_model_and_config(
            model_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_uuid=mlflow_run_uuid,
        )
        self.num_classes = self.loaded_model.output.shape[-1]
        self.config_output_dir = config_output_dir
        self.save_params = save_params
        self.graph_search = graph_search
        self.trim_maps = trim_maps
        self.trim_ref_ind = trim_ref_ind
        self.trim_window = trim_window

        self.col_error_range = col_error_range
        if col_error_range is None:
            self.col_error_range = range(dataset.images[0].shape[1])  # image_width
