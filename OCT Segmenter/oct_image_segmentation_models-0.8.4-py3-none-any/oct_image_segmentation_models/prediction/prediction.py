from __future__ import annotations

import time

import h5py
import logging as log
from matplotlib import cm
import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import List, Union

from tensorflow.keras.utils import to_categorical

from oct_image_segmentation_models.common import (
    dataset_construction as datacon,
    plotting,
    utils,
)
from oct_image_segmentation_models.min_path_processing import graph_search
from oct_image_segmentation_models.models import get_model_class
from oct_image_segmentation_models.prediction.prediction_parameters import (
    PredictionParams,
)


@typechecked
class PredictionOutput:
    def __init__(
        self,
        image: np.ndarray,
        image_name: Path,
        image_output_dir: Path,
        predicted_labels: np.ndarray,
        categorical_pred: np.ndarray,
        boundary_maps: np.ndarray,
        gs_pred_segs: Union[np.ndarray, None],
    ) -> None:
        self.image = image
        self.image_name = image_name
        self.image_output_dir = image_output_dir
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.boundary_maps = boundary_maps
        self.gs_pred_segs = gs_pred_segs


def predict(predict_params: PredictionParams) -> List[PredictionOutput]:
    dataset = predict_params.dataset
    predict_images = dataset.images
    predict_image_names = dataset.image_names
    predict_image_output_dirs = dataset.image_output_dirs

    save_predict_config_file(predict_params)

    prediction_outputs = []

    # Build Model Container Class from 'loaded_model' name to preprocess
    # the images
    try:
        model_class = get_model_class(predict_params.loaded_model.name)
    except ValueError as e:
        log.error(e)
        exit(1)

    model_container = model_class(**predict_params.model_config)
    model_preprocessing_input_fn = model_container.get_preprocess_input_fn()

    # pass images to network one at a time
    for i, (predict_image, image_name, image_output_dir) in enumerate(
        zip(predict_images, predict_image_names, predict_image_output_dirs)
    ):
        log.info(f"Inferring image {i}: {image_name}")
        start_predict_time = time.time()
        predicted_probs = predict_params.loaded_model.predict(
            np.expand_dims(
                model_preprocessing_input_fn(predict_image), axis=0
            ),  # keras works with batches
            verbose=2,
            batch_size=1,
        )
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        log.info("Converting predictions to boundary maps...")

        start_convert_time = time.time()

        """
        predicted_labels: A matrix of shape (1, image_width, image_height) that
        contains the predicted classes numbered from 0 to num_classes - 1.

        categorical_pred: A matrix of shape (num_classes, image_width,
        image_height) that contains:
            - If 'bin' == True: 1 on pixels that belong the class and 0
            otherwise.
            - If 'bin' == False: Prediction probabilities for each pixel per
            class.
        """
        [predicted_labels, categorical_pred] = utils.perform_argmax(
            predicted_probs, bin=True
        )

        boundary_maps = utils.convert_predictions_to_maps_semantic(
            np.array(categorical_pred),
            bg_ilm=True,
            bg_csi=False,
        )

        end_convert_time = time.time()
        convert_time = end_convert_time - start_convert_time

        predicted_labels = np.squeeze(predicted_labels)
        categorical_pred = np.squeeze(categorical_pred)
        boundary_maps = np.squeeze(boundary_maps)

        # save data to files
        save_image_prediction_results(
            predict_params,
            predict_image,
            image_name,
            predicted_labels,
            categorical_pred,
            boundary_maps,
            predict_time,
            convert_time,
            image_output_dir,
        )

        if predict_params.graph_search:
            # Segment probability maps using graph search
            log.info("Running graph search, segmenting boundary maps...")
            num_classes = len(categorical_pred)
            predict_image_t = np.transpose(predict_image, axes=[1, 0, 2])
            boundary_maps_t = np.transpose(boundary_maps, axes=[0, 2, 1])
            graph_structure = graph_search.create_graph_structure(predict_image_t.shape)

            start_graph_time = time.time()
            gs_pred_segs, _, _ = graph_search.segment_maps(
                boundary_maps_t, None, graph_structure
            )

            reconstructed_maps = datacon.create_area_mask(
                predict_image_t.shape, gs_pred_segs
            )

            reconstructed_maps = to_categorical(
                reconstructed_maps, num_classes=num_classes
            )
            reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

            [gs_prediction_label, reconstructed_maps] = utils.perform_argmax(
                reconstructed_maps
            )

            gs_prediction_label = np.transpose(np.squeeze(gs_prediction_label))

            end_graph_time = time.time()
            graph_time = end_graph_time - start_graph_time

            save_graph_based_prediction_results(
                predict_params,
                predict_image,
                image_name,
                gs_prediction_label,
                gs_pred_segs,
                graph_time,
                image_output_dir,
            )
        else:
            gs_pred_segs = None

        prediction_output = PredictionOutput(
            image=predict_image,
            image_name=image_name,
            image_output_dir=image_output_dir,
            predicted_labels=predicted_labels,
            categorical_pred=categorical_pred,
            boundary_maps=boundary_maps,
            gs_pred_segs=gs_pred_segs,
        )

        prediction_outputs.append(prediction_output)
        log.info(f"DONE processing image number {i}: {image_name}")

    return prediction_outputs


@typechecked
def save_predict_config_file(predict_params: PredictionParams):
    config_file = h5py.File(
        predict_params.config_output_dir / Path("prediction_params.hdf5"), "w"
    )

    config_file.attrs["model_filename"] = np.array(
        predict_params.model_path, dtype="S1000"
    )
    config_file.attrs["error_col_inc_range"] = np.array(
        (predict_params.col_error_range[0], predict_params.col_error_range[-1])
    )
    config_file.close()


@typechecked
def save_image_prediction_results(
    pred_params: PredictionParams,
    predict_image: np.ndarray,
    image_name: Path,
    predicted_labels: np.ndarray,
    categorical_pred: np.ndarray,
    boundary_maps: np.ndarray,
    predict_time: float,
    convert_time: float,
    output_dir: Path,
):
    hdf5_file = h5py.File(output_dir / Path("prediction_info.hdf5"), "w")

    if pred_params.save_params.categorical_pred is True:
        hdf5_file.create_dataset(
            "categorical_pred", data=categorical_pred, dtype="uint8"
        )

        if pred_params.save_params.png_images is True:
            for map_ind in range(len(categorical_pred)):
                plotting.save_image_plot(
                    categorical_pred[map_ind],
                    output_dir / Path("categorical_pred_" + map_ind + ".png"),
                    cmap=cm.Blues,
                )

    np.savetxt(
        output_dir / Path("segmentation_map.csv"),
        predicted_labels,
        fmt="%d",
        delimiter=",",
    )

    if pred_params.save_params.predicted_labels is True:
        hdf5_file.create_dataset(
            "predicted_labels", data=predicted_labels, dtype="uint8"
        )

        if pred_params.save_params.png_images is True:
            plotting.save_image_plot(
                predicted_labels,
                output_dir / Path("segmentation_map.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(categorical_pred)
                ),
            )

    if pred_params.save_params.boundary_maps is True:
        hdf5_file.create_dataset("boundary_maps", data=boundary_maps, dtype="uint8")

    hdf5_file.create_dataset("raw_image", data=predict_image, dtype="uint8")

    plotting.save_image_plot(
        predict_image,
        output_dir / Path("raw_image.png"),
        cmap=None if predict_image.shape[2] == 3 else cm.gray,
        vmin=0,
        vmax=255,
    )

    hdf5_file.attrs["model_filename"] = np.array(pred_params.model_path, dtype="S1000")
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(utils.get_timestamp(), dtype="S1000")
    hdf5_file.attrs["predict_time"] = np.array(predict_time)
    hdf5_file.attrs["convert_time"] = convert_time
    hdf5_file.close()


@typechecked
def save_graph_based_prediction_results(
    predict_params: PredictionParams,
    predict_image: np.ndarray,
    image_name: Path,
    gs_prediction_label: np.ndarray,
    gs_pred_segs: np.ndarray,
    graph_time: float,
    output_dir: Path,
):
    num_classes = gs_pred_segs.shape[0] + 1
    # Save graph search based prediction results
    hdf5_file = h5py.File(output_dir / Path("graph_search_prediction_info.hdf5"), "w")

    np.savetxt(
        output_dir / Path("gs_boundaries.csv"),
        gs_pred_segs,
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        output_dir / Path("gs_segmentation_map.csv"),
        gs_prediction_label,
        fmt="%d",
        delimiter=",",
    )

    hdf5_file.create_dataset("gs_pred_segs", data=gs_pred_segs, dtype="uint16")

    hdf5_file.create_dataset(
        "gs_predicted_labels", data=gs_prediction_label, dtype="uint8"
    )

    plotting.save_image_plot(
        gs_prediction_label,
        output_dir / Path("gs_predicted_segmentation_map.png"),
        cmap=plotting.colors.ListedColormap(plotting.region_colours, N=num_classes),
    )

    plotting.save_segmentation_plot(
        predict_image,
        cm.gray,
        output_dir / Path("gs_predicted_boundaries_ovelay_plot.png"),
        gs_pred_segs,
        predictions=None,
        column_range=predict_params.col_error_range,
    )

    hdf5_file.attrs["model_filename"] = np.array(
        predict_params.model_path, dtype="S1000"
    )
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(utils.get_timestamp(), dtype="S1000")
    hdf5_file.attrs["graph_time"] = np.array(graph_time)

    hdf5_file.close()
