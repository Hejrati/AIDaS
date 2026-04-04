from __future__ import annotations

import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import List, Tuple


def load_training_data(hdf5_data_file):
    train_images = hdf5_data_file["train_images"][:]
    train_labels = hdf5_data_file["train_labels"][:]

    return train_images, train_labels


def load_validation_data(hdf5_data_file):
    val_images = hdf5_data_file["val_images"][:]
    val_labels = hdf5_data_file["val_labels"][:]

    return val_images, val_labels


@typechecked
def load_testing_data(
    hdf5_data_file,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    test_images = hdf5_data_file["test_images"][:]
    test_labels = hdf5_data_file["test_labels"][:]
    test_image_paths = [
        Path(str(x, "ascii")) for x in hdf5_data_file.get("test_images_source")
    ]

    return test_images, test_labels, test_image_paths
