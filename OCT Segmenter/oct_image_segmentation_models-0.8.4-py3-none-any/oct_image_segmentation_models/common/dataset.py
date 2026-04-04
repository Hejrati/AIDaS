from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional
from typeguard import typechecked


@typechecked
class Dataset:
    """
    'images' with shape: (number of images, width, height, channels)
    (dtype = 'uint8')

    'images_masks' with shape: (number of images, width, height, channels)
    (dtype = 'uint8')

    'images_names' with shape: (number of images,)
    (dtype = 'S' - fixed length strings)
    """

    def __init__(
        self,
        images: np.ndarray,
        image_masks: Optional[np.ndarray],
        image_names: List[Path],
        image_output_dirs: List[Path],
    ):
        self.images = images
        self.image_masks = image_masks
        self.image_names = image_names
        self.image_output_dirs = image_output_dirs
