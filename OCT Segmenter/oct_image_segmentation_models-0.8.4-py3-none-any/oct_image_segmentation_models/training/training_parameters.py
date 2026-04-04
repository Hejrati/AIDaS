import logging as log
from pathlib import Path
from typeguard import typechecked
from typing import List, Tuple, Union

from oct_image_segmentation_models.common import AUG_MODES
from oct_image_segmentation_models.common import augmentation as aug


@typechecked
class TrainingParams:
    """
    Parameters for training a network:

        aug_fn_args: list of dictions containing augmentation functions and
        argument pairs. Each element in the list is a dictionary with the
        following structure:
        {
            "name": <augmentation_fn_name>,
            "arguments": {
                "<argument_name>": "<argument_value>" ,
                ...
            }
        }
        _________

        aug_mode: mode to use for augmentation

        none: no augmentations -> will just use what is in the images and
        labels arrays as is
        one: for each image, one augmentation will be picked from the list of
        possible augmentation functions chosen based on probabilities in
        aug_probs.
        all: for each image, all augmentations will be performed creating a
        new separate image for each

        note that for patch mode: augs are applied to the full size images
        before being broken into patches
        _________

        aug_probs: probabilities used for selecting augmentations in 'one'
        mode. Should be values between 0 and 1 which add to 1.
        _________

        aug_val: boolean used to apply the same augmentation policy to the
        validation dataset.
        _________
    """

    def __init__(
        self,
        model_architecture: Union[str, None],
        training_dataset_path: Path,
        initial_model: Union[Path, None],
        results_location: Path,
        opt_con,
        loss: str,
        metric: str,
        epochs: int,
        batch_size: int,
        model_hyperparameters: dict = {},
        opt_params: dict = {},
        loss_fn_kwargs: dict = {},
        augmentations: List[dict] = [],
        aug_mode: str = "none",
        aug_probs: Tuple = (),
        aug_fly: bool = False,
        aug_val: bool = True,
        shuffle: bool = True,
        model_save_best: bool = True,
        model_save_monitor=("val_acc", "max"),
        class_weight: Union[list, str, None] = None,
        channels_last: bool = True,
        early_stopping: bool = True,
        restore_best_weights: bool = True,
        patience: int = 50,
    ):
        if (model_architecture is None and initial_model is None) or (
            model_architecture is not None and initial_model is not None
        ):
            log.error(
                "Either 'model_architecture' or 'initial_model' "
                "need to be provided in the `config.json`."
            )
            exit(1)

        self.model_architecture = model_architecture
        self.model_hyperparameters = model_hyperparameters
        self.training_dataset_path = training_dataset_path
        self.initial_model = initial_model
        self.results_location = results_location
        self.opt_con = opt_con
        self.opt_params = opt_params
        self.loss = loss
        self.loss_fn_kwargs = loss_fn_kwargs
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size

        if aug_mode not in AUG_MODES:
            log.error(f"Augmentation mode: '{aug_mode}' is not supported.")
            exit(1)
        self.aug_mode = aug_mode

        self.aug_fn_args = []
        for augmentation in augmentations:
            aug_fn = aug.augmentation_map.get(augmentation["name"])
            if aug_fn is None:
                log.error(f"Augmentation: '{augmentation['name']}' is not supported.")
                exit(1)
            self.aug_fn_args.append(
                (
                    aug_fn,
                    augmentation.get("arguments", {}),
                )
            )
        self.augmentations = augmentations

        self.aug_probs = aug_probs
        self.aug_fly = aug_fly
        self.aug_val = aug_val
        self.shuffle = shuffle
        self.model_save_best = model_save_best
        self.model_save_monitor = model_save_monitor
        self.class_weight = class_weight
        self.channels_last = channels_last
        self.early_stopping = early_stopping
        self.restore_best_weights = restore_best_weights
        self.patience = patience

        if self.model_save_monitor[0] == "val_acc":
            self.model_save_monitor = [
                "val_" + self.metric,
                model_save_monitor[1],
            ]
