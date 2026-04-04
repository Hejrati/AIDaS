import atexit
import os
import sys

import h5py
import json
import logging as log
import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from typeguard import typechecked
from typing import Union

from oct_image_segmentation_models.common import (
    custom_losses,
    custom_metrics,
    data_generator as data_gen,
    dataset_loader,
    utils,
)
from oct_image_segmentation_models.common.mlflow_parameters import (
    MLflowParameters,
)
from oct_image_segmentation_models.models import get_model_class
from oct_image_segmentation_models.training import (
    training_callbacks,
    training_parameters as tparams,
)
from oct_image_segmentation_models.training.training_parameters import (
    TrainingParams,
)


@typechecked
def save_training_params_file(
    save_foldername: Path,
    model_summary: str,
    model_config: dict,
    training_dataset_md5: str,
    class_weight: Union[np.ndarray, None],
    timestamp,
    train_params: TrainingParams,
    opt,
):
    with open(save_foldername / Path("model_config.json"), "w") as config_file:
        json.dump(model_config, config_file)
    config_filename = save_foldername / Path("training_params.hdf5")

    config_file = h5py.File(config_filename, "w")

    config_file.attrs["timestamp"] = np.array(timestamp, dtype="S100")
    config_file.attrs["model_summary"] = np.array(model_summary, dtype="S1000")
    config_file.attrs["train_dataset_md5"] = np.array(
        training_dataset_md5, dtype="S1000"
    )

    config_file.attrs["epochs"] = train_params.epochs
    config_file.attrs["loss_name"] = np.array(train_params.loss, dtype="S1000")
    config_file.attrs["metric_name"] = np.array(train_params.metric, dtype="S1000")

    if class_weight is None:
        config_file.attrs["class_weight"] = np.array("None", dtype="S1000")
    else:
        config_file.attrs["class_weight"] = np.array("array", dtype="S1000")
        config_file["class_weight"] = class_weight

    config_file.attrs["metric"] = np.array(train_params.metric, dtype="S100")
    config_file.attrs["loss"] = np.array(train_params.loss, dtype="S100")
    config_file.attrs["batch_size"] = train_params.batch_size
    config_file.attrs["shuffle"] = train_params.shuffle

    config_file.attrs["aug_mode"] = np.array(train_params.aug_mode, dtype="S100")

    if train_params.aug_mode != "none":
        for aug_ind in range(len(train_params.aug_fn_args)):
            aug_fn = train_params.aug_fn_args[aug_ind][0]
            aug_arg = train_params.aug_fn_args[aug_ind][1]

            aug_desc = aug_fn(None, None, aug_arg, True)

            if type(aug_arg) is not dict:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(
                    aug_desc, dtype="S1000"
                )
            else:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(
                    aug_fn.__name__, dtype="S100"
                )

                for aug_param_key in aug_arg.keys():
                    val = aug_arg[aug_param_key]
                    if type(val) is int or type(val) is float:
                        config_file.attrs[
                            "aug_" + str(aug_ind + 1) + "_param: " + aug_param_key
                        ] = np.array(val)
                    elif type(val) is str:
                        config_file.attrs[
                            "aug_" + str(aug_ind + 1) + "_param: " + aug_param_key
                        ] = np.array(val, dtype="S100")
                    elif type(val) is list and (
                        type(val[0]) is int
                        or type(val[0]) is str
                        or type(val[0]) is float
                    ):
                        config_file.attrs[
                            "aug_" + str(aug_ind + 1) + "_param: " + aug_param_key
                        ] = np.array(str(val), dtype="S100")

            if train_params.aug_mode == "one":
                config_file.attrs["aug_probs"] = np.array(train_params.aug_probs)

        config_file.attrs["aug_fly"] = train_params.aug_fly
        config_file.attrs["aug_val"] = train_params.aug_val

    config_file.attrs["optimizer"] = np.array(
        train_params.opt_con.__name__, dtype="S100"
    )

    opt_config = opt.get_config()

    for key in opt_config:
        if type(opt_config[key]) is dict:
            config_file.attrs["opt_param: " + key] = np.string_(str(opt_config[key]))
        else:
            config_file.attrs["opt_param: " + key] = opt_config[key]

    config_file.close()


@typechecked
def train_model(
    training_params: tparams.TrainingParams,
    mlflow_params: MLflowParameters | None,
):
    if mlflow_params:
        mlflow.tensorflow.autolog(keras_model_kwargs={"save_format": "h5"})
        if mlflow_params.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_params.username

        if mlflow_params.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_params.password

        mlflow.set_tracking_uri(mlflow_params.tracking_uri)
        try:
            mlflow.set_experiment(mlflow_params.experiment)
            mlflow_run = mlflow.start_run()
            log.info(f"MLFlow Run ID: {mlflow_run.info.run_id}")
        except MlflowException as exc:
            if exc.get_http_status_code() == 401:
                log.error(
                    "Looks like the MLFLow client is not authorized to log "
                    "into the MLFlow server. Make sure the environment "
                    "variables 'MLFLOW_TRACKING_USERNAME' and "
                    "'MLFLOW_TRACKING_PASSWORD' are correct"
                )
            log.exception(msg="An error occurred while setting MLflow experiment")
            sys.exit(1)

    if training_params.channels_last:
        tf.keras.backend.set_image_data_format("channels_last")

    training_dataset_path = training_params.training_dataset_path
    training_hdf5_file = h5py.File(training_dataset_path, "r")

    # images numpy array should be of the shape: (number of images, image
    # width, image height, 1) segments numpy array should be of the shape:
    # (number of images, number of boundaries, image width)
    train_images, train_labels = dataset_loader.load_training_data(training_hdf5_file)
    val_images, val_labels = dataset_loader.load_validation_data(training_hdf5_file)

    num_classes = len(np.unique(train_labels))
    log.info(f"Detected {num_classes} classes")

    _, image_height, image_width, input_channels = train_images.shape
    log.info(
        f"Detected input image dimensions (h x w): {image_height} x " f"{image_width}."
    )
    log.info(f"Detected {input_channels} input channels.")

    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore

    log.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    optimizer_con = training_params.opt_con
    optimizer_params = training_params.opt_params

    optimizer = optimizer_con(**optimizer_params)

    loss = custom_losses.custom_loss_objects.get(training_params.loss)
    if loss is None:
        log.error(f"Loss '{training_params.loss}' not found. Exiting...")
        exit(1)
    else:
        if training_params.class_weight == "balanced":
            dataset_labels = np.concatenate((train_labels, val_labels))
            c_weight = class_weight.compute_class_weight(
                "balanced",
                classes=np.unique(dataset_labels),
                y=dataset_labels.flatten(),
            )
        elif type(training_params.class_weight) == list:
            c_weight = np.array(training_params.class_weight)
        else:
            c_weight = None
        sparse_labels = loss["takes_sparse"]
        loss_fn = loss["function"](
            num_classes=num_classes,
            is_y_true_sparse=sparse_labels,
            **training_params.loss_fn_kwargs,
        )

    metric = custom_metrics.training_monitor_metric_objects.get(training_params.metric)
    if metric is None:
        log.error(f"Metric '{training_params.metric}' not found. Exiting...")
        exit(1)
    else:
        metric_fn = metric(sparse_labels, num_classes)

    if not sparse_labels:
        train_labels = to_categorical(train_labels, num_classes)
        val_labels = to_categorical(val_labels, num_classes)

    training_dataset_md5 = utils.md5(training_dataset_path)

    epochs = training_params.epochs
    initial_model_path = training_params.initial_model
    early_stopping = training_params.early_stopping
    model_architecture = training_params.model_architecture
    model_hyperparameters = training_params.model_hyperparameters

    if initial_model_path:
        log.info(f"Starting training from model: {initial_model_path}")
        model = utils.load_model(initial_model_path)
    else:
        log.info(f"Starting training from scratch {model_architecture} model")

        with strategy.scope():
            try:
                model_class = get_model_class(
                    training_params.model_architecture,
                )
            except ValueError as e:
                log.error(e)
                exit(1)

            model_container = model_class(
                input_channels=input_channels,
                num_classes=num_classes,
                image_height=image_height,
                image_width=image_width,
                **model_hyperparameters,
            )

            model = model_container.build_model()

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=[metric_fn],
            )

    batch_size = training_params.batch_size
    aug_fn_args = training_params.aug_fn_args
    aug_mode = training_params.aug_mode
    aug_probs = training_params.aug_probs
    aug_fly = training_params.aug_fly
    aug_val = training_params.aug_val
    patience = training_params.patience
    restore_best_weights = training_params.restore_best_weights

    mlflow.log_params(
        {
            "model_architecture": training_params.model_architecture,
            "training_dataset_path": training_dataset_path,
            "training_dataset_md5": training_dataset_md5,
            "augmentation_mode": training_params.aug_mode,
            "augmentations": training_params.augmentations,
            "loss_name": training_params.loss,
            "loss_fn_kwargs": training_params.loss_fn_kwargs,
            "metric_name": training_params.metric,
            "loss_fn_class_weight": training_params.class_weight,
            "class_weight_array": c_weight,
        }
    )

    mlflow.log_dict(model_container.get_config(), "model/data/model_config.json")

    if aug_val is False:
        aug_val_mode = "none"
        aug_val_fn_args = []
        aug_val_probs = ()
        aug_val_fly = False
    else:
        aug_val_mode = aug_mode
        aug_val_fn_args = aug_fn_args
        aug_val_probs = aug_probs
        aug_val_fly = aug_fly

    monitor = training_params.model_save_monitor
    save_best = training_params.model_save_best
    timestamp = utils.get_timestamp()

    results_location = training_params.results_location
    run_id = "" if not mlflow_params else mlflow_run.info.run_id
    save_foldername = (
        results_location
        / Path(run_id)
        / Path(timestamp + "_" + model_architecture)
    )

    os.makedirs(save_foldername)

    epoch_model_name = "model_epoch{epoch:02d}.hdf5"

    savemodel = ModelCheckpoint(
        filepath=save_foldername / Path(epoch_model_name),
        save_best_only=save_best,
        monitor=monitor[0],
        mode=monitor[1],
    )

    history = training_callbacks.SaveEpochInfo(
        save_folder=save_foldername,
        train_params=training_params,
    )

    callbacks_list = [savemodel, history]

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{training_params.metric}",
            mode="max",
            patience=patience,
            restore_best_weights=restore_best_weights,
        )
        callbacks_list.append(early_stopping_callback)

    model_summary = []
    model.summary(print_fn=lambda line: model_summary.append(line))

    save_training_params_file(
        save_foldername,
        "\n".join(model_summary),
        model_container.get_config(),
        training_dataset_md5,
        c_weight,
        timestamp,
        training_params,
        optimizer,
    )

    train_gen = data_gen.DataGenerator(
        train_images,
        train_labels,
        batch_size,
        aug_fn_args,
        aug_mode,
        aug_probs,
        aug_fly,
        model_container.get_preprocess_input_fn(),
    )

    val_gen = data_gen.DataGenerator(
        val_images,
        val_labels,
        batch_size,
        aug_val_fn_args,
        aug_val_mode,
        aug_val_probs,
        aug_val_fly,
        model_container.get_preprocess_input_fn(),
    )

    train_gen_total_samples = train_gen.get_total_samples()
    if batch_size > train_gen_total_samples:
        log.error(
            f"The batch size ({batch_size}) cannot be larger than the number "
            f"of training samples ({train_gen_total_samples})"
        )
        exit(1)
    log.info(f"Train generator total number of samples: {train_gen_total_samples}")

    val_gen_total_samples = val_gen.get_total_samples()
    if batch_size > val_gen_total_samples:
        log.error(
            f"The batch size ({batch_size}) cannot be larger than the number "
            f"of validation samples ({val_gen_total_samples})"
        )
        exit(1)
    log.info(
        "Validation generator total number of samples: " f"{val_gen_total_samples}"
    )

    model.summary()
    model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    mlflow.end_run()
