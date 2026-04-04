from typeguard import typechecked
from typing import Type

from . import base_model
from . import deeplabv3plus
from . import unet


model_name_map = {
    deeplabv3plus.DEEPLABV3PLUS_MODEL_NAME: deeplabv3plus.DeeplabV3Plus,
    unet.UNET_MODEL_NAME: unet.UNet,
}


@typechecked
def get_model_class(model_name: str) -> Type[base_model.BaseModel]:
    model_class = model_name_map.get(model_name)

    if model_class is None:
        raise ValueError(f"Model name: '{model_name}' could not be found.")

    return model_class
