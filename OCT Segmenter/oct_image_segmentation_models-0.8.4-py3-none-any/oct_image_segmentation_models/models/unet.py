from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    concatenate,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from typeguard import typechecked
from typing import Callable, Union

from oct_image_segmentation_models.models.base_model import BaseModel

UNET_MODEL_NAME = "unet"


def batch_activate(x):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def convolution_block(x, filters, kernel):
    x = Conv2D(filters, kernel, strides=(1, 1), padding="same", dilation_rate=1)(x)
    x = batch_activate(x)
    return x


def unet_enc_block(inp, size, enc_kernel=(3, 3), conv_layers=2, pool="max"):
    for _ in range(conv_layers):
        inp = convolution_block(inp, filters=size, kernel=enc_kernel)
    c = inp
    if pool == "max":
        inp = MaxPooling2D(pool_size=(2, 2))(inp)
    return [inp, c]


def upsample_conv(inp, size, kernel):
    x = UpSampling2D()(inp)
    x = convolution_block(x, filters=size, kernel=kernel)
    return x


def unet_dec_block(
    inp, size, concat_map, enc_kernel=(3, 3), dec_kernel=(2, 2), conv_layers=2
):
    x = upsample_conv(inp, size, dec_kernel)

    x = concatenate([x, concat_map])
    [x, _] = unet_enc_block(
        x, size, enc_kernel=enc_kernel, conv_layers=conv_layers, pool=False
    )

    return x


@typechecked
class UNet(BaseModel):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        image_height: int,
        image_width: int,
        start_neurons: int = 8,
        pool_layers: int = 4,
        conv_layers: int = 2,
        enc_kernel: Union[list, tuple] = (3, 3),
        dec_kernel: Union[list, tuple] = (2, 2),
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            image_height=image_height,
            image_width=image_width,
        )
        self.start_neurons = start_neurons
        self.pool_layers = pool_layers
        self.conv_layers = conv_layers
        self.enc_kernel = tuple(enc_kernel)
        self.dec_kernel = tuple(dec_kernel)

    def get_preprocess_input_fn(self) -> Callable:
        def preprocess_input_inner(x):
            return x / 255.0

        return preprocess_input_inner

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "start_neurons": self.start_neurons,
                "pool_layers": self.pool_layers,
                "conv_layers": self.conv_layers,
                "enc_kernel": self.enc_kernel,
                "dec_kernel": self.dec_kernel,
            }
        )
        return config

    def build_model(self) -> Model:
        inp = Input(batch_shape=(None, None, None, self.input_channels))

        x = inp

        enc = []

        for i in range(self.pool_layers):
            [x, c] = unet_enc_block(
                x,
                self.start_neurons * (2**i),
                enc_kernel=self.enc_kernel,
                conv_layers=self.conv_layers,
            )

            enc.append(c)

        [x, _] = unet_enc_block(
            x,
            self.start_neurons * (2**self.pool_layers),
            enc_kernel=self.enc_kernel,
            conv_layers=self.conv_layers,
            pool=False,
        )
        x = Dropout(0.5)(x)

        for i in range(self.pool_layers):
            x = unet_dec_block(
                x,
                self.start_neurons * (2 ** (self.pool_layers - 1 - i)),
                enc[self.pool_layers - 1 - i],
                enc_kernel=self.enc_kernel,
                dec_kernel=self.dec_kernel,
                conv_layers=self.conv_layers,
            )

        o = Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="softmax",
        )(x)

        return Model(
            inputs=inp,
            outputs=o,
            name=UNET_MODEL_NAME,
        )
