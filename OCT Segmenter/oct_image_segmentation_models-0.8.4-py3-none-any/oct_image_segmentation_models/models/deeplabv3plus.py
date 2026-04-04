import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typeguard import typechecked
from typing import Callable

from oct_image_segmentation_models.models.base_model import BaseModel

DEEPLABV3PLUS_MODEL_NAME = "deeplabv3plus"


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)

    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    """
    'dspp_input' shape = (batch_size, height, width, channels)

    Since we want 'out_pool' to have the same dimesions as the input
    (i.e. dims), the upsampling is done by taking dims[-3] (Recall that given
    the input shape dims[-3] == dims[1] == height) and dividing by the height
    of 'x' (i.e. x.shape[1]). Then, the dimension size of the output is
    dims[-3]. It is equivalent with the width: dims[-2] = dims[2] = width
    """

    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)

    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


@typechecked
class DeeplabV3Plus(BaseModel):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            image_height=image_height,
            image_width=image_width,
        )

    def get_config(self) -> dict:
        return super().get_config()

    def get_preprocess_input_fn(self) -> Callable:
        return keras.applications.resnet50.preprocess_input

    def build_model(
        self,
        **kwargs,
    ) -> Model:
        model_input = keras.Input(shape=(self.image_height, self.image_width, 3))
        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )

        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(
                self.image_height // 4 // x.shape[1],
                self.image_width // 4 // x.shape[2],
            ),
            interpolation="bilinear",
        )(x)

        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(
                self.image_height // x.shape[1],
                self.image_width // x.shape[2],
            ),
            interpolation="bilinear",
        )(x)

        model_output = layers.Conv2D(
            self.num_classes,
            kernel_size=(1, 1),
            padding="same",
            activation="softmax",
        )(x)

        return Model(
            inputs=model_input,
            outputs=model_output,
            name=DEEPLABV3PLUS_MODEL_NAME,
        )
