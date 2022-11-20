"""
This file contains the blocks used to build the U-Net models and the VGG16 feature extractor.
"""

from keras.models import Model
from keras.layers import Input, Convolution2D, LeakyReLU, AveragePooling2D, UpSampling2D, Concatenate, \
    ReLU, MaxPooling2D

import tensorflow as tf


def _down_sampling_block(x: tf.Tensor, filters: int, size: int, trainable: bool = True) -> tf.Tensor:
    """
    U-Net down-sampling block.

    :param x: the input tensor
    :param filters: the number of filters of the output layer
    :param size: the size of the convolutional kernel
    :param trainable: whether the block is trainable
    :return: the output tensor
    """

    x = AveragePooling2D(pool_size=2, padding="valid")(x)
    x = Convolution2D(filters=filters, kernel_size=(size, size), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Convolution2D(filters=filters, kernel_size=(size, size), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def _up_sampling_block(x: tf.Tensor, y: tf.Tensor, filters: int, trainable: bool = True) -> tf.Tensor:
    """
    U-Net up-sampling block.

    :param x: the input tensor
    :param y: the residual skip connection tensor
    :param filters: the number of filters of the output layer
    :param trainable: whether the block is trainable
    :return: the output tensor
    """

    x = UpSampling2D(interpolation="bilinear")(x)
    x = Convolution2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Concatenate(axis=-1)([x, y])
    x = Convolution2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def _u_net_block(x: tf.Tensor, filters: int, trainable: bool = True) -> tf.Tensor:
    """
    U-Net block.

    :param x: the input tensor
    :param filters: the number of filters of the output layer
    :param trainable: whether the block is trainable
    :return: the output tensor
    """

    x = Convolution2D(filters=32, kernel_size=(7, 7), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Convolution2D(filters=32, kernel_size=(7, 7), strides=1, padding="same", trainable=trainable)(x)

    x_0 = LeakyReLU(alpha=0.1)(x)
    x_1 = _down_sampling_block(x_0, 64, 5, trainable=trainable)
    x_2 = _down_sampling_block(x_1, 128, 3, trainable=trainable)
    x_3 = _down_sampling_block(x_2, 256, 3, trainable=trainable)
    x_4 = _down_sampling_block(x_3, 512, 3, trainable=trainable)

    x = _down_sampling_block(x_4, 512, 3, trainable=trainable)

    x = _up_sampling_block(x, x_4, 512, trainable=trainable)
    x = _up_sampling_block(x, x_3, 256, trainable=trainable)
    x = _up_sampling_block(x, x_2, 128, trainable=trainable)
    x = _up_sampling_block(x, x_1, 64, trainable=trainable)
    x = _up_sampling_block(x, x_0, 32, trainable=trainable)

    x = Convolution2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same", trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def build_model_base_flows(trainable: bool = True) -> Model:
    """
    Builds the U-Net model used to compute the base flows.

    :param trainable: whether the model is trainable
    :return: the model
    """

    input_frame_0 = Input(shape=(None, None, 3))
    input_frame_1 = Input(shape=(None, None, 3))

    x = Concatenate(axis=-1)([input_frame_0, input_frame_1])

    model = Model([input_frame_0, input_frame_1], _u_net_block(x, 4, trainable=trainable))

    return model


def build_model_offset_flows(trainable: bool = True) -> Model:
    """
    Builds the U-Net model used to compute the offset flows.

    :param trainable: whether the model is trainable
    :return: the model
    """

    input_frame_0 = Input(shape=(None, None, 3))
    input_frame_1 = Input(shape=(None, None, 3))

    input_base_flow_01 = Input(shape=(None, None, 2))
    input_base_flow_10 = Input(shape=(None, None, 2))

    input_base_flow_t1 = Input(shape=(None, None, 2))
    input_base_flow_t0 = Input(shape=(None, None, 2))

    input_frame_1_inter = Input(shape=(None, None, 3))
    input_frame_0_inter = Input(shape=(None, None, 3))

    x = Concatenate(axis=-1)([
        input_frame_0, input_frame_1, input_base_flow_01, input_base_flow_10, input_base_flow_t1, input_base_flow_t0,
        input_frame_1_inter, input_frame_0_inter
    ])

    model = Model([
        input_frame_0, input_frame_1, input_base_flow_01, input_base_flow_10, input_base_flow_t1, input_base_flow_t0,
        input_frame_1_inter, input_frame_0_inter
    ], _u_net_block(x, 5, trainable=trainable))

    return model


def _convolution_block(
        x: tf.Tensor, length: int, filters: int, size: int, pooling: bool = True, trainable: bool = False
) -> tf.Tensor:
    """
    VGG16 convolution block.

    :param x: the input tensor
    :param length: the number of convolution layers within the block
    :param filters: the number of filters of the output layer
    :param size: the size of the convolution kernel
    :param pooling: whether to apply max-pooling
    :param trainable: whether the block is trainable
    :return: the output tensor
    """

    x = Convolution2D(filters=filters, kernel_size=size, strides=1, padding="same", trainable=trainable)(x)

    for _ in range(length - 1):
        x = ReLU()(x)
        x = Convolution2D(filters=filters, kernel_size=size, strides=1, padding="same", trainable=trainable)(x)

    if pooling:
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=2)(x)

    return x


def build_feature_extractor(trainable: bool = False) -> Model:
    """
    Builds the VGG16 feature extractor used for the perceptual loss.

    :param trainable: whether the model is trainable (should be False if used as a loss)
    :return: the model
    """

    input_frame = Input(shape=(None, None, 3))

    x = _convolution_block(input_frame, 2, 64, 3, pooling=True, trainable=trainable)
    x = _convolution_block(x, 2, 128, 3, pooling=True, trainable=trainable)
    x = _convolution_block(x, 3, 256, 3, pooling=True, trainable=trainable)
    x = _convolution_block(x, 3, 512, 3, pooling=False, trainable=trainable)

    model = Model(input_frame, x, trainable=trainable)

    return model
