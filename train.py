"""
This file is used for the training of both models.

Arguments:
    -v: path to the folder containing the training videos (don't specify if the dataset has already been generated)
    -d: path to the dataset containing the decompressed videos
    -w: path to the weights (this folder contains the checkpoints of the models, saved after each epoch)
    -e: number of epochs
    -b: batch size (default: 7)
    -l: learning rate (default: 0.00002)
    -i: in between frames between the high and low frame-rate videos (default: 7)
    -c: crop size of the random crop used for data augmentation if specified (optional)
    -p: the last checkpoint to load the weights from (optional)
    -s: the number of samples per sequence, used to avoid over-fitting on longer sequences (default: 50)
"""

import os.path
import shutil

from keras.optimizers import Adam
from keras.utils import Progbar

from model import build_model_base_flows, build_model_offset_flows, build_feature_extractor
from utils import to_input_format, video_to_frames

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

import argparse
import cv2


def _get_dataset_samples(path_dataset: str, in_between_frames: int = 7) -> list:
    """
    Returns the number of sequences of the specified size available in each video of the dataset.

    Parameters
    ----------
    path_dataset: str
        The path to the dataset
    in_between_frames: int
        The number of in between frames

    Returns
    -------
    available_sequences: list
        The list of available sequences count
    """

    available_sequences = []

    for k in range(len(os.listdir(path_dataset))):
        sequence_path = os.path.join(path_dataset, str(k).zfill(3))

        sequence_length = len(os.listdir(sequence_path))

        if sequence_length % (in_between_frames + 1) == 0:
            sequence_length -= 1

        sequence_length = sequence_length - sequence_length % (in_between_frames + 1)
        sequence_length = int(sequence_length / (in_between_frames + 1))

        available_sequences.append(sequence_length)

    return available_sequences


def _generate_samples(available_sequences: list, sample_per_sequence: int = 50) -> list:
    """
    Samples random sequences from each video of the dataset.

    Parameters
    ----------
    available_sequences: list
        The list of available sequences count
    sample_per_sequence: int
        The number of samples per sequence

    Returns
    -------
    sequence_pairs: list
        The list of sampled sequences
    """

    sequence_pairs = []

    for sequence in range(len(available_sequences)):
        available_frames = list(range(available_sequences[sequence]))

        np.random.shuffle(available_frames)

        for frame in available_frames[:sample_per_sequence]:
            sequence_pairs.append((sequence, frame))

    np.random.shuffle(sequence_pairs)

    return sequence_pairs


def _sample_sequence(
        path_dataset: str, sequence: int, frame_index: int, batch_size: int = 1, in_between_frames: int = 7,
        crop_size: int = 128
) -> (tf.Tensor, tf.Tensor, list):
    """
    Samples a sequence from the dataset.

    Parameters
    ----------
    path_dataset: str
        The path to the dataset
    sequence: int
        The sequence index
    frame_index: int
        The frame index
    batch_size: int
        The batch size
    in_between_frames: int
        The number of in between frames
    crop_size: int
        The crop size

    Returns
    -------
    frame_0: tf.Tensor
        The starting frame of the sub-sequence
    frame_1: tf.Tensor
        The ending frame of the sub-sequence
    batch_frame: list
        The batched frames of the sub-sequence (excluding the first and last frame)
    """

    path_seq = os.path.join(path_dataset, str(sequence).zfill(3))

    frame_0 = cv2.imread(os.path.join(path_seq, str(frame_index * (in_between_frames + 1)).zfill(5) + '.png'))
    frame_1 = cv2.imread(os.path.join(path_seq, str((frame_index + 1) * (in_between_frames + 1)).zfill(5) + '.png'))

    height, width, _ = frame_0.shape

    crop_y = np.random.randint(0, height - crop_size)
    crop_x = np.random.randint(0, width - crop_size)

    frame_0 = to_input_format(frame_0[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size])
    frame_1 = to_input_format(frame_1[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size])

    frame_0 = tf.convert_to_tensor(frame_0, dtype=tf.float32)
    frame_1 = tf.convert_to_tensor(frame_1, dtype=tf.float32)

    frame_t = []

    for k in range(1, 1 + in_between_frames):
        frame = cv2.imread(os.path.join(path_seq, str(k + frame_index * (in_between_frames + 1)).zfill(5) + '.png'))
        frame = to_input_format(frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size])

        frame_t.append(frame)

    batch_frame = [
        tf.convert_to_tensor(frame_t[k * batch_size:k * batch_size + batch_size], dtype=tf.float32) for k in
        range(in_between_frames // batch_size + int(in_between_frames % batch_size != 0))
    ]

    # noinspection PyTypeChecker
    return frame_0, frame_1, batch_frame


def train_model(
        path_dataset: str, path_weights: str, epochs: int = 10, batch_size: int = 7, learning_rate: float = 1e-5,
        in_between_frames: int = 7, crop_size: int = 128, last_checkpoint: int = None, sample_per_sequence: int = 50
) -> None:
    """
    Custom training loop for both models.

    Parameters
    ----------
    path_dataset: str
        The path to the dataset containing the sequence
    path_weights: str
        The path to the folder where the weights will be saved
    epochs: int
        The number of epochs to train the model
    batch_size: int
        The batch size
    learning_rate: float
        The learning rate of the optimizer
    in_between_frames: int
        The number of in-between frames
    crop_size: int
        The random crop size
    last_checkpoint: int
        The last checkpoint to load the weights from
    sample_per_sequence: int
        The number of samples per sequence
    """

    optimizer = Adam(learning_rate=learning_rate)

    model_base_flows = build_model_base_flows(trainable=True)
    model_offset_flows = build_model_offset_flows(trainable=True)

    feature_extractor = build_feature_extractor(trainable=False)
    feature_extractor.load_weights("weights/weights_vgg16.h5", by_name=True, skip_mismatch=True)

    trainable_variables = model_base_flows.trainable_variables + model_offset_flows.trainable_variables

    if last_checkpoint is not None:
        print(f"Starting from previous checkpoint {last_checkpoint}")

        path_optimizer = os.path.join(path_weights, f"optimizer_{last_checkpoint}.npy")

        if os.path.exists(path_optimizer):
            zero_grads = [tf.zeros_like(w) for w in trainable_variables]

            weights = np.load(path_optimizer, allow_pickle=True)

            optimizer.apply_gradients(zip(zero_grads, trainable_variables))
            optimizer.set_weights(weights)

        model_base_flows.load_weights(os.path.join(path_weights, f"weights_base_{last_checkpoint}.h5"))
        model_offset_flows.load_weights(os.path.join(path_weights, f"weights_offset_{last_checkpoint}.h5"))

        checkpoint = last_checkpoint + 1

    else:
        model_base_flows.load_weights("weights/weights_base.h5")
        model_offset_flows.load_weights("weights/weights_offset.h5")

        checkpoint = 0

    available_sequences = _get_dataset_samples(path_dataset)

    times = [t / (in_between_frames + 1) for t in range(1, in_between_frames + 1)]
    batch_times = [
        tf.convert_to_tensor(times[k * batch_size:k * batch_size + batch_size], dtype=tf.float32) for k in
        range(in_between_frames // batch_size + int(in_between_frames % batch_size != 0))
    ]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(crop_size, crop_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(crop_size, crop_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, crop_size, crop_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ))
    def train_step(frame_0: tf.Tensor, frame_1: tf.Tensor, frame_t_gt: tf.Tensor, stamps: tf.Tensor) -> tf.Tensor:
        """
        The train step function. It takes the input frames, the ground truth, and the timestamps and returns the loss
        while updating the weights of both models.

        Parameters
        ----------
        frame_0: tf.Tensor
            The first frame of the sequence
        frame_1: tf.Tensor
            The last frame of the sequence
        frame_t_gt: tf.Tensor
            The ground truth in-between frames of the sequence
        stamps: tf.Tensor
            The timestamps of the ground truth in-between frames

        Returns
        -------
        loss: tf.Tensor
            The loss value
        """

        with tf.GradientTape() as tape:
            frame_0 = tf.expand_dims(frame_0, axis=0)
            frame_1 = tf.expand_dims(frame_1, axis=0)

            base_flows = model_base_flows([frame_0, frame_1])

            base_flow_01 = base_flows[..., :2]
            base_flow_10 = base_flows[..., 2:]

            frame_0 = tf.repeat(frame_0, repeats=7, axis=0)
            frame_1 = tf.repeat(frame_1, repeats=7, axis=0)

            base_flow_01 = tf.repeat(base_flow_01, repeats=7, axis=0)
            base_flow_10 = tf.repeat(base_flow_10, repeats=7, axis=0)

            stamps = tf.reshape(stamps, shape=(-1, 1, 1, 1))

            base_flow_t0 = - stamps * (1 - stamps) * base_flow_01 + tf.square(stamps) * base_flow_10
            base_flow_t1 = tf.square(1 - stamps) * base_flow_01 - stamps * (1 - stamps) * base_flow_10

            frame_0_inter = tfa.image.dense_image_warp(frame_0, - base_flow_t0)
            frame_1_inter = tfa.image.dense_image_warp(frame_1, - base_flow_t1)

            offset_flows = model_offset_flows([
                frame_0, frame_1, base_flow_01, base_flow_10, base_flow_t1, base_flow_t0, frame_1_inter, frame_0_inter
            ])

            flow_t0 = offset_flows[..., :2] + base_flow_t0
            flow_t1 = offset_flows[..., 2:4] + base_flow_t1
            visibility_t0 = tf.nn.sigmoid(offset_flows[..., 4])

            frame_0_warped = tfa.image.dense_image_warp(frame_0, - flow_t0)
            frame_1_warped = tfa.image.dense_image_warp(frame_1, - flow_t1)

            visibility_t0 = tf.expand_dims(visibility_t0, axis=-1)
            visibility_t1 = 1 - visibility_t0

            # noinspection PyTypeChecker
            frame_t_predicted = (1 - stamps) * visibility_t0 * frame_0_warped + stamps * visibility_t1 * frame_1_warped
            frame_t_predicted = frame_t_predicted / ((1 - stamps) * visibility_t0 + stamps * visibility_t1)

            # noinspection PyTypeChecker
            loss_reconstruction = tf.reduce_mean(tf.abs(frame_t_predicted - frame_t_gt), axis=(1, 2, 3))

            loss_perception = tf.reduce_mean(tf.square(
                feature_extractor(frame_t_predicted) - feature_extractor(frame_t_gt)
            ), axis=(1, 2, 3))

            loss_warping = tf.reduce_mean(tf.abs(frame_t_gt - frame_0_inter), axis=(1, 2, 3)) + \
                tf.reduce_mean(tf.abs(frame_t_gt - frame_1_inter), axis=(1, 2, 3)) + \
                tf.reduce_mean(tf.abs(frame_0 - tfa.image.dense_image_warp(frame_1, - base_flow_01)), axis=(1, 2, 3)) +\
                tf.reduce_mean(tf.abs(frame_1 - tfa.image.dense_image_warp(frame_0, - base_flow_10)), axis=(1, 2, 3))

            loss_smoothness = \
                tf.reduce_mean(tf.abs(base_flow_01[:, :-1, ...] - base_flow_01[:, 1:, ...]), axis=(1, 2, 3)) + \
                tf.reduce_mean(tf.abs(base_flow_01[..., :-1, :] - base_flow_01[..., 1:, :]), axis=(1, 2, 3)) + \
                tf.reduce_mean(tf.abs(base_flow_10[:, :-1, ...] - base_flow_10[:, 1:, ...]), axis=(1, 2, 3)) + \
                tf.reduce_mean(tf.abs(base_flow_10[..., :-1, :] - base_flow_10[..., 1:, :]), axis=(1, 2, 3))

            total_loss = 204 * loss_reconstruction + 102 * loss_warping + 0.005 * loss_perception + loss_smoothness

        gradients_offset = tape.gradient(total_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients_offset, trainable_variables))

        return total_loss

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        sequence_pairs = _generate_samples(available_sequences, sample_per_sequence)

        bar = Progbar(len(sequence_pairs) * len(batch_times))

        current_sequence = 0

        for sequence, frame_index in sequence_pairs:
            frame_0_un_batched, frame_1_un_batched, batched_frames_t = _sample_sequence(
                path_dataset, sequence, frame_index, batch_size, in_between_frames, crop_size
            )

            for times_t, frames_t in zip(batch_times, batched_frames_t):
                loss = train_step(frame_0_un_batched, frame_1_un_batched, frames_t, times_t)

                current_sequence += 1

                bar.update(current_sequence, [("loss", tf.reduce_mean(loss).numpy())])

        np.save(os.path.join(path_weights, f"optimizer_{checkpoint}.npy"), optimizer.get_weights(), allow_pickle=True)

        model_base_flows.save_weights(os.path.join(path_weights, f"weights_base_{checkpoint}.h5"))
        model_offset_flows.save_weights(os.path.join(path_weights, f"weights_offset_{checkpoint}.h5"))

        print(f"Saved models @ {checkpoint}")

        checkpoint += 1


def create_dataset(path_video: str, path_dataset: str):
    """
    Creates the dataset sequences

    Parameters
    ----------
    path_video: str
        The path to the folder containing the training videos
    path_dataset: str
        The path to the dataset
    """

    if os.path.exists(path_dataset):
        shutil.rmtree(path_dataset)

    os.mkdir(path_dataset)

    for k, file in enumerate(os.listdir(path_video)):
        path_sequence = os.path.join(path_dataset, f"{str(k).zfill(3)}/")
        path_video = os.path.join(path_video, file)

        video_to_frames(path_video, path_sequence, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videos", type=str, default=None, help="Path to the video folder")
    parser.add_argument("-d", "--dataset", type=str, default="dataset/", help="Path to dataset")
    parser.add_argument("-w", "--weights", type=str, default="weights/", help="Path to weights")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=7, help="Batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("-i", "--in_between_frames", type=int, default=7, help="Number of in-between frames")
    parser.add_argument("-c", "--crop_size", type=int, default=128, help="Crop size")
    parser.add_argument("-p", "--checkpoint", type=int, default=None, help="Checkpoint")
    parser.add_argument("-s", "--samples", type=int, default=50, help="Number of samples")

    args = parser.parse_args()

    _path_videos = args.videos
    _path_dataset = args.dataset
    _path_weights = args.weights

    _epochs = args.epochs
    _batch_size = args.batch_size
    _learning_rate = args.learning_rate
    _in_between_frames = args.in_between_frames
    _crop_size = args.crop_size
    _checkpoint = args.checkpoint
    _samples = args.samples

    if _path_videos:
        print("Creating dataset...")
        create_dataset(_path_videos, _path_dataset)

    train_model(
        _path_dataset, _path_weights, _epochs, _batch_size, _learning_rate, _in_between_frames, _crop_size, _checkpoint,
        _samples
    )
