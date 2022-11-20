"""
This file is used to up-sample a given video using the trained models

Arguments:
    -i: the path to the input video
    -o: the path to the output video
    -f: the temporary frame folder (leave unspecified if you don't need to retrieve the frames)
    -mf: the number of frame to extract from the input video
    -p: the model checkpoint to use
    -b: batch size
    -r: the up-sampling rate
    -fps: the frame rate of the output video
"""

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input video")
#     parser.add_argument("-o", "--output", type=str, default="generated.mp4", help="Path to the output video")
#     parser.add_argument("-f", "--frames", type=str, default=None, help="Temporary path to the frames")
#     parser.add_argument("-mf", "--max_frames", type=int, default=None, help="The number of frames to extract")
#     parser.add_argument("-p", "--checkpoint", type=int, default=0, help="The model checkpoint")
#     parser.add_argument("-b", "--batch_size", type=int, default=7, help="Batch size")
#     parser.add_argument("-r", "--rate", type=int, default=7, help="The up-sampling rate")
#     parser.add_argument("-fps", "--frame_rate", type=int, default=30, help="The frame rate of the output video")

import os.path

from keras.utils import Progbar

from model import build_model_base_flows, build_model_offset_flows
from utils import to_input_format, video_to_frames

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

import argparse
import shutil
import cv2


def up_sample(path_frames: str, batch_size: int = 1, up_sample_rate: int = 1, checkpoint: int = 0) -> None:
    """
    Up-samples the video

    Parameters
    ----------
    path_frames: str
        The path to the extracted video frames
    batch_size: int
        The batch size used
    up_sample_rate: int
        The up-sample rate
    checkpoint: int
        The last checkpoint to load the weights from
    """

    data = []

    for k in range(len(os.listdir(path_frames)) - 1):
        frame_0_path = os.path.join(path_frames, f"{str(k * up_sample_rate).zfill(5)}.png")
        frame_1_path = os.path.join(path_frames, f"{str((k + 1) * up_sample_rate).zfill(5)}.png")

        data.append([frame_0_path, frame_1_path])

    times = [t / up_sample_rate for t in range(1, up_sample_rate)]
    batch_times = [
        times[k * batch_size:k * batch_size + batch_size] for k in
        range((up_sample_rate - 1) // batch_size + int((up_sample_rate - 1) % batch_size != 0))
    ]

    sample_frame = cv2.imread(os.path.join(path_frames, "00000.png"))

    original_height, original_width, _ = sample_frame.shape
    resize_height, resize_width, _ = to_input_format(sample_frame).shape

    model_base_flows = build_model_base_flows(trainable=False)
    model_base_flows.load_weights(f"weights/weights_base_{checkpoint}.h5")

    model_offset_flows = build_model_offset_flows(trainable=False)
    model_offset_flows.load_weights(f"weights/weights_offset_{checkpoint}.h5")

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(resize_height, resize_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(resize_height, resize_width, 3), dtype=tf.float32)
    ))
    def compute_base_flows(frame_0: tf.Tensor, frame_1: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Computes the base optic flow between the two frames

        Parameters
        ----------
        frame_0: tf.Tensor
            The first frame of the sequence
        frame_1: tf.Tensor
            The last frame of the sequence

        Returns
        -------
        base_flow_01: tf.Tensor
            The base optic flow from frame_0 to frame_1
        base_flow_10: tf.Tensor
            The base optic flow from frame_1 to frame_0
        """

        frame_0 = tf.expand_dims(frame_0, axis=0)
        frame_1 = tf.expand_dims(frame_1, axis=0)

        base_flows = model_base_flows([frame_0, frame_1])

        base_flow_01 = base_flows[..., :2]
        base_flow_10 = base_flows[..., 2:]

        # noinspection PyTypeChecker
        return base_flow_01, base_flow_10

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, resize_height, resize_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, resize_height, resize_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, resize_height, resize_width, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, resize_height, resize_width, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, ), dtype=tf.float32)
    ))
    def compute_frame(
            frame_0: tf.Tensor, frame_1: tf.Tensor, base_flow_01: tf.Tensor, base_flow_10: tf.Tensor, stamps: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the interpolated optic flow between the two frames

        Parameters
        ----------
        frame_0: tf.Tensor
            The first frame of the sequence
        frame_1: tf.Tensor
            The last frame of the sequence
        base_flow_01: tf.Tensor
            The base optic flow from frame_0 to frame_1
        base_flow_10: tf.Tensor
            The base optic flow from frame_1 to frame_0
        stamps: tf.Tensor
            The time stamps of the generated images

        Returns
        -------
        frame_t_predicted: tf.Tensor
            The predicted frames
        """

        stamps = tf.reshape(stamps, shape=(-1, 1, 1, 1))

        # noinspection PyTypeChecker
        base_flow_t0 = - stamps * (1 - stamps) * base_flow_01 + tf.square(stamps) * base_flow_10

        # noinspection PyTypeChecker
        base_flow_t1 = tf.square(1 - stamps) * base_flow_01 - stamps * (1 - stamps) * base_flow_10

        frame_0_inter = tfa.image.dense_image_warp(frame_0, - base_flow_t0)
        frame_1_inter = tfa.image.dense_image_warp(frame_1, - base_flow_t1)

        offset_flows = model_offset_flows([
            frame_0, frame_1, base_flow_01, base_flow_10, base_flow_t1, base_flow_t0, frame_1_inter, frame_0_inter
        ])

        flow_t0 = offset_flows[..., :2] + base_flow_t0 - 0.5
        flow_t1 = offset_flows[..., 2:4] + base_flow_t1 - 0.5
        visibility_t0 = tf.nn.sigmoid(offset_flows[..., 4])

        frame_0_warped = tfa.image.dense_image_warp(frame_0, - flow_t0)
        frame_1_warped = tfa.image.dense_image_warp(frame_1, - flow_t1)

        visibility_t0 = tf.expand_dims(visibility_t0, axis=-1)
        visibility_t1 = 1 - visibility_t0

        # noinspection PyTypeChecker
        frame_t_predicted = (1 - stamps) * visibility_t0 * frame_0_warped + stamps * visibility_t1 * frame_1_warped
        frame_t_predicted = frame_t_predicted / ((1 - stamps) * visibility_t0 + stamps * visibility_t1)

        return frame_t_predicted

    bar = Progbar(len(data))

    index = 0

    for k in range(len(data)):
        frame_0_path, frame_1_path = data[k]

        frame_0_original = cv2.imread(frame_0_path)
        frame_1_original = cv2.imread(frame_1_path)

        index += 1

        frame_0_un_batched = to_input_format(frame_0_original)
        frame_1_un_batched = to_input_format(frame_1_original)

        frame_0_un_batched = tf.convert_to_tensor(frame_0_un_batched, dtype=tf.float32)
        frame_1_un_batched = tf.convert_to_tensor(frame_1_un_batched, dtype=tf.float32)

        base_flow_01_un_batched, base_flow_10_un_batched = compute_base_flows(
            frame_0_un_batched, frame_1_un_batched
        )

        for times in batch_times:
            frame_batch_size = len(times)

            times_t = tf.convert_to_tensor(np.array(times), dtype=tf.float32)

            frame_0_batched = tf.stack([frame_0_un_batched] * frame_batch_size, axis=0)
            frame_1_batched = tf.stack([frame_1_un_batched] * frame_batch_size, axis=0)

            base_flow_01_batched = tf.concat([base_flow_01_un_batched] * frame_batch_size, axis=0)
            base_flow_10_batched = tf.concat([base_flow_10_un_batched] * frame_batch_size, axis=0)

            frames_t = compute_frame(
                frame_0_batched, frame_1_batched, base_flow_01_batched, base_flow_10_batched, times_t
            )

            frames_t = frames_t.numpy()

            for frame_t in frames_t:
                frame_t[..., 0] += 0.429
                frame_t[..., 1] += 0.431
                frame_t[..., 2] += 0.397
                frame_t = (frame_t * 256.).astype(np.uint8)

                frame_t = cv2.resize(frame_t, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(os.path.join(path_frames, f"{str(index).zfill(5)}.png"), frame_t)

                index += 1

        bar.add(1)


def frames_to_video(path_frames: str, path_output: str, frames_per_second: int):
    sample_frame = cv2.imread(os.path.join(path_frames, "00000.png"))

    height, width, _ = sample_frame.shape

    out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc("m", "p", "4", "v"), frames_per_second, (width, height))

    for index in range(len(os.listdir(path_frames))):
        frame = cv2.imread(os.path.join(path_frames, f"{str(index).zfill(5)}.png"))

        out.write(frame)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input video")
    parser.add_argument("-o", "--output", type=str, default="generated.mp4", help="Path to the output video")
    parser.add_argument("-f", "--frames", type=str, default=None, help="Temporary path to the frames")
    parser.add_argument("-mf", "--max_frames", type=int, default=None, help="The number of frames to extract")
    parser.add_argument("-p", "--checkpoint", type=int, default=0, help="The model checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=7, help="Batch size")
    parser.add_argument("-r", "--rate", type=int, default=7, help="The up-sampling rate")
    parser.add_argument("-fps", "--frame_rate", type=int, default=30, help="The frame rate of the output video")

    args = parser.parse_args()

    _input = args.input
    _output = args.output
    _frames = args.frames
    _max_frames = args.max_frames
    _checkpoint = args.checkpoint
    _batch_size = args.batch_size
    _up_sample_rate = args.rate
    _frames_per_second = args.frame_rate

    if _frames:
        _path_frames = _frames
    else:
        _path_frames = "_tmp"

    print("Decompressing the original video...")
    video_to_frames(_input, _path_frames, _up_sample_rate, _max_frames)

    print("Up-sampling the video...")
    up_sample(_path_frames, _batch_size, _up_sample_rate, _checkpoint)

    print("Constructing the final video...")
    frames_to_video(_path_frames, _output, _frames_per_second)

    if not _frames:
        shutil.rmtree(_path_frames)

    print("Done!")
