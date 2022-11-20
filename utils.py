import numpy as np

import shutil
import cv2
import os


def to_input_format(image: np.ndarray) -> np.ndarray:
    """
    Resizes the image to fit the input format of the model (the dimension of the frames must be a multiple of 32).

    Parameters
    ----------
    image: np.ndarray
        The input image

    Returns
    -------
    input_image: np.ndarray
        The formatted image
    """

    height, width, _ = image.shape

    image = cv2.resize(image, (int(width / 32) * 32, int(height / 32) * 32), interpolation=cv2.INTER_LINEAR)

    image = image.astype(float) / 256.
    image[..., 0] -= 0.429
    image[..., 1] -= 0.431
    image[..., 2] -= 0.397

    return image


def video_to_frames(path_video: str, path_frames: str, up_sample_rate: int = 1, max_frames: int = None) -> None:
    """
    Extracts the frames of a video with time stamps

    Parameters
    ----------
    path_video: str
        The path to the video file
    path_frames: str
        The path to the frame folder
    up_sample_rate: int
        The number of inserted frames
    max_frames: int
        The maximum number of frames to extract
    """

    if os.path.exists(path_frames):
        shutil.rmtree(path_frames)

    os.mkdir(path_frames)

    video_capture = cv2.VideoCapture(path_video)
    count = 0

    success, frame = video_capture.read()

    while success:
        cv2.imwrite(os.path.join(path_frames, f"{str(count * up_sample_rate).zfill(5)}.png"), frame)

        success, frame = video_capture.read()
        count += 1

        if max_frames and count >= max_frames:
            break
