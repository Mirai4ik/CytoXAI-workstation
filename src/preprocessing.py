from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image
import torch

from .config import IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD

ImageInput = Union[str, Path, BinaryIO, Image.Image]


def load_image(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input)
    return image.convert("RGB")


def image_to_tensor(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    mean: list[float] | tuple[float, float, float] = NORMALIZATION_MEAN,
    std: list[float] | tuple[float, float, float] = NORMALIZATION_STD,
) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    mean_arr = np.array(mean, dtype="float32")
    std_arr = np.array(std, dtype="float32")
    arr = (arr - mean_arr) / std_arr
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_numpy_image(
    tensor: torch.Tensor,
    mean: list[float] | tuple[float, float, float] = NORMALIZATION_MEAN,
    std: list[float] | tuple[float, float, float] = NORMALIZATION_STD,
) -> np.ndarray:
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    mean_arr = np.array(mean, dtype="float32")
    std_arr = np.array(std, dtype="float32")
    arr = arr * std_arr + mean_arr
    arr = np.clip(arr, 0.0, 1.0)
    return arr
