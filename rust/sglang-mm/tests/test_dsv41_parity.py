import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.multimodal._core import dsv41


def reference_resize_patchify(
    pixels: np.ndarray,
    output_size: tuple[int, int],
    resize_size: tuple[int, int],
    padding_start: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    out_h, out_w = output_size
    resize_h, resize_w = resize_size
    top, left = padding_start
    resized = np.asarray(
        Image.fromarray(pixels).resize(
            (resize_w, resize_h), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )
    canvas = np.full((out_h, out_w, 3), 127, dtype=np.uint8)
    canvas[top : top + resize_h, left : left + resize_w] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255
    tensor = ((tensor - 0.5) / 0.5).to(torch.bfloat16)
    grid_h, grid_w = out_h // patch_size, out_w // patch_size
    patches = (
        tensor.reshape(3, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
    )
    return patches.view(torch.uint16).numpy().reshape(-1)


@pytest.mark.parametrize(
    "input_size,output_size,resize_size,padding_start,patch_size",
    [
        ((53, 37), (56, 56), (56, 39), (0, 8), 14),
        ((37, 53), (56, 56), (39, 56), (8, 0), 14),
        ((28, 28), (28, 28), (28, 28), (0, 0), 14),
        ((91, 157), (112, 168), (97, 168), (7, 0), 14),
    ],
)
def test_resize_patchify_matches_pil(
    input_size, output_size, resize_size, padding_start, patch_size
):
    height, width = input_size
    pixels = np.random.default_rng(height * 1000 + width).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    actual = dsv41.resize_patchify(
        pixels, output_size, resize_size, padding_start, patch_size
    )
    expected = reference_resize_patchify(
        pixels, output_size, resize_size, padding_start, patch_size
    )
    np.testing.assert_array_equal(actual, expected)

