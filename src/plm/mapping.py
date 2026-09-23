import numpy as np
import torch
from torch import nn


def fixed_mapping_lut() -> torch.Tensor:
    """Return Equation 4's exact NumPy-round lookup table for uint8 values."""
    values = np.arange(256, dtype=np.float64)
    mapped = values - np.round(values / 256.0, decimals=2) * 256.0
    return torch.from_numpy(mapped.astype(np.float32))


class FixedPixelMapping(nn.Module):
    """Map RGB uint8 values using the paper's fixed, channel-shared table."""

    def __init__(self):
        super().__init__()
        self.register_buffer("lut", fixed_mapping_lut(), persistent=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.uint8:
            raise TypeError(f"FixedPixelMapping expects uint8 input, got {image.dtype}")
        return self.lut[image.long()]


def random_pixel_mapping(image: torch.Tensor, generator=None) -> torch.Tensor:
    """Equation 5/6 ablation: independent table per sample and channel."""
    if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW uint8 RGB input")
    table = torch.empty(image.shape[0], 3, 256, device=image.device)
    table.uniform_(-1.0, 1.0, generator=generator)
    batch = torch.arange(image.shape[0], device=image.device)[:, None, None, None]
    channel = torch.arange(3, device=image.device)[None, :, None, None]
    return table[batch, channel, image.long()]
