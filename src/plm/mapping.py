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


def smooth_mapping_lut(spacing: int, seed: int = 42,
                       per_channel: bool = True) -> torch.Tensor:
    """Create seeded piecewise-linear, nonmonotonic LUTs over intensity.

    Control points are sampled from U(-1, 1) at intensity coordinates
    ``0, spacing, 2 * spacing, ...``.  The final control point may lie above
    255 so every interval has the same width and the Lipschitz bound remains
    ``2 / spacing``.
    """
    spacing = int(spacing)
    if not 1 <= spacing <= 255:
        raise ValueError(f"spacing must be in [1, 255], got {spacing}")
    channels = 3 if per_channel else 1
    control_count = (255 + spacing - 1) // spacing + 1
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    controls = torch.empty(channels, control_count, dtype=torch.float32)
    controls.uniform_(-1.0, 1.0, generator=generator)

    values = torch.arange(256, dtype=torch.float32)
    lower = torch.div(values.long(), spacing, rounding_mode="floor")
    fraction = (values - lower.float() * spacing) / float(spacing)
    lut = controls[:, lower] * (1.0 - fraction) + controls[:, lower + 1] * fraction
    return lut


class SmoothPixelMapping(nn.Module):
    """Seeded smooth nonmonotonic mapping for BCHW uint8 RGB images."""

    def __init__(self, spacing=16, seed=42, per_channel=True):
        super().__init__()
        self.spacing = int(spacing)
        self.seed = int(seed)
        self.per_channel = bool(per_channel)
        self.register_buffer(
            "lut",
            smooth_mapping_lut(self.spacing, self.seed, self.per_channel),
            persistent=True,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("SmoothPixelMapping expects BCHW uint8 RGB input")
        lut = self.lut if self.per_channel else self.lut.expand(3, -1)
        channel = torch.arange(3, device=image.device)[None, :, None, None]
        return lut[channel, image.long()]


def build_pixel_mapping(configuration=None):
    """Build a mapping from a checkpoint-compatible configuration mapping."""
    configuration = configuration or {"type": "fixed"}
    mapping_type = configuration.get("type", "fixed")
    if mapping_type == "fixed":
        return FixedPixelMapping()
    if mapping_type == "smooth":
        return SmoothPixelMapping(
            spacing=configuration.get("spacing", 16),
            seed=configuration.get("seed", 42),
            per_channel=configuration.get("per_channel", True),
        )
    raise ValueError(f"Unknown pixel mapping type: {mapping_type}")


def random_pixel_mapping(image: torch.Tensor, generator=None) -> torch.Tensor:
    """Equation 5/6 ablation: independent table per sample and channel."""
    if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW uint8 RGB input")
    table = torch.empty(image.shape[0], 3, 256, device=image.device)
    table.uniform_(-1.0, 1.0, generator=generator)
    batch = torch.arange(image.shape[0], device=image.device)[:, None, None, None]
    channel = torch.arange(3, device=image.device)[None, :, None, None]
    return table[batch, channel, image.long()]
