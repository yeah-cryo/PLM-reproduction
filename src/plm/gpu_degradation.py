"""Batched CUDA implementation of the random-level SR degradation profile."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


JPEG_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)
JPEG_CHROMA = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _dct_matrix(device):
    frequency = torch.arange(8, device=device, dtype=torch.float32)[:, None]
    position = torch.arange(8, device=device, dtype=torch.float32)[None, :]
    matrix = torch.cos(math.pi * (2 * position + 1) * frequency / 16)
    matrix[0] *= math.sqrt(1 / 8)
    matrix[1:] *= math.sqrt(2 / 8)
    return matrix


def _separable_filter(images, horizontal, vertical=None):
    vertical = horizontal if vertical is None else vertical
    batch, channels, height, width = images.shape
    kernel_size = horizontal.shape[1]
    padding = kernel_size // 2
    packed = images.reshape(1, batch * channels, height, width)
    horizontal_weight = horizontal.repeat_interleave(channels, 0)[:, None, None, :]
    packed = F.pad(packed, (padding, padding, 0, 0), mode="reflect")
    packed = F.conv2d(packed, horizontal_weight, groups=batch * channels)
    vertical_weight = vertical.repeat_interleave(channels, 0)[:, None, :, None]
    packed = F.pad(packed, (0, 0, padding, padding), mode="reflect")
    packed = F.conv2d(packed, vertical_weight, groups=batch * channels)
    return packed.reshape(batch, channels, height, width)


def _gaussian_kernels(sigmas, kernel_size=37):
    radius = kernel_size // 2
    position = torch.arange(-radius, radius + 1, device=sigmas.device,
                            dtype=torch.float32)[None]
    kernels = torch.exp(-0.5 * (position / sigmas[:, None].clamp_min(0.05)) ** 2)
    return kernels / kernels.sum(1, keepdim=True)


def _sinc_kernels(cutoffs, kernel_size=15):
    radius = kernel_size // 2
    position = torch.arange(-radius, radius + 1, device=cutoffs.device,
                            dtype=torch.float32)[None]
    kernels = torch.sinc(cutoffs[:, None] * position)
    window = torch.hann_window(kernel_size, periodic=False, device=cutoffs.device)
    kernels = kernels * window
    return kernels / kernels.sum(1, keepdim=True)


def jpeg_approximation(images, qualities):
    """GPU 8x8 DCT JPEG simulation with per-image quality and no entropy coding."""
    batch, _, height, width = images.shape
    rgb = images.clamp(0, 1) * 255
    red, green, blue = rgb.unbind(1)
    ycbcr = torch.stack((
        0.299 * red + 0.587 * green + 0.114 * blue,
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128,
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128,
    ), 1)
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        ycbcr = F.pad(ycbcr, (0, pad_width, 0, pad_height), mode="replicate")
    padded_height, padded_width = ycbcr.shape[-2:]
    blocks = ycbcr.unfold(2, 8, 8).unfold(3, 8, 8) - 128
    transform = _dct_matrix(images.device)
    coefficients = torch.einsum("iu,bchwuv,jv->bchwij", transform, blocks, transform)

    tables = torch.stack((JPEG_LUMA, JPEG_CHROMA, JPEG_CHROMA)).to(images.device)
    quality = qualities.float().clamp(1, 100)
    scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality)
    quantizers = ((tables[None] * scale[:, None, None, None] + 50) / 100)
    quantizers = quantizers.floor().clamp(1, 255)[:, :, None, None]
    coefficients = (coefficients / quantizers).round() * quantizers
    restored = torch.einsum("iu,bchwij,jv->bchwuv", transform, coefficients, transform)
    restored = restored + 128
    restored = restored.permute(0, 1, 2, 4, 3, 5).reshape(
        batch, 3, padded_height, padded_width)[:, :, :height, :width]

    y_channel, cb, cr = restored.unbind(1)
    red = y_channel + 1.402 * (cr - 128)
    green = y_channel - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    blue = y_channel + 1.772 * (cb - 128)
    return torch.stack((red, green, blue), 1).div(255).clamp(0, 1)


class GPURandomSRDegradation(nn.Module):
    """Apply uniformly sampled severity levels 0..5 to a uint8 image batch."""

    def __init__(self, levels=range(6)):
        super().__init__()
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < 0 or level > 5 for level in levels):
            raise ValueError("levels must be a non-empty sequence drawn from 0..5")
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("level_counts", torch.zeros(6, dtype=torch.long))

    def reset_stats(self):
        self.level_counts.zero_()

    def stats(self):
        return self.level_counts.detach().cpu().tolist()

    def _stage(self, images, level, second=False):
        batch = images.shape[0]
        blur_probability = min(1.0, 0.55 + 0.09 * level)
        blur_mask = torch.rand(batch, device=images.device) < blur_probability
        blur_min = 0.25 + 0.35 * level
        blur_max = 0.5 + (0.85 if second else 1.1) * level
        sigma_x = torch.empty(batch, device=images.device).uniform_(blur_min, blur_max)
        sigma_y = torch.empty(batch, device=images.device).uniform_(blur_min, blur_max)
        blurred = _separable_filter(images, _gaussian_kernels(sigma_x),
                                    _gaussian_kernels(sigma_y))
        images = torch.where(blur_mask[:, None, None, None], blurred, images)

        scale_low = max(0.1, 1.0 - (0.14 if second else 0.18) * level)
        scale_high = max(scale_low, 1.0 - (0.07 if second else 0.10) * level)
        scale = float(torch.empty((), device=images.device).uniform_(scale_low, scale_high))
        reduced_size = tuple(max(1, round(size * scale)) for size in images.shape[-2:])
        mode_index = int(torch.randint(0, 3, (), device=images.device))
        mode = ("area", "bilinear", "bicubic")[mode_index]
        if mode == "area":
            reduced = F.interpolate(images, reduced_size, mode="area")
            images = F.interpolate(reduced, images.shape[-2:], mode="bilinear",
                                   align_corners=False, antialias=True)
        else:
            reduced = F.interpolate(images, reduced_size, mode=mode,
                                    align_corners=False, antialias=True)
            images = F.interpolate(reduced, images.shape[-2:], mode=mode,
                                   align_corners=False, antialias=True)

        noise_probability = min(1.0, 0.7 + 0.06 * level)
        noise_mask = torch.rand(batch, device=images.device) < noise_probability
        noise_min = (1.0 if second else 1.5) * level
        noise_max = (6.0 if second else 8.0) * level
        sigma = torch.empty(batch, device=images.device).uniform_(noise_min, noise_max)
        gaussian = images + torch.randn_like(images) * sigma[:, None, None, None] / 255
        peak = (255 / sigma.clamp_min(0.25))[:, None, None, None]
        poisson = torch.poisson(images.clamp(0, 1) * peak) / peak
        noisy = torch.where((torch.rand(batch, device=images.device) < 0.35)[:, None, None, None],
                            poisson, gaussian)
        images = torch.where(noise_mask[:, None, None, None], noisy, images).clamp(0, 1)

        jpeg_probability = min(1.0, 0.75 + 0.05 * level)
        jpeg_mask = torch.rand(batch, device=images.device) < jpeg_probability
        quality_low = max(5, 100 - (16 if second else 14) * level)
        quality_high = max(quality_low, 100 - 7 * level)
        qualities = torch.randint(quality_low, quality_high + 1, (batch,), device=images.device)
        compressed = jpeg_approximation(images, qualities)
        return torch.where(jpeg_mask[:, None, None, None], compressed, images)

    @torch.no_grad()
    def forward(self, uint8_images):
        if uint8_images.dtype != torch.uint8 or uint8_images.ndim != 4:
            raise TypeError("GPURandomSRDegradation expects BCHW uint8 images")
        images = uint8_images.float() / 255
        choices = torch.randint(len(self.levels), (len(images),), device=images.device)
        sampled_levels = self.levels[choices]
        self.level_counts.add_(torch.bincount(sampled_levels, minlength=6))
        result = images.clone()
        for level in range(1, 6):
            indices = (sampled_levels == level).nonzero(as_tuple=True)[0]
            if not len(indices):
                continue
            degraded = self._stage(images[indices], level, second=False)
            if level >= 2:
                degraded = self._stage(degraded, level, second=True)
            sinc_probability = min(1.0, 0.35 + 0.13 * level)
            sinc_mask = torch.rand(len(indices), device=images.device) < sinc_probability
            cutoffs = torch.empty(len(indices), device=images.device).uniform_(0.35, 0.9)
            filtered = _separable_filter(degraded, _sinc_kernels(cutoffs))
            degraded = torch.where(sinc_mask[:, None, None, None], filtered, degraded)
            quality_low = max(5, 100 - 18 * level)
            quality_high = max(quality_low, 100 - 10 * level)
            qualities = torch.randint(quality_low, quality_high + 1, (len(indices),),
                                      device=images.device)
            result[indices] = jpeg_approximation(degraded, qualities)
        return (result * 255).round().clamp(0, 255).to(torch.uint8)
