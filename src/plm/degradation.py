"""Deterministic, severity-controlled image degradations.

The pipeline operates on RGB PIL images before PLM cropping and pixel mapping.
Levels are integers from 0 (clean) through 5 (extreme).  Atomic profiles isolate
one failure mode; composed profiles approximate image processing in the wild.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter


PROFILES = (
    "clean", "jpeg", "webp", "resize", "blur", "noise", "sharpen",
    "color", "bit_depth", "mixed", "sr", "social", "screenshot",
)


def _resampling(name: str):
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,
    }[name]


def _stable_seed(seed: int, key: object) -> int:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _encode(image: Image.Image, codec: str, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    kwargs = {"quality": int(quality)}
    if codec == "JPEG":
        kwargs.update(subsampling=2, optimize=False)
    image.save(buffer, codec, **kwargs)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resize_roundtrip(image: Image.Image, scale: float, method: str) -> Image.Image:
    original = image.size
    reduced = (max(1, round(original[0] * scale)), max(1, round(original[1] * scale)))
    if reduced == original:
        return image
    interpolation = _resampling(method)
    return image.resize(reduced, interpolation).resize(original, interpolation)


def _noise(image: Image.Image, sigma: float, rng: random.Random, poisson=False) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    generator = np.random.default_rng(rng.getrandbits(64))
    if poisson:
        peak = max(2.0, 255.0 / max(sigma, 0.25))
        array = generator.poisson(np.clip(array, 0, 255) / 255.0 * peak) / peak * 255.0
    else:
        array += generator.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), "RGB")


def _anisotropic_blur(image: Image.Image, sigma_x: float, sigma_y: float,
                      angle: float) -> Image.Image:
    radius = max(1, math.ceil(3 * max(sigma_x, sigma_y)))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    theta = math.radians(angle)
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    yr = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = torch.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    kernel /= kernel.sum()
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0) / 255.0
    weight = kernel.expand(3, 1, -1, -1)
    tensor = F.pad(tensor, (radius,) * 4, mode="reflect")
    tensor = F.conv2d(tensor, weight, groups=3)
    result = (tensor[0].permute(1, 2, 0) * 255).round().clamp(0, 255).byte().numpy()
    return Image.fromarray(result, "RGB")


def _sinc_filter(image: Image.Image, cutoff: float) -> Image.Image:
    # Windowed separable sinc: useful for ringing/overshoot seen after resampling.
    radius = 7
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.sinc(cutoff * x) * torch.hann_window(2 * radius + 1, periodic=False)
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0) / 255.0
    tensor = F.pad(tensor, (radius,) * 4, mode="reflect")
    result = F.conv2d(tensor, kernel.expand(3, 1, -1, -1), groups=3)
    array = (result[0].permute(1, 2, 0) * 255).round().clamp(0, 255).byte().numpy()
    return Image.fromarray(array, "RGB")


@dataclass(frozen=True)
class DegradationConfig:
    profile: str = "clean"
    level: int = 0
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}; choose from {PROFILES}")
        if not 0 <= self.level <= 5:
            raise ValueError("degradation level must be an integer from 0 to 5")


class DegradationPipeline:
    """Apply a degradation profile at a controlled level.

    With deterministic=True, ``key`` identifies a sample and reproduces exactly
    the same random parameters across workers and runs.  With deterministic=False,
    worker-local Python RNG state supplies fresh training augmentations.
    """

    def __init__(self, config: DegradationConfig):
        self.config = config

    def _rng(self, key: object | None):
        if self.config.deterministic:
            if key is None:
                raise ValueError("A stable sample key is required in deterministic mode")
            return random.Random(_stable_seed(self.config.seed, key))
        return random

    def __call__(self, image: Image.Image, key: object | None = None) -> Image.Image:
        image = image.convert("RGB")
        if self.config.profile == "clean" or self.config.level == 0:
            return image.copy()
        rng = self._rng(key)
        level = self.config.level
        profile = self.config.profile
        if profile in {"jpeg", "webp", "resize", "blur", "noise", "sharpen",
                       "color", "bit_depth"}:
            return self._atomic(image, profile, level, rng)
        if profile == "mixed":
            return self._mixed(image, level, rng)
        if profile == "sr":
            return self._super_resolution(image, level, rng)
        if profile == "social":
            return self._social(image, level, rng)
        if profile == "screenshot":
            return self._screenshot(image, level, rng)
        raise AssertionError(profile)


    def _atomic(self, image, operation, level, rng):
        fraction = level / 5.0
        if operation == "jpeg":
            return _encode(image, "JPEG", (100, 90, 70, 45, 25, 8)[level])
        if operation == "webp":
            return _encode(image, "WEBP", (100, 90, 65, 40, 20, 5)[level])
        if operation == "resize":
            scale = (1.0, 0.8, 0.6, 0.4, 0.25, 0.125)[level]
            method = ("bilinear", "bicubic", "area", "lanczos")[rng.randrange(4)]
            return _resize_roundtrip(image, scale, method)
        if operation == "blur":
            return image.filter(ImageFilter.GaussianBlur((0, 0.8, 1.8, 3.0, 5.0, 8.0)[level]))
        if operation == "noise":
            return _noise(image, (0, 5, 12, 22, 35, 55)[level], rng)
        if operation == "sharpen":
            return image.filter(ImageFilter.UnsharpMask(radius=1 + 4 * fraction,
                                                        percent=100 * level, threshold=1))
        if operation == "color":
            # Apply deterministic but randomly signed exposure, contrast, saturation and gamma shifts.
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast,
                             ImageEnhance.Color):
                signed = -1 if rng.random() < 0.5 else 1
                image = enhancer(image).enhance(max(0.15, 1 + signed * 0.75 * fraction))
            array = np.asarray(image, dtype=np.float32) / 255.0
            gamma = max(0.25, 1 + (-1 if rng.random() < 0.5 else 1) * fraction)
            return Image.fromarray(np.clip(np.rint(array ** gamma * 255), 0, 255).astype(np.uint8))
        if operation == "bit_depth":
            bits = (8, 7, 6, 5, 4, 2)[level]
            step = 2 ** (8 - bits)
            array = np.asarray(image, dtype=np.uint8)
            return Image.fromarray((array // step * step).astype(np.uint8), "RGB")
        raise AssertionError(operation)

    def _mixed(self, image, level, rng):
        operations = ["jpeg", "resize", "blur", "noise", "sharpen", "color", "bit_depth"]
        rng.shuffle(operations)
        count = min(len(operations), 1 + level)
        for operation in operations[:count]:
            local_level = rng.randint(max(1, level - 1), min(5, level + 1))
            image = self._atomic(image, operation, local_level, rng)
        return image

    def _sr_stage(self, image, level, rng, second=False):
        blur_probability = min(1.0, 0.55 + 0.09 * level)
        if rng.random() < blur_probability:
            blur_min = 0.25 + 0.35 * level
            blur_max = 0.5 + (0.85 if second else 1.1) * level
            sigma_x = rng.uniform(blur_min, blur_max)
            sigma_y = rng.uniform(blur_min, blur_max)
            image = _anisotropic_blur(image, sigma_x, sigma_y, rng.uniform(0, 180))
        scale_low = max(0.1, 1.0 - (0.14 if second else 0.18) * level)
        scale_high = max(scale_low, 1.0 - (0.07 if second else 0.10) * level)
        scale = rng.uniform(scale_low, scale_high)
        method = rng.choice(("area", "bilinear", "bicubic"))
        image = _resize_roundtrip(image, scale, method)
        if rng.random() < min(1.0, 0.7 + 0.06 * level):
            noise_min = (1.0 if second else 1.5) * level
            noise_max = (6.0 if second else 8.0) * level
            sigma = rng.uniform(noise_min, noise_max)
            image = _noise(image, sigma, rng, poisson=rng.random() < 0.35)
        if rng.random() < min(1.0, 0.75 + 0.05 * level):
            quality_low = max(5, 100 - (16 if second else 14) * level)
            quality_high = max(quality_low, 100 - 7 * level)
            image = _encode(image, "JPEG", rng.randint(quality_low, quality_high))
        return image

    def _super_resolution(self, image, level, rng):
        image = self._sr_stage(image, level, rng, second=False)
        if level >= 2:
            image = self._sr_stage(image, level, rng, second=True)
        if rng.random() < min(1.0, 0.35 + 0.13 * level):
            image = _sinc_filter(image, rng.uniform(0.35, 0.9))
        # Final codec models storage after restoration/upscaling.
        quality_low = max(5, 100 - 18 * level)
        quality_high = max(quality_low, 100 - 10 * level)
        return _encode(image, "JPEG", rng.randint(quality_low, quality_high))

    def _social(self, image, level, rng):
        scale = (1.0, 0.85, 0.65, 0.45, 0.3, 0.18)[level]
        image = _resize_roundtrip(image, scale, rng.choice(("bilinear", "bicubic", "lanczos")))
        if level >= 2:
            image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=20 * level, threshold=3))
        codec = "WEBP" if rng.random() < 0.35 else "JPEG"
        return _encode(image, codec, (100, 88, 70, 50, 30, 10)[level])

    def _screenshot(self, image, level, rng):
        # Fractional display scaling followed by recapture-like noise and compression.
        scale = (1.0, 0.9, 0.75, 0.55, 0.35, 0.2)[level]
        image = _resize_roundtrip(image, scale, "bilinear")
        if level >= 2:
            image = _noise(image, 2.0 * level, rng)
        return _encode(image, "JPEG", (100, 90, 75, 55, 35, 15)[level])


class RandomLevelDegradationPipeline:
    """Sample a degradation level independently for every training image."""

    def __init__(self, profile="sr", levels=range(6), seed=42, deterministic=False):
        self.profile = profile
        self.levels = tuple(int(level) for level in levels)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}")
        if not self.levels or any(level < 0 or level > 5 for level in self.levels):
            raise ValueError("levels must be a non-empty sequence drawn from 0..5")

    def __call__(self, image: Image.Image, key: object | None = None) -> Image.Image:
        if self.deterministic:
            if key is None:
                raise ValueError("A stable sample key is required in deterministic mode")
            level_rng = random.Random(_stable_seed(self.seed, f"level:{key}"))
            level = level_rng.choice(self.levels)
        else:
            level = random.choice(self.levels)
        config = DegradationConfig(
            profile=self.profile, level=level, seed=self.seed,
            deterministic=self.deterministic,
        )
        return DegradationPipeline(config)(image, key=key)
