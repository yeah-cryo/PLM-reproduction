import numpy as np
import torch

from plm.mapping import (FixedPixelMapping, SmoothPixelMapping,
                         build_pixel_mapping, fixed_mapping_lut,
                         random_pixel_mapping, smooth_mapping_lut)


def test_fixed_mapping_matches_equation_four_exactly():
    values = np.arange(256, dtype=np.float64)
    expected = values - np.round(values / 256.0, 2) * 256.0
    np.testing.assert_array_equal(fixed_mapping_lut().numpy(), expected.astype(np.float32))


def test_fixed_mapping_indexes_uint8_rgb():
    image = torch.tensor([[[[0, 1, 2, 255]]]], dtype=torch.uint8).expand(1, 3, 1, 4)
    actual = FixedPixelMapping()(image)
    assert actual.shape == image.shape
    torch.testing.assert_close(actual[0, 0], fixed_mapping_lut()[image[0, 0].long()])


def test_random_mapping_shape_and_range():
    image = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    mapped = random_pixel_mapping(image, torch.Generator().manual_seed(1))
    assert mapped.shape == image.shape
    assert mapped.min() >= -1 and mapped.max() <= 1


def test_smooth_mapping_is_seeded_and_bounded():
    first = smooth_mapping_lut(spacing=16, seed=7)
    second = smooth_mapping_lut(spacing=16, seed=7)
    different = smooth_mapping_lut(spacing=16, seed=8)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)
    assert first.shape == (3, 256)
    assert first.min() >= -1 and first.max() <= 1


def test_smooth_mapping_obeys_lipschitz_bound():
    spacing = 16
    lut = smooth_mapping_lut(spacing=spacing, seed=42)
    adjacent_change = (lut[:, 1:] - lut[:, :-1]).abs()
    assert adjacent_change.max() <= 2.0 / spacing + 1e-6


def test_smooth_mapping_linearly_interpolates_control_points():
    spacing = 16
    lut = smooth_mapping_lut(spacing=spacing, seed=42)
    midpoint = spacing // 2
    expected = (lut[:, 0] + lut[:, spacing]) / 2
    torch.testing.assert_close(lut[:, midpoint], expected)


def test_smooth_mapping_indexes_each_rgb_channel():
    image = torch.tensor([[[[0, 16]], [[32, 48]], [[64, 80]]]], dtype=torch.uint8)
    mapping = SmoothPixelMapping(spacing=16, seed=42, per_channel=True)
    actual = mapping(image)
    assert actual.shape == image.shape
    for channel in range(3):
        torch.testing.assert_close(
            actual[0, channel], mapping.lut[channel, image[0, channel].long()]
        )


def test_mapping_factory_preserves_fixed_default_and_builds_smooth():
    assert isinstance(build_pixel_mapping(), FixedPixelMapping)
    mapping = build_pixel_mapping({"type": "smooth", "spacing": 64, "seed": 3})
    assert isinstance(mapping, SmoothPixelMapping)
    assert mapping.spacing == 64
