import numpy as np
import torch

from plm.mapping import FixedPixelMapping, fixed_mapping_lut, random_pixel_mapping


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
