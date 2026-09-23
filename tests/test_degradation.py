import numpy as np
import pytest
from PIL import Image

from plm.degradation import (DegradationConfig, DegradationPipeline, PROFILES,
                             RandomLevelDegradationPipeline)


def sample_image():
    y, x = np.mgrid[:96, :128]
    array = np.stack((x * 2, y * 2, (x + y) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(array, "RGB")


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_preserves_mode_and_size(profile):
    image = sample_image()
    output = DegradationPipeline(DegradationConfig(profile, 3))(image, key="sample")
    assert output.mode == "RGB"
    assert output.size == image.size


def test_level_zero_is_clean_for_every_profile():
    image = sample_image()
    expected = np.asarray(image)
    for profile in PROFILES:
        actual = DegradationPipeline(DegradationConfig(profile, 0))(image, key="x")
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_deterministic_key_reproduces_composed_degradation():
    pipeline = DegradationPipeline(DegradationConfig("sr", 4, seed=123))
    first = np.asarray(pipeline(sample_image(), key="a"))
    second = np.asarray(pipeline(sample_image(), key="a"))
    np.testing.assert_array_equal(first, second)


def test_invalid_level_rejected():
    with pytest.raises(ValueError):
        DegradationConfig("jpeg", 6)


@pytest.mark.parametrize("profile", ("jpeg", "resize", "blur", "noise", "bit_depth"))
def test_extreme_atomic_level_changes_image_more_than_level_one(profile):
    image = sample_image()
    source = np.asarray(image).astype(np.float32)
    light = DegradationPipeline(DegradationConfig(profile, 1))(image, key="same")
    extreme = DegradationPipeline(DegradationConfig(profile, 5))(image, key="same")
    light_error = np.abs(np.asarray(light).astype(np.float32) - source).mean()
    extreme_error = np.abs(np.asarray(extreme).astype(np.float32) - source).mean()
    assert extreme_error > light_error


def test_random_level_pipeline_reproduces_level_for_a_key():
    pipeline = RandomLevelDegradationPipeline("jpeg", range(6), seed=11,
                                              deterministic=True)
    first = np.asarray(pipeline(sample_image(), key="same"))
    second = np.asarray(pipeline(sample_image(), key="same"))
    np.testing.assert_array_equal(first, second)


def test_random_level_pipeline_rejects_invalid_levels():
    with pytest.raises(ValueError):
        RandomLevelDegradationPipeline("sr", [])
