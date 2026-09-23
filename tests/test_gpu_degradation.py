import torch

from plm.gpu_degradation import GPURandomSRDegradation, jpeg_approximation


def test_jpeg_approximation_preserves_shape_and_range():
    images = torch.rand(3, 3, 19, 23)
    result = jpeg_approximation(images, torch.tensor([95, 50, 10]))
    assert result.shape == images.shape
    assert torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_level_zero_is_exact_identity():
    images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    pipeline = GPURandomSRDegradation([0])
    result = pipeline(images)
    torch.testing.assert_close(result, images, rtol=0, atol=0)
    assert pipeline.stats() == [4, 0, 0, 0, 0, 0]


def test_random_gpu_pipeline_returns_uint8_and_counts_levels():
    images = torch.randint(0, 256, (12, 3, 32, 32), dtype=torch.uint8)
    pipeline = GPURandomSRDegradation(range(6))
    result = pipeline(images)
    assert result.dtype == torch.uint8
    assert result.shape == images.shape
    assert sum(pipeline.stats()) == 12
