from plm.metrics import average_precision, binary_metrics


def test_perfect_binary_metrics():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["average_precision"] == 1.0


def test_average_precision_uses_fake_as_positive():
    assert average_precision([1, 0], [0.9, 0.1]) == 1.0
