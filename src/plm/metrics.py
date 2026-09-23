import numpy as np


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= 0.5
    real = labels == 0
    fake = labels == 1
    real_accuracy = float((predictions[real] == 0).mean())
    fake_accuracy = float((predictions[fake] == 1).mean())
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (real_accuracy + fake_accuracy) / 2,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "average_precision": average_precision(labels, probabilities),
        "images": int(len(labels)),
    }
