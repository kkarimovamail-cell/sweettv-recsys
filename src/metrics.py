"""MAP@K exactly as in ml_metrics (benhamner/Metrics): AP is divided by min(len(actual), k)."""
import numpy as np


def apk(actual, predicted, k: int = 5) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # a repeated prediction counts once
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k: int = 5) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))
