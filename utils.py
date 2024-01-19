import json
from typing import Iterable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from numpy import nan


def write_results(results: Iterable, filename: str):
    """Write results to the given file."""
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def performance_metrics(y_true, y_pred):
    """Calculates performance metrics for predictions given true values."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=nan
        ),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=nan),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=nan),
    }
