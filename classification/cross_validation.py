import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from classification.bert import run_bert
from classification.nn import (
    create_single_model,
    performance_metrics,
    preprocess_data,
    reshape,
)
from utils import write_results


def classification_pipeline(X, y, textbooks, leave_out_textbook, params):
    """Run the classification pipeline by leaving out one textbook for test data."""
    print(f"Running pipeline for leave-out-textbook {leave_out_textbook}...")
    num_classes, X_train, X_test, y_train, y_test = preprocess_data(
        X, y, textbooks, leave_out_textbook
    )
    params = {
        k.removeprefix("model__"): v
        for k, v in params.items()
        if k.startswith("model__")
    }
    model = create_single_model(num_classes, reshape(X_train), **params)
    model.fit(reshape(X_train), y_train, batch_size=params["batch_size"])

    y_pred_probabilities = model.predict(reshape(X_test))
    y_pred = np.argmax(y_pred_probabilities, axis=1)

    results = performance_metrics(y_pred, y_test)
    print(
        f"Finished running pipeline for leave-out-textbook {leave_out_textbook}: {results}"
    )
    return {"leave_out_textbook": leave_out_textbook} | results


def cross_validate(X, y, textbooks, base_textbook, params):
    """Cross-validate the classification pipeline by leaving out one textbook each time."""

    all_textbooks = (p.stem for p in Path("textbooks-parsed").glob("*"))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = (
            executor.submit(
                classification_pipeline, X, y, textbooks, leave_out_textbook, params
            )
            for leave_out_textbook in all_textbooks
            if leave_out_textbook != base_textbook
        )
        results = [future.result() for future in as_completed(futures)]
    write_results(results, "evaluation-data/classification-pipeline.jsonl")
