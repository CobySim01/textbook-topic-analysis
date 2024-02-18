from contextlib import redirect_stdout

from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

from classification.bert import run_bert
from classification.nn import create_model, preprocess_data, reshape

from utils import performance_metrics

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "f1": make_scorer(f1_score, average="weighted", zero_division=0),
}


def grid_search_neural_networks(num_classes, X_train, y_train, param_grid, n_splits=5):
    """Performs a grid search on neural networks."""

    # Wrap the model using KerasClassifier
    model = KerasClassifier(
        model=create_model,
        callbacks=[EarlyStopping(monitor="loss", patience=5)],
        epochs=50,
        verbose=0,
        model__num_classes=num_classes,
        model__input_shape=(1, X_train.shape[-1]),
    )

    with redirect_stdout(None):
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=n_splits,
            n_jobs=-1,
            scoring=scoring,
            refit="accuracy",
        )
    # Fit the grid search to the data
    grid_result = grid.fit(X_train, y_train)
    # Summarize results
    print(f"Best: {grid_result.best_score_:.2} using {grid_result.best_params_}")
    return grid_result


def advanced_language_model_cv(dataset, param_grid, with_concepts, n_splits=5):
    X, y = run_bert(dataset, with_concepts)
    num_classes, X_train, X_test, y_train, y_test = preprocess_data(X, y)
    best_model = grid_search_neural_networks(
        num_classes=num_classes,
        X_train=reshape(X_train),
        y_train=y_train,
        param_grid=param_grid,
        n_splits=n_splits,
    )
    y_pred = best_model.predict(reshape(X_test))
    results_summary = best_model.best_params_ | performance_metrics(y_pred, y_test)
    return results_summary, best_model.cv_results_
