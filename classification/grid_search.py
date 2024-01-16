from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from classification.nn import create_model


def grid_search_neural_networks(num_classes, X_train, y_train, param_grid):
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
    # Create GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
    # Fit the grid search to the data
    grid_result = grid.fit(X_train, y_train)
    # Summarize results
    print(f"Best: {grid_result.best_score_:.2} using {grid_result.best_params_}")
    return grid_result
