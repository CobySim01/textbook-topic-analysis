from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_neural_networks(num_classes, X_train, y_train, param_grid):
    """Performs a grid search on neural networks."""

    def create_model(units, dropout_rate, model_type):
        """Factory function for the model."""
        model = Sequential()
        model.add(model_type(units, input_shape=(1, X_train.shape[-1])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    # Wrap the model using KerasClassifier
    model = KerasClassifier(
        model=create_model,
        callbacks=[EarlyStopping(monitor="loss", patience=5)],
        epochs=50,
        verbose=0,
    )
    # Create GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
    # Fit the grid search to the data
    grid_result = grid.fit(X_train, y_train)
    # Summarize results
    print(f"Best: {grid_result.best_score_:.2} using {grid_result.best_params_}")
    return grid_result
