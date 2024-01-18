import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout
from keras.models import Sequential

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

tf.experimental.numpy.experimental_enable_numpy_behavior()


def reshape(array):
    """Reshapes the given array from 3D to 2D."""
    return array.reshape(-1, 1, array.shape[-1])


def performance_metrics(y_true, y_pred):
    """Calculates performance metrics for predictions given true values."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=np.nan
        ),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=np.nan),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=np.nan),
    }


def preprocess_data(X, y, textbooks=None, leave_out=None):
    """Preprocess and perform a train-test split."""
    # Preprocess the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Preprocess the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    if textbooks is not None and leave_out is not None:
        mask = tf.equal(textbooks, leave_out)
        X_train = tf.boolean_mask(X_scaled, tf.logical_not(mask))
        y_train = tf.boolean_mask(y_encoded, tf.logical_not(mask))
        X_test = tf.boolean_mask(X_scaled, mask)
        y_test = tf.boolean_mask(y_encoded, mask)
        return num_classes, X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=2024
    )
    return num_classes, X_train, X_test, y_train, y_test


def create_model(num_classes, input_shape, units, dropout_rate, model_type):
    """Factory function for the model."""
    model = Sequential()
    model.add(model_type(units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def create_single_model(num_classes, X_train, units, dropout_rate, model_type):
    """Creates and returns a single model with specific parameters."""
    input_shape = (1, X_train.shape[-1])
    model = create_model(num_classes, input_shape, units, dropout_rate, model_type)
    return model
