from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


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
        mask = textbooks == leave_out
        X_train = X[mask]
        y_train = y[mask]
        X_test = X[~mask]
        y_test = y[~mask]
        return num_classes, X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=2024
    )
    return num_classes, X_train, X_test, y_train, y_test
