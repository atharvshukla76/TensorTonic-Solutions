import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    labels, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)
    candidates = labels[counts == max_count]

    for label in y_train:
        if label in candidates:
            majority_label = label
            break

    return np.full(X_test.shape[0], majority_label, dtype=int)
    