import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_pred and y_true must have matching dimensions ")
    true_class_probs = y_pred[np.arange(y_true.shape[0]), y_true]
    return -np.mean(np.log(true_class_probs))