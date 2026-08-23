import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Return the mean multiclass cross-entropy loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype = float)
    correct_probs = y_pred[np.arange(len(y_true)), y_true]
    loss = - np.log(correct_probs)
    return float(np.mean(loss))