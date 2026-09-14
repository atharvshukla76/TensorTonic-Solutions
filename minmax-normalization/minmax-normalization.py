import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X)
    X_max = np.max(X, axis = axis, keepdims = True)
    X_min = np.min(X, axis = axis, keepdims = True)

    range = X_max - X_min
    result = np.divide(X - X_min,
                       range,
                      out = np.zeros_like(X, dtype=float),
                      where = range > eps)
    return result