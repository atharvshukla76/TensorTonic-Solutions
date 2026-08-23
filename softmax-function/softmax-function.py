import numpy as np

def softmax(x: list) -> np.ndarray:
    """Return stable softmax probabilities with the same shape as x."""
    x = np.asarray(x)
    if x.ndim == 1:
        x_max = np.max(x)
        exp_x = np.exp(x-x_max)
        return exp_x / np.sum(exp_x)
    elif x.ndim == 2:
        x_max = np.max(x, axis = 1, keepdims = True)
        exp_x = np.exp(x-x_max)
        return  exp_x/np.sum(exp_x, axis = 1, keepdims = True)
    else:
        raise ValueError("x should be of 1D and 2D")