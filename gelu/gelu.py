import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    from scipy.special import erf
    x = np.asarray(x, dtype = float)
    return 0.5 * x * (1+erf(x/np.sqrt(2)))
    