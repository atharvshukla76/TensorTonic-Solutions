import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.asarray(x)
    p = np.asarray(p)
    pmf = np.asarray(np.where(x==0, 1-p, p))
    mean = float(p)
    variance = float(p*(1-p))
    return {
        "pmf":pmf,
        "mean":mean,
        "variance":variance
    }
    