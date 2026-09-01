import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.sort(np.asarray(x))
    n = len(x)
    q = np.asarray(q)
    r = (q/100)*(n-1)
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)
    w = r-l
    p = (1-w)*x[l]+w*x[u]
    return np.asarray(p)