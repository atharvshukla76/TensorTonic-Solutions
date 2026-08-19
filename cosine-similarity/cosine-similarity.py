import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    dot_product = np.sum(a*b)
    if a.shape != b.shape:
        raise ValueError("a and b must be of same shape")
    else:
        cosine_sim = dot_product/(a_norm*b_norm+1e-8)
    return cosine_sim