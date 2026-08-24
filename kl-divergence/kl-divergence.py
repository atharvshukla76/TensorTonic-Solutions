import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """Return KL divergence from p to q."""
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype = float)
    mask = p>0
    p_pos = p[mask]
    q_pos = np.maximum(q[mask], eps)
    kl = np.sum(p_pos * np.log(p_pos/q_pos))
    return float(kl)
    