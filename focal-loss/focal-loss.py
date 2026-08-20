import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p, dtype = float)
    y = np.asarray(y)
    if not np.all(np.isin(y, [0,1])):
        raise ValueError("y must be in 0 or 1")
    
    eps = 1e-7
    p = np.clip(p, eps, 1-eps)
    p_t = y*p + (1-y)*(1-p)
    fl = -(1-p_t) ** gamma *np.log(p_t)
    return float(np.mean(fl))