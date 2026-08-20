import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)
    if not np.all(np.isin(y, [0,1])):
        raise ValueError("y must be o or 1")
    if reduction not in("mean", "sum"):
        raise ValueError("reduction must be mean or sum")
    distance = np.linalg.norm(a-b, axis=-1)
    loss = (y*distance**2 + (1-y)*np.maximum(0, margin - distance)**2)
    if reduction == "mean":
        loss = float(np.mean(loss))
    elif reduction == "sum":
        loss  = float(np.sum(loss))
    return loss
    
    