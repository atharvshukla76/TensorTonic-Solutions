import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.asarray(anchor, dtype = float)
    positive = np.asarray(positive, dtype = float)
    negative = np.asarray(negative, dtype = float)
    pos_dist = np.sum((anchor-positive)**2, axis=-1)
    neg_dist = np.sum((anchor-negative)**2, axis=-1)
    losses = np.maximum(0, pos_dist - neg_dist + margin)
    return float(np.mean(losses))