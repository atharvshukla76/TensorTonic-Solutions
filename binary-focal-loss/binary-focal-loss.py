def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    import numpy as np 
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps = 1e-7
    predictions = np.clip(predictions, eps, 1-eps)
    p_t = np.where(targets ==1, predictions, 1-predictions)
    focal_factor = (1-p_t)**gamma
    loss = -alpha*focal_factor*np.log(p_t)
    return float(np.mean(loss))
    