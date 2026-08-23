def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    import numpy as np
    
    k = len(predictions)
    q = np.full(k, epsilon/k)
    q[target] = (1-epsilon) + (epsilon/k)
    predictions = np.clip(predictions, 1e-7, 1.0)
    loss = -np.sum(q*np.log(predictions))
    return loss
        