import numpy as np


def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts = True)
    probabilities = counts / len(y)
    entorpy = 0.0
    
    entropy = -np.sum(probabilities*np.log2(probabilities))
    return float(entropy)
    
