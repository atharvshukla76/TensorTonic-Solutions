import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distance = np.sum(np.abs(x-y))
    return float(distance)