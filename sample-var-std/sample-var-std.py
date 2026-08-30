import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x)
    n = len(x)
    Center_the_observation = x-np.mean(x)
    variance = float((np.sum(Center_the_observation**2)) / (n-1))
    standard_deviation = float(np.sqrt(variance))
    return {
        "variance":variance,
        "standard_deviation":standard_deviation
    }