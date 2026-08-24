import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """Return the coefficient of determination."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype = float)
    p_res = np.sum((y_true - y_pred)**2)
    p_total = np.sum((y_true - np.mean(y_true))**2)
    if p_total == 0:
        return 1.0 if p_res == 0 else 0.0
    return float(1-(p_res)/(p_total))