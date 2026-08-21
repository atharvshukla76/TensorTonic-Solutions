def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp += 1
        else:
            fp += 1
            fn += 1
    f1 = (2*tp)/((2*tp) + fp + fn)
    return round(float(f1),4)
    