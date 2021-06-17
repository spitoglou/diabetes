import numpy as np


def mean_adjusted_exponent_error(y_true, y_pred, center=125, critical_range=55, slope=100, verbose=False):
    def exponent(y_hat: float, y_i: float, a=center, b=critical_range, c=slope) -> float:
        return 2 - np.tanh(((y_i - a) / b)) * ((y_hat - y_i) / c)
    sum_ = 0
    for i in range(len(y_true)):
        exp = exponent(y_pred[i], y_true[i])
        if verbose:
            print(exp)
        sum_ += abs((y_pred[i] - y_true[i])) ** exp
    return sum_ / len(y_true)
