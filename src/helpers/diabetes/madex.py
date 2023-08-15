import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt



def mean_adjusted_exponent_error(y, y_pred, center=125, critical_range=55, slope=100, verbose=False):
    def exponent(y_hat: float, y_i: float, a=center, b=critical_range, c=slope) -> float:
        return 2 - np.tanh(((y_i - a) / b)) * ((y_hat - y_i) / c)
    sum_ = 0
    for i in range(len(y)):
        exp = exponent(y_pred[i], y[i])
        if verbose:
            print(exp)
        sum_ += abs((y_pred[i] - y[i])) ** exp
    return sum_ / len(y)


def madex(y, y_pred):
    return mean_adjusted_exponent_error(list(y), list(y_pred))

def rmadex(y, y_pred):
    # return 10
    return np.sqrt(mean_adjusted_exponent_error(list(y), list(y_pred)))

def graph_vs_mse(value, value_range, action=None, save_folder='.'):
    prediction = np.arange(value - value_range, value + value_range)
    errors = []
    mse = []
    for pred in prediction:
        errors.append(mean_adjusted_exponent_error([value], [pred]))
        mse.append(mean_squared_error([value], [pred]))
    plt.plot(prediction, errors, label='madex')
    plt.plot(prediction, mse, label='mse', ls='dotted')
    plt.axvline(value, label='Reference Value', color='k', ls='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Error')
    plt.title(f'{value} +- {value_range}')
    plt.legend()
    if action == 'save':
        plt.savefig(
            f'{save_folder}/compare_vs_mse({value}+-{value_range}).png')
        plt.clf()
    else:
        return plt
