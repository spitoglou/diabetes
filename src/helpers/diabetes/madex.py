from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error


def mean_adjusted_exponent_error(
    y: Sequence[float],
    y_pred: Sequence[float],
    center: float = 125,
    critical_range: float = 55,
    slope: float = 100,
    verbose: bool = False,
) -> float:
    """
    Calculate the Mean Adjusted Exponent Error (MADEX).

    This metric weights prediction errors based on glucose level criticality,
    penalizing errors more heavily in hypoglycemic ranges.

    :param y: Actual glucose values
    :param y_pred: Predicted glucose values
    :param center: Center point for exponent calculation (default 125 mg/dL)
    :param critical_range: Range for tanh scaling (default 55)
    :param slope: Slope for error scaling (default 100)
    :param verbose: If True, print exponent values during calculation
    :return: Mean adjusted exponent error value
    """

    def exponent(
        y_hat: float,
        y_i: float,
        a: float = center,
        b: float = critical_range,
        c: float = slope,
    ) -> float:
        return 2 - np.tanh(((y_i - a) / b)) * ((y_hat - y_i) / c)

    sum_: float = 0
    for i in range(len(y)):
        exp = exponent(y_pred[i], y[i])
        if verbose:
            print(exp)
        # Clip base and exponent to avoid overflow
        base: float = min(abs(y_pred[i] - y[i]), 1e6)
        exp_clipped: float = min(max(exp, 0), 10)
        sum_ += base**exp_clipped
    return sum_ / len(y)


def madex(
    y: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """
    Calculate MADEX metric for sklearn compatibility.

    :param y: Actual glucose values
    :param y_pred: Predicted glucose values
    :param sample_weight: Sample weights (unused, for sklearn API compatibility)
    :return: MADEX value
    """
    y_arr = np.asarray(y, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return mean_adjusted_exponent_error(y_arr.tolist(), y_pred_arr.tolist())


def rmadex(
    y: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """
    Calculate Root Mean Adjusted Exponent Error.

    :param y: Actual glucose values
    :param y_pred: Predicted glucose values
    :param sample_weight: Sample weights (unused, for sklearn API compatibility)
    :return: RMADEX value (square root of MADEX)
    """
    y_arr = np.asarray(y, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(
        np.sqrt(mean_adjusted_exponent_error(y_arr.tolist(), y_pred_arr.tolist()))
    )


def graph_vs_mse(
    value: float,
    value_range: float,
    action: str | None = None,
    save_folder: str = ".",
) -> Any:
    """
    Generate a comparison graph of MADEX vs MSE for a given reference value.

    :param value: Reference glucose value
    :param value_range: Range around the reference value to plot
    :param action: If "save", saves the plot to a file; otherwise returns plt
    :param save_folder: Folder to save the plot (if action="save")
    :return: matplotlib.pyplot module if action is not "save"
    """
    prediction: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.arange(
        value - value_range, value + value_range
    )
    errors: list[float] = []
    mse: list[float] = []
    for pred in prediction:
        errors.append(mean_adjusted_exponent_error([value], [pred]))
        mse.append(mean_squared_error([value], [pred]))
    plt.plot(prediction, errors, label="madex")
    plt.plot(prediction, mse, label="mse", ls="dotted")
    plt.axvline(value, label="Reference Value", color="k", ls="--")  # type: ignore[reportCallIssue]
    plt.xlabel("Predicted Value")
    plt.ylabel("Error")
    plt.title(f"{value} +- {value_range}")
    plt.legend()
    if action == "save":
        plt.savefig(f"{save_folder}/compare_vs_mse({value}+-{value_range}).png")
        plt.clf()
        return None
    else:
        return plt
