from __future__ import annotations

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def mean_adjusted_exponent_error(
    y: Sequence[float],
    y_pred: Sequence[float],
    center: float = 125,
    critical_range: float = 55,
    slope: float = 100,
    verbose: bool = False,
) -> float:
    """Calculate the Mean Adjusted Exponent Error (MADEX) for diabetes prediction.

    MADEX is a custom metric that penalizes prediction errors more heavily
    in critical glucose ranges.

    Args:
        y: Actual glucose values
        y_pred: Predicted glucose values
        center: Center point of the critical range (default: 125 mg/dL)
        critical_range: Width of the critical range (default: 55 mg/dL)
        slope: Slope parameter for error adjustment (default: 100)
        verbose: Whether to print intermediate exponent values

    Returns:
        The mean adjusted exponent error value
    """

    def exponent(
        y_hat: float,
        y_i: float,
        a: float = center,
        b: float = critical_range,
        c: float = slope,
    ) -> float:
        return float(2 - np.tanh(((y_i - a) / b)) * ((y_hat - y_i) / c))

    sum_ = 0.0
    for i in range(len(y)):
        exp = exponent(y_pred[i], y[i])
        if verbose:
            print(exp)
        # Clip base and exponent to avoid overflow
        base = min(abs(y_pred[i] - y[i]), 1e6)
        exp_clipped = min(max(exp, 0), 10)
        sum_ += float(base**exp_clipped)
    return float(sum_ / len(y))


def madex(y: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> float:
    """MADEX scorer function compatible with sklearn.

    Args:
        y: Actual values
        y_pred: Predicted values
        sample_weight: Not used, included for sklearn compatibility

    Returns:
        MADEX error value
    """
    return mean_adjusted_exponent_error(list(y), list(y_pred))


def rmadex(y: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> float:
    """Root MADEX scorer function compatible with sklearn.

    Args:
        y: Actual values
        y_pred: Predicted values
        sample_weight: Not used, included for sklearn compatibility

    Returns:
        Root MADEX error value
    """
    return float(np.sqrt(mean_adjusted_exponent_error(list(y), list(y_pred))))


def graph_vs_mse(
    value: float,
    value_range: float,
    action: Optional[str] = None,
    save_folder: str = ".",
) -> Optional[Any]:
    """Create a comparison graph between MADEX and MSE errors.

    Args:
        value: Reference value to compare against
        value_range: Range around the value to plot
        action: If 'save', saves the figure to disk; otherwise returns plt
        save_folder: Folder to save the figure in

    Returns:
        Matplotlib pyplot object if action is not 'save', None otherwise
    """
    prediction = np.arange(value - value_range, value + value_range)
    errors = []
    mse = []
    for pred in prediction:
        errors.append(mean_adjusted_exponent_error([value], [pred]))
        from sklearn.metrics import mean_squared_error

        mse.append(mean_squared_error([value], [pred]))
    plt.plot(prediction, errors, label="madex")
    plt.plot(prediction, mse, label="mse", ls="dotted")
    plt.axvline(value, label="Reference Value", color="k", ls="--")
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
