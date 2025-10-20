"""Functions used tn model carryover effects in marketing."""

import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable


def adstock(
    x: TensorVariable | NDArray,
    retention_rate: TensorVariable,
    max_length: int = 13,
) -> TensorVariable:
    """Apply adstock transformation.

    Parameters
    ----------
    x : TensorVariable | NDArray
        The input values (typically investments)
    retention_rate : TensorVariable
        The retention rate
    max_length : int, default 13 (weeks)
        The maximum lookback windows

    Returns
    -------
    TensorVariable
        The adstock transformed values.

    """
    x_cycle = pt.stack(
        [
            pt.concatenate([pt.zeros(i), x[: x.shape[0] - i]])
            for i in range(max_length)
        ]
    )
    w = pt.as_tensor_variable(
        [pt.power(retention_rate, i) for i in range(max_length)]
    )
    return pt.dot(w, x_cycle)
