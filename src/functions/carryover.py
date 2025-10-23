"""Functions used tn model carryover effects in marketing."""

import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable


def adstock(
    x: TensorVariable | NDArray,
    retention_rate: TensorVariable,
    max_length: int = 13,
) -> TensorVariable:
    """Apply adstock transformation to a single media channel.

    Parameters
    ----------
    x : TensorVariable | NDArray, of shape (T, )
        The input values (typically investments)
    retention_rate : TensorVariable, scalar
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


def adstock_multiple(
    x: TensorVariable | NDArray,
    retention_rate: TensorVariable,
    n_channels: int,
    max_length: int = 13,
) -> TensorVariable:
    """Apply adstock transformation to multiple media channels.

    Parameters
    ----------
    x : TensorVariable | NDArray, of shape (T, M)
        The input values (typically investments)
    retention_rate : TensorVariable, of shape (M, )
        The retention rate
    n_channels : int
        Number of media channels in the calculation, expected to be equal to M.
    max_length : int, default 13 (weeks)
        The maximum lookback windows

    Returns
    -------
    TensorVariable of shape (T, M)
        The adstock transformed values.

    """
    return pt.stack(
        [
            adstock(x[:, m], retention_rate[m], max_length)
            for m in range(n_channels)
        ],
        axis=1,
    )
