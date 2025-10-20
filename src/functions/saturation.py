"""Functions used to model diminishing returns (saturation effects)."""

import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable


def exp(
    x: NDArray | TensorVariable,
    shape: TensorVariable,
) -> TensorVariable:
    """Apply inverse exponential saturation function.

    Parameters
    ----------
    x : NDArray | TensorVariable, shape (T, )
        The investment timeseries over the full history of a given channel.
    shape : TensorVariable
        The shape of the current channel.

    Returns
    -------
    TensorVariable
        The transformed effects between (0, 1).

    """
    return 1 - pt.exp(-x / shape)


def hill(
    x: NDArray | TensorVariable,
    half_saturation: TensorVariable,
    shape: TensorVariable,
) -> TensorVariable:
    """Apply Hill saturation function.

    Parameters
    ----------
    x : NDArray | TensorVariable, shape (T, )
        The investment timeseries over the full history of a given channel.
    half_saturation : TensorVariable
        The half saturation point of the current channel.
    shape : TensorVariable
        The shape of the current channel.

    Returns
    -------
    TensorVariable, shape (T, )
        The transformed effects between (0, 1)

    """
    return 1 / (1 + (x / half_saturation) ** (-shape))
