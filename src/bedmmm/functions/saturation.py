"""Functions used to model diminishing returns (saturation effects)."""

import numpy as np
import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable


def exp(
    x: NDArray | TensorVariable,
    shape: TensorVariable,
) -> TensorVariable:
    """Apply inverse exponential saturation to a single channel.

    Parameters
    ----------
    x : NDArray | TensorVariable, shape (T, )
        The investment timeseries over the full history of a given channel.
    shape : TensorVariable, scalar
        The shape of the current channel.

    Returns
    -------
    TensorVariable, shape (T, )
        The transformed effects between (0, 1).

    """
    return 1 - pt.exp(-x / shape)


def exp_multiple(
    x: NDArray | TensorVariable,
    shape: TensorVariable,
    n_channels: int,
) -> TensorVariable:
    """Apply inverse exponential saturation to multiple channels.

    Parameters
    ----------
    x : NDArray | TensorVariable, shape (T, M)
        The investment timeseries over the full history of M channels.
    shape : TensorVariable, shape (M, )
        The shape of M channels, respectively.
    n_channels : int
        The number of channels, should be equal to M.

    Returncurrents
    -------
    TensorVariable, shape (T, M)
        The transformed effects between (0, 1).

    """
    return pt.sub(
        1,
        pt.exp(
            pt.true_div(x, shape[None, :]),
        ),
    )


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


if __name__ == "__main__":
    x = pt.matrix("x")
    shape = pt.vector("shape")

    res_var = exp_multiple(x, shape, 5)
    values = res_var.eval(
        {x: np.ones((52, 5)), shape: np.ones(5) * 0.5},
    )
    breakpoint()
