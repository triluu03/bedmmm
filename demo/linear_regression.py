"""Demo in Bayesian experimental design (BED) using linear regression."""

import arviz as az
import numpy as np
import plotly.express as px
import scipy.stats as sps
from numpy.typing import NDArray
from pymc import HalfCauchy, Model, Normal, sample


def generate_toy_data() -> tuple[NDArray, NDArray]:
    """Generate toy data.

    The data is generated from a simple linear regression model with
    the true intercept of 1 and slope of 2, with added Gaussian noise.

    Returns
    -------
    tuple[NDArray, NDArray]
        The generated toy data.

    """
    np.random.seed(42)

    size = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + np.random.normal(loc=0, scale=0.5, size=size)

    # Plot the generated data
    figure = px.scatter(x=x, y=y, title="Generated Data")
    figure.add_trace(
        px.line(
            x=x,
            y=true_regression_line,
            title="True Regression Line",
        ).data[0],
    )
    figure.show()

    return x, y


def fit_and_draw_samples(x: NDArray, y: NDArray) -> az.InferenceData:
    """Draw posterior samples from a Bayesian linear regression.

    Parameters
    ----------
    x, y : NDArray
        The toy data.

    Returns
    -------
    InferenceData
        The posterior samples.

    """
    with Model() as _model:
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("Intercept", 0, sigma=20)
        slope = Normal("slope", 0, sigma=20)

        _likelihood = Normal(
            "y", mu=intercept + slope * x, sigma=sigma, observed=y
        )

        idata = sample(
            draws=6000,
            tune=2000,
            chains=4,
            cores=4,
            random_seed=42,
            progressbar=True,
        )

        return idata


def expected_information_gain(
    x: float,
    slope_samples: NDArray[np.float64],
    intercept_est: float,
    sigma_est: float,
) -> float:
    """Estimate the expected information gain (EIG).

    Parameters
    ----------
    x : float
        The experiment design point.
    slope_samples : NDArray[np.float64] of shape (N, M)
        The posterior samples of the slope.
    intercept_est : float
        The point estimate of the intercept.
    sigma_est : float
        The point estimate of the noise.

    Returns
    -------
    float
        The estimated EIG at the design point.

    """
    np.random.seed(42)

    y_samples = np.random.normal(
        loc=slope_samples[:, 0] * x + intercept_est, scale=sigma_est
    )

    return np.log(
        sps.norm.pdf(
            x=y_samples,
            loc=slope_samples[:, 0] * x + intercept_est,
            scale=sigma_est,
        )
        / sps.norm.pdf(
            y_samples[:, np.newaxis],
            loc=slope_samples[:, 1:] * x + intercept_est,
            scale=sigma_est,
        ).mean(axis=1)
    ).mean()
