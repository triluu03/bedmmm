"""Bayesian Experimental Designer (BED).

The module containing the main optimization algorithm to design
the most optimal experiment.

"""

import logging
from typing import Literal, Self

import arviz as az
import numpy as np
import scipy.optimize
import scipy.stats as sps
from bedmmm.model import MarketingMixModel
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExperimentDesigner:
    """Experiment Designer."""

    def __init__(
        self: Self,
        n_media_channels: int,
        c: NDArray,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        n_media_channels : int
            The number of media channels in the model.
            This is the same as "M" defined in the MarketingMixModel class.
        c : NDArray, shape (n_control_variables, )
            The values of the control variables during the experiment time.

        """
        self.n_media_channels = n_media_channels
        self.n_control_variables = c.shape[0]
        self.c: NDArray = c
        self.l_lookback_window = 13  # L = 13 weeks

        self.regularized_type: Literal["none", "cost", "cost_and_sales"] = (
            "none"
        )
        self.x_0: NDArray
        self.regularized_weight: float

        self.target_parameters: list[
            Literal["saturation", "shape", "retention_rate"]
        ] = ["saturation", "shape", "retention_rate"]

        # Boundaries for the experiments
        self.__lower_bound: NDArray | None = None
        self.__upper_bound: NDArray | None = None

        # Posterior samples
        self.__samples_added: bool = False

        self.__retention_rates: NDArray[np.float64]
        self.__saturations: NDArray[np.float64]
        self.__shapes: NDArray[np.float64]
        self.__gamma_est: NDArray
        self.__baseline_est: float
        self.__sigma_est: float = 5.0

        # Last L weeks media data for look-back window
        # Shape (l_lookback_window - 1, n_media_channels)
        self.__last_l_week_media: NDArray[np.float64]

    def add_posterior_samples(self: Self, idata: az.InferenceData) -> Self:
        """Add posterior samples.

        Parameters
        ----------
        idata : az.InferenceData
            The inference data containing the posterior samples
            from the MarketingMixModel. It's expected to have the following:
                - posterior: the posterior samples
                - constant_data: the data used to fit the model, containing
                    the `X_media_data`.

        """
        # Step 0 (context layer): do the validations on the idata passed in
        for media_param in ["retention_rates", "saturations", "shapes"]:
            assert (
                idata.posterior.sizes.get(f"{media_param}_dim_0")
                == self.n_media_channels
            )
        for control_param in ["gammas"]:
            assert (
                idata.posterior.sizes.get(f"{control_param}_dim_0")
                == self.n_control_variables
            )

        # Step 1: figure out the values for S1 x S2 based on chains and draws
        n_samples = idata.posterior.sizes.get(
            "chain"
        ) * idata.posterior.sizes.get("draw")
        S1, S2 = calculate_optimal_sample_number(n_samples)

        # Step 2: decompose the idata into its corresponding variables
        self.__retention_rates = idata.posterior.get(
            "retention_rates"
        ).values.reshape((S1, S2, self.n_media_channels))
        self.__saturations = idata.posterior.get("saturations").values.reshape(
            (S1, S2, self.n_media_channels)
        )
        self.__shapes = idata.posterior.get("shapes").values.reshape(
            (S1, S2, self.n_media_channels)
        )

        self.__gamma_est = (
            idata.posterior.get("gammas").mean(axis=(0, 1)).values
        )
        self.__baseline_est = idata.posterior.get("baseline").mean().item()

        # Step 3: switch the flag of the class
        self.__samples_added = True

        # Step 4: save the last L weeks media data for MMM response function.
        self.__last_l_week_media = idata.constant_data.X_media_data.values[
            -(self.l_lookback_window - 1) :, :
        ]

        # Step 5: prepare the samples based on the target variables
        self.__prepare_samples_based_on_target_variables()

        return self

    def configure_experiment_space(
        self: Self, lower_bound: NDArray, upper_bound: NDArray
    ) -> Self:
        """Configure the experiment space.

        This is essentially the same as configuring the boundaries for
        the optimization problem.

        Parameters
        ----------
        lower_bound : NDArray, shape (n_media_channels, )
            The lower bound of the investments for `n_media_channels`
            for the experiments.
        upper_bound : NDArray, shape (n_media_channels, )
            The upper bound of the investments for `n_media_channels`
            for the experiments.

        """
        assert lower_bound.shape == (self.n_media_channels,)
        assert upper_bound.shape == (self.n_media_channels,)

        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

        return self

    def configure_regularization(
        self: Self,
        x_0: NDArray,
        regularized_weight: float,
        regularized_type: Literal["cost", "cost_and_sales"],
    ) -> Self:
        """Configure the regularized utility function.

        Parameters
        ----------
        x_0 : NDArray, shape (M, )
            The usual planned investments if no experiments
            are carried out.
        regularized_weight : float
            The weight to be used in the regularization.
        regularized_type : Literal['cost', 'cost_and_sales']
            Regularization type. This indicates which utility function
            to be used when finding the optimal experiment.

        """
        self.x_0 = x_0
        self.regularized_type = regularized_type
        self.regularized_weight = regularized_weight

        return self

    def configure_target_variables(
        self: Self,
        target_parameters: list[
            Literal["saturation", "shape", "retention_rate"]
        ],
    ) -> Self:
        """Configure target variables to be optimized.

        Parameters
        ----------
        target_parameters : list[str]
            The target parameters to be optimized in the optimal experiment.

        """
        self.target_parameters = target_parameters
        self.__prepare_samples_based_on_target_variables()

        return self

    def __prepare_samples_based_on_target_variables(self: Self):
        """Prepare samples based on the target variables.

        If a variable is not in the list of target variables to be optimized,
        we only keep the point estimate of that parameter. The point estimate
        is the mean of the posterior samples.

        """
        if not self.__samples_added:
            return

        if "saturation" not in self.target_parameters:
            self.__saturations = np.ones(self.__saturations.shape) * np.mean(
                self.__saturations
            )

        if "shape" not in self.target_parameters:
            self.__shapes = np.ones(self.__shapes.shape) * np.mean(
                self.__shapes
            )

        if "retention_rate" not in self.target_parameters:
            self.__retention_rates = np.ones(
                self.__retention_rates.shape
            ) * np.mean(self.__retention_rates)

    def __objective_function(self: Self, x: NDArray) -> float:
        """Calculate objective function for finding the optimal experiment.

        Parameters
        ----------
        x : NDArray, shape (n_media_channels, )
            The current investments in the experiment

        Returns
        -------
        float
            The negative utility function (or minimization).

        """
        match self.regularized_type:
            case "none":
                d = np.vstack([self.__last_l_week_media, x])
                return -self.eig(
                    d=d,
                    retention_rates=self.__retention_rates,
                    saturations=self.__saturations,
                    shapes=self.__shapes,
                    c=self.c,
                    gamma_est=self.__gamma_est,
                    baseline_est=self.__baseline_est,
                    sigma_est=self.__sigma_est,
                    random_seed=42,
                )
            case "cost":
                d = np.vstack([self.__last_l_week_media, x])
                d_0 = np.vstack([self.__last_l_week_media, self.x_0])
                return -self.cost_regularized_eig(
                    d=d,
                    d_0=d_0,
                    retention_rates=self.__retention_rates,
                    saturations=self.__saturations,
                    shapes=self.__shapes,
                    c=self.c,
                    gamma_est=self.__gamma_est,
                    baseline_est=self.__baseline_est,
                    sigma_est=self.__sigma_est,
                    regularized_weight=self.regularized_weight,
                    random_seed=42,
                )
            case "cost_and_sales":
                d = np.vstack([self.__last_l_week_media, x])
                d_0 = np.vstack([self.__last_l_week_media, self.x_0])
                return -self.cost_and_sales_regularized_eig(
                    d=d,
                    d_0=d_0,
                    retention_rates=self.__retention_rates,
                    saturations=self.__saturations,
                    shapes=self.__shapes,
                    c=self.c,
                    gamma_est=self.__gamma_est,
                    baseline_est=self.__baseline_est,
                    sigma_est=self.__sigma_est,
                    regularized_weight=self.regularized_weight,
                    random_seed=42,
                )

    def find_optimal_experiment(self: Self) -> NDArray:
        """Find the optimal experiment.

        Returns
        -------
        NDArray, shape (self.n_media_channels, )
            The optimal experiment.

        """
        if not self.__samples_added:
            raise RuntimeWarning(
                "The current `ExperimentDesigner` does not have any samples "
                "added yet. Please add the samples by running the method "
                "`add_posterior_samples()` with an `idata` object."
            )

        lb = (
            self.__lower_bound
            if self.__lower_bound is not None
            else np.ones(self.n_media_channels) * np.inf
        )
        ub = (
            self.__upper_bound
            if self.__upper_bound is not None
            else np.ones(self.n_media_channels) * np.inf
        )

        res = scipy.optimize.minimize(
            fun=self.__objective_function,
            x0=np.ones(self.n_media_channels),  # NOTE: placeholder
            method="L-BFGS-B",
            bounds=scipy.optimize.Bounds(
                lb=lb,
                ub=ub,
                keep_feasible=True,
            ),
        )

        print(res)
        return res.x

    @property
    def experiment_space(
        self: Self,
    ) -> dict[Literal["lower_bound", "upper_bound"], NDArray | None]:
        """Get the experiment space of the current ExperimentDesigner."""
        return {
            "lower_bound": self.__lower_bound,
            "upper_bound": self.__upper_bound,
        }

    @staticmethod
    def eig(
        d: NDArray,
        retention_rates: NDArray,
        saturations: NDArray,
        shapes: NDArray,
        c: NDArray,
        gamma_est: NDArray,
        baseline_est: float,
        sigma_est: float,
        random_seed: int = 0,
    ) -> float:
        """Estimate the expected information gain (EIG).

        This function estimates the EIG using Nested Monte Carlo (NMC) method.
        This function assumes that the target follows a Normal distribution.

        Parameters
        ----------
        d : NDArray, shape (L, M)
            The experiment design point, containing the investments
            of all M channels in the model along with the recorded investments
            for the last L - 1 weeks (to calculate the carryover effects).
        retention_rates : NDArray, shape (S1, S2, M)
            S samples of retention rates of all M channels.
        saturations : NDArray, shape (S1, S2, M)
            S samples of saturations of all M channels.
        shapes : NDArray, shape (S1, S2, M)
            S samples of shapes of all M channels.
        c : NDArray, shape (C, )
            The control variables values.
        gamma_est : NDArray, shape (C, )
            The point estimate of the control effects.
        baseline_est : float
            The point estimate of the baseline value.
        sigma_est : float
            The point estimate of the scale for the observational
            normal distribution.
        random_seed : int, default 0
            The random seed to draw samples for the target values.

        Returns
        -------
        float
            The estimated EIG at the design point.

        """
        np.random.seed(random_seed)

        y_samples = np.random.normal(
            loc=MarketingMixModel.response_function(
                x=d,
                retention_rates=retention_rates[:, 0, :],
                saturations=saturations[:, 0, :],
                shapes=shapes[:, 0, :],
                c=c,
                gammas=gamma_est,
                baseline=baseline_est,
            ),
            scale=sigma_est,
        )

        return np.log(
            sps.norm.pdf(
                x=y_samples,
                loc=MarketingMixModel.response_function(
                    x=d,
                    retention_rates=retention_rates[:, 0, :],
                    saturations=saturations[:, 0, :],
                    shapes=shapes[:, 0, :],
                    c=c,
                    gammas=gamma_est,
                    baseline=baseline_est,
                ),
                scale=sigma_est,
            )
            / sps.norm.pdf(
                y_samples[:, np.newaxis],
                loc=MarketingMixModel.response_function_nested_samples(
                    x=d,
                    retention_rates=retention_rates[:, 1:, :],
                    saturations=saturations[:, 1:, :],
                    shapes=shapes[:, 1:, :],
                    c=c,
                    gammas=gamma_est,
                    baseline=baseline_est,
                ),
                scale=sigma_est,
            ).mean(axis=1)
        ).mean()

    @staticmethod
    def cost_regularized_eig(
        d: NDArray,
        d_0: NDArray,
        retention_rates: NDArray,
        saturations: NDArray,
        shapes: NDArray,
        c: NDArray,
        gamma_est: NDArray,
        baseline_est: float,
        sigma_est: float,
        regularized_weight: float,
        random_seed: int = 0,
    ) -> float:
        """Estimate the cost regularized expected information gain (CREIG).

        This function estimates the EIG with a monteray regularization term.
        This function punishes the experiments with too extreme media spends.

        Parameters
        ----------
        d : NDArray, shape (L, M)
            The experiment design point, containing the investments
            of all M channels in the model along with the recorded investments
            for the last L - 1 weeks (to calculate the carryover effects).
        d_0 : NDArray, shape (L, M)
            The investments if no experiments are carried out.
        retention_rates : NDArray, shape (S1, S2, M)
            S samples of retention rates of all M channels.
        saturations : NDArray, shape (S1, S2, M)
            S samples of saturations of all M channels.
        shapes : NDArray, shape (S1, S2, M)
            S samples of shapes of all M channels.
        c : NDArray, shape (C, )
            The control variables values.
        gamma_est : NDArray, shape (C, )
            The point estimate of the control effects.
        baseline_est : float
            The point estimate of the baseline value.
        sigma_est : float
            The point estimate of the scale for the observational
            normal distribution.
        regularized_weight : float
            The weight used in regularization.
        random_seed : int, default 0
            The random seed to draw samples for the target values.

        """
        # Expected information gain (EIG)
        eig = ExperimentDesigner.eig(
            d,
            retention_rates,
            saturations,
            shapes,
            c,
            gamma_est,
            baseline_est,
            sigma_est,
            random_seed,
        )
        regularized_term = np.mean(np.abs(d - d_0))

        return eig + regularized_weight * np.log(regularized_term)

    @staticmethod
    def cost_and_sales_regularized_eig(
        d: NDArray,
        d_0: NDArray,
        retention_rates: NDArray,
        saturations: NDArray,
        shapes: NDArray,
        c: NDArray,
        gamma_est: NDArray,
        baseline_est: float,
        sigma_est: float,
        regularized_weight: float,
        random_seed: int = 0,
    ) -> float:
        """Estimate the regularized expected information gain (REIG).

        This function estimates the EIG with a monteray regularization term.

        Parameters
        ----------
        d : NDArray, shape (L, M)
            The experiment design point, containing the investments
            of all M channels in the model along with the recorded investments
            for the last L - 1 weeks (to calculate the carryover effects).
        d_0 : NDArray, shape (L, M)
            The investments if no experiments are carried out.
        retention_rates : NDArray, shape (S1, S2, M)
            S samples of retention rates of all M channels.
        saturations : NDArray, shape (S1, S2, M)
            S samples of saturations of all M channels.
        shapes : NDArray, shape (S1, S2, M)
            S samples of shapes of all M channels.
        c : NDArray, shape (C, )
            The control variables values.
        gamma_est : NDArray, shape (C, )
            The point estimate of the control effects.
        baseline_est : float
            The point estimate of the baseline value.
        sigma_est : float
            The point estimate of the scale for the observational
            normal distribution.
        regularized_weight : float
            The regularization weight.
        random_seed : int, default 0
            The random seed to draw samples for the target values.

        """
        # Expected information gain (EIG)
        eig = ExperimentDesigner.eig(
            d,
            retention_rates,
            saturations,
            shapes,
            c,
            gamma_est,
            baseline_est,
            sigma_est,
            random_seed,
        )

        regularized_term = np.mean(
            # Expected sales if the experiment is carried out
            MarketingMixModel.response_function_nested_samples(
                x=d,
                retention_rates=retention_rates,
                saturations=saturations,
                shapes=shapes,
                c=c,
                gammas=gamma_est,
                baseline=baseline_est,
            )
            -
            # Expected sales if the normal investments is carried on
            MarketingMixModel.response_function_nested_samples(
                x=d_0,
                retention_rates=retention_rates,
                saturations=saturations,
                shapes=shapes,
                c=c,
                gammas=gamma_est,
                baseline=baseline_est,
            )
        ) - np.mean(d - d_0)

        return eig + regularized_weight * regularized_term


def calculate_optimal_sample_number(n_samples: int) -> tuple[int, int]:
    """Calculate the recommended optimal sample numbers.

    The Nested Monte Carlo (NMC) method used to estimate the utility function
    depends on the samples of the parameters to be in a matrix. Let's suppose
    that the matrix has dimension S1 x S2. It's recommended that S1 ~ S2^2.

    Parameters
    ----------
    n_samples : int
        The number of posterior samples.

    Returns
    -------
    tuple[int, int]
        The values for S1 and S2.

    """
    S2 = int(np.ceil(np.cbrt(n_samples)))
    while n_samples // S2 != n_samples / S2:
        S2 += 1

    S1 = n_samples // S2

    return (S1, S2)
