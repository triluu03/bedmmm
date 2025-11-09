"""Module to generate synthetic data."""

import logging
from typing import Literal, Self

import numpy as np
import polars as pl
from bedmmm.functions.carryover import adstock_numpy
from bedmmm.functions.saturation import exp_numpy
from bedmmm.model import MarketingMixModel
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DataGenerator:
    """Synethetic data generator."""

    def __init__(
        self: Self,
        n_time_periods: int,
        n_future_periods: int = 0,
        baseline_df: pl.DataFrame = pl.DataFrame(),
        control_df: pl.DataFrame = pl.DataFrame(),
        media_df: pl.DataFrame = pl.DataFrame(),
        target_df: pl.DataFrame = pl.DataFrame(),
        baseline_param: dict[Literal["baseline", "noise"], float] = {},
        control_params: list[dict] = [],
        media_params: list[dict] = [],
        target_param: dict[Literal["noise"], float] = {},
    ):
        """Initialize.

        Parameters
        ----------
        n_time_periods : int
            The number of weeks for the data generation.
        n_future_periods : int, default 0
            The number of weeks to generate the data for the future.
            Such data is not expected to be used to fit the model, but more
            for doing experiments and causal-factual analysis.
        baseline_df : pl.DataFrame
            The generated current baseline dataframe.
        control_df : pl.DataFrame
            The generated current control dataframe.
        media_df : pl.DataFrame
            The generated current media dataframe.
        target_df : pl.DataFrame
            The generated current target dataframe.
        baseline_param : dict[str, float], default {}
            The current generated baseline parameter.
        control_params : list[dict], default []
            The current generated control parameters.
        media_params : list[dict], default []
            The current generated media parameters.
        target_param : dict[str, float], default {}
            The current generated target parameters.

        """
        self.T = n_time_periods
        self.future_T = n_future_periods

        self.M = media_df.shape[1]  # Number of media channels generated
        self.C = control_df.shape[1]  # number of control variables generated
        self.L = 13  # Max length days for the carryover effects

        # TODO: Develop validations to make sure the data contains enough
        # for self.T + self.future_T

        # Observed data
        self.__baseline_df: pl.DataFrame = baseline_df
        self.__control_df: pl.DataFrame = control_df
        self.__media_df: pl.DataFrame = media_df
        self.__target_df: pl.DataFrame = target_df

        # Generation parameters
        self.__baseline_param: dict[Literal["baseline", "noise"], float] = (
            baseline_param
        )
        self.__control_params = control_params
        self.__media_params = media_params
        self.__target_param: dict[Literal["noise"], float] = target_param

    def generate_baseline(
        self: Self,
        base: float,
        noise: float = 0,
    ) -> "DataGenerator":
        """Generate baseline.

        Parameters
        ----------
        base : float
            The value of the base.
        noise : float, default 0
            The noise added to the baseline. By default, there is no noise,
            making the baseline a flat horizontal line.

        """
        baseline = np.random.normal(
            loc=base, scale=noise, size=self.T + self.future_T
        )

        return DataGenerator(
            n_time_periods=self.T,
            n_future_periods=self.future_T,
            baseline_df=self.__baseline_df.with_columns(
                pl.Series("baseline", baseline)
            ),
            control_df=self.__control_df,
            media_df=self.__media_df,
            target_df=self.__target_df,
            baseline_param={
                **self.__baseline_param,
                "baseline": base,
                "noise": noise,
            },
            control_params=self.__control_params,
            media_params=self.__media_params,
            target_param=self.__target_param,
        )

    def generate_control(
        self: Self,
        control_effect: float,
        noise: float = 0,
        type: Literal["seasonality"] = "seasonality",
    ) -> "DataGenerator":
        """Generate the arbitrary control variable.

        Parameters
        ----------
        control_effect : float
            The effect coefficient of the control variable.
        noise : float, default 0
            The noise added to the control variable.
        type : str, default "seasonality"
            Type of control variable to generate. Supported types are:
                - seasonality

        """
        match type:
            case "seasonality":
                control_var = np.random.normal(
                    loc=np.cos(
                        2
                        * np.pi
                        * (np.arange(self.T + self.future_T) - 12)
                        / 52
                    ),
                    scale=noise,
                    size=self.T + self.future_T,
                )

                return DataGenerator(
                    n_time_periods=self.T,
                    n_future_periods=self.future_T,
                    baseline_df=self.__baseline_df,
                    control_df=self.__control_df.with_columns(
                        pl.Series(f"control_{self.C}", control_var)
                    ),
                    media_df=self.__media_df,
                    target_df=self.__target_df,
                    baseline_param=self.__baseline_param,
                    control_params=self.__control_params
                    + [
                        {
                            "control_variable": f"control_{self.C}",
                            "control_effect": control_effect,
                        }
                    ],
                    media_params=self.__media_params,
                    target_param=self.__target_param,
                )

    def generate_media(
        self: Self,
        retention_rate: float,
        saturation: float,
        shape: float,
        metric_lower_bound: float | None = None,
        noise: float = 0,
    ) -> "DataGenerator":
        """Generate the metrics of one media channel.

        Parameters
        ----------
        retention_rate : float, between 0 and 1
            The retention rate of the channel.
            Used in adstock function for carryover effects.
        saturation : float
            The saturation level of the channel.
            Used in exponential saturation function.
        shape : float
            The shape of the channel.
            Used in exponential saturation function.
        metric_lower_bound : Optional[float], default None
            The lower bound for the generated media data.
        noise : float, default 0
            The noise added to the media metrics.

        """
        if self.C == 0:
            raise ValueError(
                "There is no control variable generated yet! Please generate "
                "at least one control variable before any media data."
            )

        if metric_lower_bound is None:
            metric_lower_bound = np.random.uniform(0, 1)
        control_corr = np.random.uniform(0.0, 0.8)

        white_noise = np.random.normal(
            loc=0,
            scale=noise,
            size=self.T + self.future_T,
        )

        media_metrics = (
            metric_lower_bound
            + control_corr * self.__control_df.to_numpy().sum(axis=1)
            + np.sqrt(1 - control_corr**2) * white_noise
        ).clip(0.0, None)

        return DataGenerator(
            n_time_periods=self.T,
            n_future_periods=self.future_T,
            baseline_df=self.__baseline_df,
            control_df=self.__control_df,
            media_df=self.__media_df.with_columns(
                pl.Series(f"media_{self.M}", media_metrics)
            ),
            target_df=self.__target_df,
            baseline_param=self.__baseline_param,
            control_params=self.__control_params,
            media_params=self.__media_params
            + [
                {
                    "media_feature": f"media_{self.M}",
                    "retention_rate": retention_rate,
                    "saturation": saturation,
                    "shape": shape,
                    "metric_lower_bound": metric_lower_bound,
                    "control_correlation": control_corr,
                }
            ],
            target_param=self.__target_param,
        )

    def generate_target(self: Self, noise: float) -> "DataGenerator":
        """Generate the sales target with noise included.

        Parameters
        ----------
        noise : float
            The observation noise added to the sales target.

        """
        if not self.__target_df.is_empty():
            logger.warning(
                "A set of target data has already generated before. "
                "Overwriting it!"
            )

        media_uplifts = [
            media_param["saturation"]
            * exp_numpy(
                x=adstock_numpy(
                    x=self.__media_df[media_param["media_feature"]].to_numpy(),
                    retention_rate=media_param["retention_rate"],
                ),
                shape=media_param["shape"],
            )
            for media_param in self.__media_params
        ]

        control_effects = [
            control_param["control_effect"]
            * self.__control_df[control_param["control_variable"]].to_numpy()
            for control_param in self.__control_params
        ]

        white_noise = np.random.normal(
            loc=0,
            scale=noise,
            size=self.T + self.future_T,
        )

        y_target = (
            np.sum(media_uplifts, axis=0)
            + np.sum(control_effects, axis=0)
            + self.__baseline_df["baseline"].to_numpy()
            + white_noise
        )

        return DataGenerator(
            n_time_periods=self.T,
            n_future_periods=self.future_T,
            baseline_df=self.__baseline_df,
            control_df=self.__control_df,
            media_df=self.__media_df,
            target_df=self.__target_df.with_columns(pl.Series("y", y_target)),
            baseline_param=self.__baseline_param,
            control_params=self.__control_params,
            media_params=self.__media_params,
            target_param={
                **self.__target_param,
                "noise": noise,
            },
        )

    def collect(
        self: Self,
        data_type: Literal["observed", "future"] = "observed",
    ) -> pl.DataFrame:
        """Collect the generated data.

        Parameters
        ----------
        data_type : str
            "observed": collect the observation data.
            "future": collect the future data.

        Returns
        -------
        pl.DataFrame
            The collected data.

        """
        match data_type:
            case "observed":
                return pl.concat(
                    [
                        self.__media_df,
                        self.__control_df,
                        self.__target_df,
                    ],
                    how="horizontal",
                ).slice(offset=0, length=self.T)
            case "future":
                return pl.concat(
                    [
                        self.__media_df,
                        self.__control_df,
                        self.__target_df,
                    ],
                    how="horizontal",
                ).slice(offset=self.T)

    def get_parameters(self: Self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Get the parameters used in the data generation.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            The media and control parameters as DataFrames.

        """
        media_param_df = pl.DataFrame(
            self.__media_params,
            schema={
                "media_feature": pl.String,
                "retention_rate": pl.Float64,
                "saturation": pl.Float64,
                "shape": pl.Float64,
                "metric_lower_bound": pl.Float64,
                "control_correlation": pl.Float64,
            },
        )
        control_param_df = pl.DataFrame(
            self.__control_params,
            schema={
                "control_variable": pl.String,
                "control_effect": pl.Float64,
            },
        )
        return media_param_df, control_param_df

    def generate_new_observation(
        self: Self, x: NDArray, c: NDArray
    ) -> "DataGenerator":
        """Generate new observation given an investment level.

        If there exists any generated future data, this new observation
        will replace the future data and decrease the value of `self.future_T`
        by 1.

        Parameters
        ----------
        x : NDArray, shape (M, )
            The investments of the current week
        c : NDArray, shape (C, )
            The values of control variables.

        """
        observed_media_df = self.__media_df.slice(offset=0, length=self.T)
        last_l_week_media = observed_media_df.to_numpy()[-(self.L - 1) :, :]
        x_series = np.vstack([last_l_week_media, x])

        retention_rates, saturations, shapes = list(
            zip(
                *[
                    (
                        param["retention_rate"],
                        param["saturation"],
                        param["shape"],
                    )
                    for param in self.__media_params
                ]
            )
        )
        retention_rates = np.array(retention_rates).reshape((1, -1))
        saturations = np.array(saturations).reshape((1, -1))
        shapes = np.array(shapes).reshape((1, -1))

        gammas = np.array(
            [param["control_effect"] for param in self.__control_params]
        ).reshape((1, -1))

        baseline = np.random.normal(
            self.__baseline_param["baseline"],
            self.__baseline_param["noise"],
            size=1,
        ).item()

        white_noise = np.random.normal(
            loc=0, scale=self.__target_param["noise"], size=1
        )

        y_target = (
            MarketingMixModel.response_function(
                x_series,
                retention_rates,
                saturations,
                shapes,
                c,
                gammas,
                baseline,
            )
            + white_noise
        )

        # Generate the new data by taking the observations, new data from the experiment
        # and the generated future data with 1 row removed.
        baseline_df = (
            self.__baseline_df.slice(offset=0, length=self.T)
            .vstack(pl.DataFrame([baseline], schema=self.__baseline_df.schema))
            .vstack(self.__baseline_df.slice(offset=self.T + 1))
        )
        control_df = (
            self.__control_df.slice(offset=0, length=self.T)
            .vstack(pl.DataFrame(c, schema=self.__control_df.schema))
            .vstack(self.__control_df.slice(offset=self.T + 1))
        )
        media_df = (
            self.__media_df.slice(offset=0, length=self.T)
            .vstack(
                pl.DataFrame(x.reshape((1, -1)), schema=self.__media_df.schema)
            )
            .vstack(self.__media_df.slice(offset=self.T + 1))
        )
        target_df = (
            self.__target_df.slice(offset=0, length=self.T)
            .vstack(
                pl.DataFrame([y_target.item()], schema=self.__target_df.schema)
            )
            .vstack(self.__target_df.slice(offset=self.T + 1))
        )

        return DataGenerator(
            n_time_periods=self.T + 1,
            n_future_periods=self.future_T - 1 if self.future_T > 0 else 0,
            baseline_df=baseline_df,
            control_df=control_df,
            media_df=media_df,
            target_df=target_df,
            baseline_param=self.__baseline_param,
            control_params=self.__control_params,
            media_params=self.__media_params,
            target_param=self.__target_param,
        )
