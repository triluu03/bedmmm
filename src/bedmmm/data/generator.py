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

    def __init__(self: Self, n_time_periods: int, random_seed: int = 0):
        """Initialize.

        Parameters
        ----------
        n_time_periods : int
            The number of weeks for the data generation.
        random_seed : int, default 0
            The seed for the random generator.


        """
        np.random.seed(random_seed)
        self.T = n_time_periods

        self.M = 0  # Number of media channels generated
        self.C = 0  # number of control variables generated

        self.__baseline_df: pl.DataFrame = pl.DataFrame()
        self.__baseline_param: dict[Literal["baseline", "noise"], float] = {}

        self.__control_df: pl.DataFrame = pl.DataFrame()
        self.__control_params = []

        self.__media_df: pl.DataFrame = pl.DataFrame()
        self.__media_params = []

        self.__target_df: pl.DataFrame = pl.DataFrame()
        self.__target_param: dict[Literal["noise"], float] = {}

    def generate_baseline(
        self: Self,
        base: float,
        noise: float = 0,
    ) -> Self:
        """Generate baseline.

        Parameters
        ----------
        base : float
            The value of the base.
        noise : float, default 0
            The noise added to the baseline. By default, there is no noise,
            making the baseline a flat horizontal line.

        Returns
        -------
        Self
            DataGenerator with baseline included.

        """
        baseline = np.random.normal(loc=base, scale=noise, size=self.T)
        self.__baseline_df = self.__baseline_df.with_columns(
            pl.Series("baseline", baseline)
        )

        self.__baseline_param["baseline"] = base
        self.__baseline_param["noise"] = noise

        return self

    def generate_control(
        self: Self,
        control_effect: float,
        noise: float = 0,
        type: Literal["seasonality"] = "seasonality",
    ) -> Self:
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
                    loc=np.cos(2 * np.pi * (np.arange(self.T) - 12) / 52),
                    scale=noise,
                    size=self.T,
                )

                self.__control_df = self.__control_df.with_columns(
                    pl.Series(f"control_{self.C}", control_var)
                )
                self.__control_params.append(
                    {
                        "control_variable": f"control_{self.C}",
                        "control_effect": control_effect,
                    }
                )

        self.C += 1
        return self

    def generate_media(
        self: Self,
        retention_rate: float,
        saturation: float,
        shape: float,
        metric_lower_bound: float | None = None,
        noise: float = 0,
    ) -> Self:
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
        white_noise = np.random.normal(0, noise, self.T)

        media_metrics = (
            metric_lower_bound
            + control_corr * self.__control_df.to_numpy().sum(axis=1)
            + np.sqrt(1 - control_corr**2) * white_noise
        ).clip(0.0, None)

        self.__media_df = self.__media_df.with_columns(
            pl.Series(f"media_{self.M}", media_metrics)
        )
        self.__media_params.append(
            {
                "media_feature": f"media_{self.M}",
                "retention_rate": retention_rate,
                "saturation": saturation,
                "shape": shape,
                "metric_lower_bound": metric_lower_bound,
                "control_correlation": control_corr,
            }
        )

        self.M += 1

        return self

    def generate_target(self: Self, noise: float) -> Self:
        """Generate the sales target with noise included.

        Parameters
        ----------
        noise : float
            The observation noise added to the sales target.

        Returns
        -------
        Self
            DataGenerator with target included.

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

        white_noise = np.random.normal(0, noise, self.T)

        y_target = (
            np.sum(media_uplifts, axis=0)
            + np.sum(control_effects, axis=0)
            + self.__baseline_df["baseline"].to_numpy()
            + white_noise
        )

        self.__target_df = self.__target_df.with_columns(
            pl.Series("y", y_target)
        )

        self.__target_param["noise"] = noise

        return self

    def collect(
        self: Self,
    ) -> pl.DataFrame:
        """Collect the generated data.

        Returns
        -------
        dict[str, NDArray | None]
            The dictionary containing the collected data.
            If None, the data has not been generated.

        """
        return pl.concat(
            [
                self.__media_df,
                self.__control_df,
                self.__target_df,
            ],
            how="horizontal",
        )

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
