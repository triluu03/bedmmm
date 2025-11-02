"""Module to generate synthetic data."""

import logging
from typing import Literal, Self

import numpy as np
import polars as pl
from bedmmm.functions.carryover import adstock_numpy
from bedmmm.functions.saturation import exp_numpy

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

        self._baseline_df: pl.DataFrame = pl.DataFrame()

        self._control_df: pl.DataFrame = pl.DataFrame()
        self._control_params = []

        self._media_df: pl.DataFrame = pl.DataFrame()
        self._media_params = []

        self._target_df: pl.DataFrame = pl.DataFrame()

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
        self._baseline_df = self._baseline_df.with_columns(
            pl.Series("baseline", baseline)
        )
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

                self._control_df = self._control_df.with_columns(
                    pl.Series(f"control_{self.C}", control_var)
                )
                self._control_params.append(
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
        noise : float, default 0
            The noise added to the media metrics.

        """
        if self.C == 0:
            raise ValueError(
                "There is no control variable generated yet! Please generate "
                "at least one control variable before any media data."
            )

        metric_lb = np.random.uniform(0, 1)
        control_corr = np.random.uniform(0.0, 0.8)
        white_noise = np.random.normal(0, noise, self.T)

        media_metrics = (
            metric_lb
            + control_corr * self._control_df.to_numpy().sum(axis=1)
            + np.sqrt(1 - control_corr**2) * white_noise
        ).clip(0.0, None)

        self._media_df = self._media_df.with_columns(
            pl.Series(f"media_{self.M}", media_metrics)
        )
        self._media_params.append(
            {
                "media_feature": f"media_{self.M}",
                "retention_rate": retention_rate,
                "saturation": saturation,
                "shape": shape,
                "metric_lower_bound": metric_lb,
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
        if not self._target_df.is_empty():
            logger.warning(
                "A set of target data has already generated before. "
                "Overwriting it!"
            )

        media_uplifts = [
            media_param["saturation"]
            * exp_numpy(
                x=adstock_numpy(
                    x=self._media_df[media_param["media_feature"]].to_numpy(),
                    retention_rate=media_param["retention_rate"],
                ),
                shape=media_param["shape"],
            )
            for media_param in self._media_params
        ]

        control_effects = [
            control_param["control_effect"]
            * self._control_df[control_param["control_variable"]].to_numpy()
            for control_param in self._control_params
        ]

        white_noise = np.random.normal(0, noise, self.T)

        y_target = (
            np.sum(media_uplifts, axis=0)
            + np.sum(control_effects, axis=0)
            + self._baseline_df["baseline"].to_numpy()
            + white_noise
        )

        self._target_df = self._target_df.with_columns(
            pl.Series("y", y_target)
        )

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
                self._media_df,
                self._control_df,
                self._target_df,
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
            self._media_params,
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
            self._control_params,
            schema={
                "control_variable": pl.String,
                "control_effect": pl.Float64,
            },
        )
        return media_param_df, control_param_df
