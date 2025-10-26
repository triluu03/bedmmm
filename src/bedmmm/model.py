"""Marketing Mix Modeling (MMM) in PyMC."""

import warnings
from typing import Any, override

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from bedmmm.functions import carryover, saturation
from pymc.util import RandomState
from pymc_extras.model_builder import ModelBuilder


class MarketingMixModel(ModelBuilder):
    """Marketing Mix Model."""

    _model_type = "MarketingMixModel"
    version = "0.1.0"

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> None:
        """Build the PyMC model for the Marketing Mix Model.

        Parameters
        ----------
        X : pd.DataFrame
            The features dataframe containing:
                - Media features with column name: `media_feature_{m}`.
                - Control variables with column name: `control_{c}`.
            where:
                - `m`: is the m-th media channel.
                - `c`: is the c-th control variable.
        y : pd.Series
            The target sales series

        """
        self._generate_and_preprocess_model_data(X, y)

        X_values = self.X  # Features array of shape: (T, (M + C))
        y_values = self.y  # Target array of shape: (T, )

        X_media_metrics = X_values[:, : self.M]
        X_control_metrics = X_values[:, self.M :]

        with pm.Model(coords=self.model_coords) as self.model:
            # Data and observations
            X_media_metrics = pm.MutableData(
                "X_media_metrics", X_media_metrics
            )
            X_control_metrics = pm.MutableData(
                "X_control_metrics", X_control_metrics
            )
            y_target = pm.MutableData("y_target", y_values)

            # Media priors
            # Retention rates
            retention_rate_alphas = np.array(
                [
                    self.model_config.get(
                        f"retention_rate_alpha_{m}"
                        for m in range(1, self.M + 1)
                    )
                ]
            )
            retention_rate_betas = np.array(
                [
                    self.model_config.get(
                        f"retention_rate_beta_{m}"
                        for m in range(1, self.M + 1)
                    )
                ]
            )
            retention_rates = pm.Beta(
                name="retention_rates",
                alpha=retention_rate_alphas,
                beta=retention_rate_betas,
            )

            # Shape of saturation function
            shape_sigmas = np.array(
                [
                    self.model_config.get(f"shape_sigma_{m}")
                    for m in range(1, self.M + 1)
                ]
            )
            shapes = pm.HalfNormal(name="shapes", sigma=shape_sigmas)

            # Saturation priors
            saturation_mus = np.array(
                [
                    self.model_config.get(
                        f"saturation_mu_{m}" for m in range(1, self.M + 1)
                    )
                ]
            )
            saturation_sigmas = np.array(
                [
                    self.model_config.get(
                        f"saturation_sigma_{m}" for m in range(1, self.M + 1)
                    )
                ]
            )
            saturations = pm.Normal(
                name="saturations",
                mu=saturation_mus,
                sigma=saturation_sigmas,
            )

            # Calculate the media uplifts
            media_uplifts = pt.sum(
                pt.stack(
                    [
                        saturations[m]
                        * saturation.exp(
                            carryover.adstock(
                                X_media_metrics[:, m], retention_rates[m]
                            ),
                            shapes[m],
                        )
                        for m in range(self.M)
                    ]
                ),
                axis=0,
            )

            # Control parameters priors
            gamma_sigmas = np.array(
                [
                    self.model_config.get(f"gamma_sigma_{c}")
                    for c in range(1, self.C)
                ]
            )
            gammas = pm.Normal(
                "gammas", mu=0, sigma=gamma_sigmas, shape=gamma_sigmas.shape
            )
            control_effects = pt.dot(X_control_metrics, gammas)

            # Baseline priors
            baseline = pm.Normal("baseline", mu=4, sigma=0.05)

            # Likelihood
            obs_sigma = self.model_config.get("obs_sigma", 0.1)
            _likelihood = pm.Normal(
                "y",
                mu=media_uplifts + control_effects + baseline,
                sigma=obs_sigma,
                observed=y_target,
            )

    def _data_setter(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> None:
        """Set the data.

        Parameters
        ----------
        X : pd.DataFrame
            The features dataframe containing:
                - Media features with column name: `media_feature_{m}_{g}`.
                - Control variables with column name: `control_{c}_{g}`.
            where:
                - `m`: is the m-th media channel.
                - `g`: is the g-th geo-location.
        y : pd.DataFrame
            The target sales dataframe containing:
                - The sales of g-th geo-location with column name: `y_{g}`.

        """
        with self.model:
            pm.set_data({"x_data": X.values})

            if y is not None:
                pm.set_data({"y_data": y.values})

    @staticmethod
    def get_default_model_config():
        """Get the default model config."""
        raise RuntimeError(
            "MarketingMixModel has been initialized without a model config. "
            "A default config is not available for this model. Please specify "
            "your config explicitly!"
        )

    @staticmethod
    def get_default_sampler_config(self) -> dict:
        """Get default sampler config."""
        return {
            "draws": 2_000,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.95,
        }

    @property
    def output_var(self):
        """Get the output variable name."""
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        """Serialize the model config."""
        return self.model_config

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Generate and preprocess the model data.

        Set the following attributes in the class:
            - self.model_coords = None
            - self.X : NDArray, shape (T, (M + C))
                All features including media and control.
            - self.y : NDArray, shape (T, )
                All the sales target.
            - self.M : int
                The total number of media channels.
            - self.C : int
                The total number of control variables.

        Parameters
        ----------
        X : pd.DataFrame
            The features dataframe containing:
                - Media features with column name: `media_feature_{m}`.
                - Control variables with column name: `control_{c}`.
            where:
                - `m`: is the m-th media channel, starts with 1.
                - `c`: is the c-th control variable, starts with 1.
        y : pd.Series
            The target sales series

        """
        self.model_coords = None

        self.X = X.values
        self.y = y.values

        self.M = X.columns.str.match(r"^media_feature_\d+$").sum().item()
        self.C = X.columns.str.match(r"^control_\d+$").sum().item()

    @override
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progressbar: bool = True,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit a model using the data passed as a parameter.

        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : pd.DataFrame
            The features dataframe containing:
                - Media features with column name: `media_feature_{m}_{g}`.
                - Control variables with column name: `control_{c}`.
            where:
                - `m`: is the m-th media channel.
                - `c`: is the c-th control variable.
        y : pd.Series
            The target sales series
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        random_seed : RandomState
            Random seed for the sampler.
        **kwargs : Any
            Custom sampler settings.

        Returns
        -------
        az.InferenceData
            Inference data of the fitted model.

        """
        self.build_model(X, y)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        self.idata = self.sample_model(**sampler_config)

        X_df = pd.DataFrame(X, columns=X.columns)
        combined_data = pd.concat([X_df, y], axis=1)
        assert all(combined_data.columns), (
            "All columns must have non-empty names"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in InferenceData",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())

        return self.idata
