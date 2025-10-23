"""Marketing Mix Modeling (MMM) in PyMC."""

import warnings
from typing import Any, override

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pymc.util import RandomState
from pymc_extras.model_builder import ModelBuilder


class MarketingMixModel(ModelBuilder):
    """Marketing Mix Model."""

    _model_type = "MarketingMixModel"
    version = "0.1.0"

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Build the PyMC model for the Marketing Mix Model.

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
        self._generate_and_preprocess_model_data(X, y)

        X_values = self.X  # Features array of shape: (T, (M + C) x G)
        y_values = self.y  # Target array of shape: (T, G)

        with pm.Model(coords=self.model_coords) as self.model:
            x_data = pm.MutableData("x_data", X_values)
            y_data = pm.MutableData("y_data", y_values)

            # Prior parameters

            # Media priors

            # Retention rates
            retention_rates = pm.Beta(
                name="retention_rates",
                alpha=np.ones(self.M),
                beta=np.ones(self.M),
            )

            # Media priors
            media_mus = np.array(
                [
                    self.model_config.get(
                        f"media_mu_{m}" for m in range(1, self.M + 1)
                    )
                ]
            )
            sigma_mus = np.array(
                [
                    self.model_config.get(
                        f"sigma_mu_{m}" for m in range(1, self.M + 1)
                    )
                ]
            )
            beta_mus = pm.Normal(
                name="beta_mus",
                mu=media_mus,
                sigma=sigma_mus,
                shape=media_mus.shape,
            )

            # Prior distributions

            # Likelihood

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
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
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
        y: pd.DataFrame,
    ) -> None:
        """Generate and preprocess the model data.

        Set the following attributes in the class:
            - self.model_coords = None
            - self.X : NDArray, shape (T, (M + C) x G)
                All features including media and control.
            - self.y : NDArray, shape (T, G)
                All the sales target.
            - self.M : int
                The total number of media channels.
            - self.C : int
                The total number of control variables.
            - self.G : int
                The total number of geo-locations.

        Parameters
        ----------
        X : pd.DataFrame
            The features dataframe containing:
                - Media features with column name: `media_feature_{m}_{g}`.
                - Control variables with column name: `control_{c}_{g}`.
            where:
                - `m`: is the m-th media channel, starts with 1.
                - `c`: is the c-th media channel, starts with 1.
                - `g`: is the g-th geo-location, starts with 1.
        y : pd.DataFrame
            The target sales dataframe containing:
                - The sales of g-th geo-location with column name: `y_{g}`.

        """
        self.model_coords = None

        self.X = X.values
        self.y = y.values

        self.M = X.columns.str.match(r"^media_feature_\d+_1$").sum().item()
        self.C = X.columns.str.match(r"^control_\d+_1$").sum().item()
        self.G = y.shape[1]

    @override
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
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
                - Control variables with column name: `control_{c}_{g}`.
            where:
                - `m`: is the m-th media channel.
                - `g`: is the g-th geo-location.
        y : pd.DataFrame
            The target sales dataframe containing:
                - The sales of g-th geo-location with column name: `y_{g}`.
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
