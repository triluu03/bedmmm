"""Functions to plot all response-curves-related figures."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray


def plot_generated_data_on_response_curve(
    data: pl.DataFrame,
    media_param_df: pl.DataFrame,
) -> None:
    """Plot the generated data on its media channel's response curves.

    Parameters
    ----------
    data : pl.DataFrame
        The generated dataframe.
    media_param_df : pl.DataFrame
        The dataframe with media channel's parameters.

    """
    n_media_channel = media_param_df["media_feature"].len()

    fig, ax = plt.subplots(ncols=n_media_channel, figsize=(20, 5))

    for i, feature in enumerate(media_param_df["media_feature"].to_list()):
        saturation = media_param_df.filter(pl.col("media_feature") == feature)[
            "saturation"
        ].item()
        shape = media_param_df.filter(pl.col("media_feature") == feature)[
            "shape"
        ].item()

        x_space = np.linspace(
            0, 8, 100
        )  # TODO: remove the hardcoded upper bound (number 8)

        ax[i].plot(
            x_space,
            saturation * (1 - np.exp(-x_space / shape)),
            color="C1",
        )
        ax[i].scatter(
            data[feature].to_numpy(),
            saturation * (1 - np.exp(-data[feature].to_numpy() / shape)),
            alpha=0.5,
        )
        ax[i].set_title(f"Response curve for {feature}")
        ax[i].set_xlabel("Investments")
        ax[i].set_ylabel("Diminishing Returns")

    plt.plot()


def plot_posterior_response_curve(
    idata: az.InferenceData,
    data: pl.DataFrame,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
):
    """Plot the posterior response curves.

    Parameters
    ----------
    idata : az.InferenceData
        The inference data with posterior distributions.
    data : pl.DataFrame
        The generated data.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    media_feature_id = int(media_feature_name.split("_")[1])

    # True response curve parameters
    true_saturation = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["saturation"].item()
    true_shape = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["shape"].item()

    # Posterior samples of the response curves
    saturation_samples = (
        idata.posterior["saturations"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    shape_samples = (
        idata.posterior["shapes"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )

    # Calculate the lower and upper 94% HDI
    x_space_col = np.linspace(0, 10, 1000)[np.newaxis, :]
    y_samples = saturation_samples * (1 - np.exp(-x_space_col / shape_samples))
    y_samples = np.apply_along_axis(
        lambda samples: az.hdi(samples, hdi_prob=0.94),
        axis=0,
        arr=y_samples,
    )
    y_lower_hdi = y_samples[0, :]
    y_upper_hdi = y_samples[1, :]

    # Generate the plot
    x_space = np.linspace(0, 10, 1000)
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        x=x_space,
        y1=y_lower_hdi,
        y2=y_upper_hdi,
        color="C0",
        alpha=0.2,
        label="94% HDI",
    )
    plt.plot(
        x_space,
        true_saturation * (1 - np.exp(-x_space / true_shape)),
        label="Ground truth",
        color="C1",
    )
    plt.scatter(
        data["media_0"].to_numpy(),
        true_saturation
        * (1 - np.exp(-data["media_0"].to_numpy() / true_shape)),
        label="Observed data points",
        edgecolors="black",
        s=25,
        linewidths=0.3,
        color="C3",
    )

    plt.xlabel("Media investments")
    plt.ylabel("Diminishing Returns")
    plt.legend()
    plt.show()


def plot_experiment_posterior_response_curve(
    idata_before_experiment: az.InferenceData,
    idata_after_experiment: az.InferenceData,
    data: pl.DataFrame,
    experiment_data: NDArray,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
) -> None:
    """Plot the posterior response curves before and after experiment.

    Parameters
    ----------
    idata_before_experiment : az.InferenceData
        The inference data before experiment.
    idata_after_experiment : az.InferenceData
        The inference data after experiment.
    data : pl.DataFrame
        The generated data.
    experiment_data : NDArray
        The experiment data point.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    media_feature_id = int(media_feature_name.split("_")[1])

    x_space = np.linspace(0, 50, 1000)

    # Before experiment
    before_saturation_samples = (
        idata_before_experiment.posterior["saturations"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    before_shape_samples = (
        idata_before_experiment.posterior["shapes"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    before_y_samples = before_saturation_samples * (
        1 - np.exp(-x_space / before_shape_samples)
    )
    before_y_samples = np.apply_along_axis(
        lambda samples: az.hdi(samples, hdi_prob=0.94),
        axis=0,
        arr=before_y_samples,
    )
    before_y_lower_hdi = before_y_samples[0, :]
    before_y_upper_hdi = before_y_samples[1, :]

    # After experiments
    after_saturation_samples = (
        idata_after_experiment.posterior["saturations"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    after_shape_samples = (
        idata_after_experiment.posterior["shapes"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    after_y_samples = after_saturation_samples * (
        1 - np.exp(-x_space / after_shape_samples)
    )
    after_y_samples = np.apply_along_axis(
        lambda samples: az.hdi(samples, hdi_prob=0.94),
        axis=0,
        arr=after_y_samples,
    )
    after_y_lower_hdi = after_y_samples[0, :]
    after_y_upper_hdi = after_y_samples[1, :]

    # Ground truth
    true_saturation = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["saturation"].item()
    true_shape = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["shape"].item()

    # Final plots
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        x=x_space,
        y1=before_y_lower_hdi,
        y2=before_y_upper_hdi,
        color="C0",
        alpha=0.2,
        label="94% HDI before experiment",
    )
    plt.fill_between(
        x=x_space,
        y1=after_y_lower_hdi,
        y2=after_y_upper_hdi,
        color="C2",
        alpha=0.2,
        label="94% HDI after experiment",
    )
    plt.plot(
        x_space,
        true_saturation * (1 - np.exp(-x_space / true_shape)),
        label="ground truth",
        color="C1",
    )

    plt.scatter(
        data[media_feature_name].to_numpy(),
        true_saturation
        * (1 - np.exp(-data[media_feature_name].to_numpy() / true_shape)),
        label="Observed data points",
        color="C3",
        edgecolors="black",
        s=25,
        linewidths=0.3,
    )
    plt.scatter(
        experiment_data[media_feature_id],
        true_saturation
        * (1 - np.exp(-experiment_data[media_feature_id] / true_shape)),
        label="Experiment data point",
        color="C4",
        edgecolors="black",
        s=50,
        linewidths=0.3,
        marker="D",
    )

    plt.xlabel("Media Spend")
    plt.ylabel("Diminishing Returns Response")
    plt.legend()
    plt.show()
