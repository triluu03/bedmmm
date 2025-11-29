"""Functions to plot posterior-distribution-related figures."""

import arviz as az
import matplotlib.pyplot as plt
import polars as pl


def plot_posterior_vs_ground_truth(
    idata: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
) -> None:
    """Plot the posterior distributions along with the ground truth.

    Parameters
    ----------
    idata : az.InferenceData
        The inference data containing posterior distributions.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    _, ax = plt.subplots(1, 3, figsize=(18, 5))

    media_feature_id = int(media_feature_name.split("_")[1])

    for i, param in enumerate(["saturation", "shape", "retention_rate"]):
        ax[i].hist(
            idata.posterior[f"{param}s"]
            .values[:, :, media_feature_id]
            .flatten(),
            bins=100,
            density=True,
            histtype="step",
            alpha=0.7,
            label="posterior distributions",
        )
        ax[i].vlines(
            media_param_df.filter(
                pl.col("media_feature") == media_feature_name
            )[param].item(),
            ymin=0,
            ymax=ax[i].get_ylim()[1],
            colors="C2",
            label="ground truth",
        )
        ax[i].set_title(f"{param.capitalize().replace('_', ' ')}")

    plt.legend()
    plt.suptitle(
        f"Posterior distributions vs Ground Truth for "
        f"{media_feature_name.capitalize().replace('_', ' ')}"
    )
    plt.show()


def plot_experiment_posterior_vs_ground_truth(
    idata_before_experiment: az.InferenceData,
    idata_after_experiment: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
) -> None:
    """Plot the posterior before and after experiment with ground truth.

    Parameters
    ----------
    idata_before_experiment : az.InferenceData
        The inference data before experiment.
    idata_after_experiment : az.InferenceData
        The inference data after experiment.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    media_feature_id = int(media_feature_name.split("_")[1])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for i, param in enumerate(["saturation", "shape", "retention_rate"]):
        ax[i].hist(
            idata_before_experiment.posterior[f"{param}s"]
            .values[:, :, media_feature_id]
            .flatten(),
            bins=100,
            density=True,
            histtype="step",
            alpha=0.7,
            label="before experiment",
        )
        ax[i].hist(
            idata_after_experiment.posterior[f"{param}s"]
            .values[:, :, media_feature_id]
            .flatten(),
            bins=100,
            density=True,
            histtype="step",
            alpha=0.7,
            color="C1",
            label="after experiment",
        )
        ax[i].vlines(
            media_param_df.filter(
                pl.col("media_feature") == media_feature_name
            )[param].item(),
            ymin=0,
            ymax=ax[i].get_ylim()[1],
            color="C2",
            label="ground truth",
        )
        ax[i].set_title(f"{param.capitalize().replace('_', ' ')}")

    plt.legend()
    plt.suptitle(
        f"Posterior distributions vs Ground truth for "
        f"{media_feature_name.capitalize().replace('_', ' ')}",
    )
    plt.show()
