"""Functions to plot posterior-distribution-related figures."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_posterior_vs_ground_truth(
    idata: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (18, 5),
) -> None:
    """Plot the posterior distributions along with the ground truth.

    Parameters
    ----------
    idata : az.InferenceData
        The inference data containing posterior distributions.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_names : list[str]
        The list of media feature names to plot against.
    figsize : tuple[int, int], default (18, 5)
        The size of the Matplotlib Figure.

    """
    fig, ax = plt.subplots(
        nrows=len(media_feature_names),
        ncols=3,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)

    if len(media_feature_names) == 1:
        ax = ax[np.newaxis, :]

    media_feature_ids = [
        int(name.split("_")[1]) for name in media_feature_names
    ]

    legend_handles, legend_labels = ([], [])
    for row, (media_feature_id, media_feature_name) in enumerate(
        zip(media_feature_ids, media_feature_names)
    ):
        for col, param in enumerate(["saturation", "shape", "retention_rate"]):
            h1 = ax[row, col].hist(
                idata.posterior[f"{param}s"]
                .values[:, :, media_feature_id]
                .flatten(),
                bins=100,
                density=True,
                histtype="step",
                alpha=0.7,
                label="posterior distributions",
            )
            h2 = ax[row, col].vlines(
                media_param_df.filter(
                    pl.col("media_feature") == media_feature_name
                )[param].item(),
                ymin=0,
                ymax=ax[row, col].get_ylim()[1],
                colors="C2",
                label="ground truth",
            )
            ax[row, col].set_title(f"{param.capitalize().replace('_', ' ')}")

            if len(legend_handles) == 0:
                legend_handles = [h1[2][0], h2]
                legend_labels = ["Posterior distribution", "Ground truth"]

        # Add the title of each row
        row_axes = ax[row, :]
        y_top = max(a.get_position().y1 for a in row_axes)
        fig.text(
            x=0.5,
            y=y_top + 0.1 / len(media_feature_names),
            s=(
                f"Posterior distributions vs Ground Truth of "
                f"Media Channel {media_feature_id + 1}"
            ),
            ha="center",
            va="center",
            fontsize=15,
        )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
    )
    plt.show()


def plot_experiment_posterior_vs_ground_truth(
    idata_before_experiment: az.InferenceData,
    idata_after_experiment: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (18, 5),
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
    media_feature_names : list[str]
        The list of media feature names to plot against.
    figsize : tuple[int, int], default (18, 5)
        The size of the Matplotlib's figure

    """
    media_feature_ids = [
        int(name.split("_")[1]) for name in media_feature_names
    ]

    fig, ax = plt.subplots(
        nrows=len(media_feature_names),
        ncols=3,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)

    legend_handles, legend_labels = ([], [])
    for row, (media_feature_id, media_feature_name) in enumerate(
        zip(media_feature_ids, media_feature_names)
    ):
        for col, param in enumerate(["saturation", "shape", "retention_rate"]):
            h1 = ax[row, col].hist(
                idata_before_experiment.posterior[f"{param}s"]
                .values[:, :, media_feature_id]
                .flatten(),
                bins=100,
                density=True,
                histtype="step",
                alpha=0.7,
                label="before experiment",
            )
            h2 = ax[row, col].hist(
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
            h3 = ax[row, col].vlines(
                media_param_df.filter(
                    pl.col("media_feature") == media_feature_name
                )[param].item(),
                ymin=0,
                ymax=ax[row, col].get_ylim()[1],
                color="C2",
                label="ground truth",
            )
            ax[row, col].set_title(f"{param.capitalize().replace('_', ' ')}")

            if len(legend_handles) == 0:
                legend_handles = [h1[2][0], h2[2][0], h3]
                legend_labels = [
                    "before experiment",
                    "after experiment",
                    "ground truth",
                ]

        # Add the title of each row
        row_axes = ax[row, :]
        y_top = max(a.get_position().y1 for a in row_axes)
        fig.text(
            x=0.5,
            y=y_top + 0.1 / len(media_feature_names),
            s=(
                f"Posterior distributions vs Ground Truth of "
                f"Media Channel {media_feature_id + 1}"
            ),
            ha="center",
            va="center",
            fontsize=15,
        )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
    )
    plt.show()
