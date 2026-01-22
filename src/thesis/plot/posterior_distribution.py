"""Functions to plot posterior-distribution-related figures."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from thesis.plot.config import (
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MEDIA_FEATURE_SUBPLOT_TITLE_PREFIXES,
    POSTERIOR_DISTRIBUTION_PLOT_KWARGS,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
)


def plot_posterior_vs_ground_truth(
    idata: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (20, 8),
    subplot_adjust_kwargs: dict[str, float] = {"top": 0.90, "hspace": 0.3},
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
    figsize : tuple[int, int], default (20, 8)
        The size of the Matplotlib Figure.
    subplot_adjust_kwargs : dict[str, float]
        - Default {"top": 0.80, "hspace": 0.3}

    """
    plt.rcParams.update(
        {
            # "font.size": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(
        nrows=len(media_feature_names),
        ncols=3,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(**subplot_adjust_kwargs)

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
            posterior_samples = (
                idata.posterior[f"{param}s"]
                .values[:, :, media_feature_id]
                .flatten()
            )
            kde = gaussian_kde(posterior_samples)
            x_space = np.linspace(
                posterior_samples.min() - 0.05,
                posterior_samples.max() + 0.05,
                1000,
            )
            y_prob_density = kde(x_space)

            h1 = ax[row, col].fill_between(
                x_space,
                y_prob_density,
                color="C0",
                label="posterior distributions",
                **POSTERIOR_DISTRIBUTION_PLOT_KWARGS,
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
            ax[row, col].set_xlabel(
                f"{param.capitalize().replace('_', ' ')}", size=LABEL_FONT_SIZE
            )
            ax[row, col].set_ylabel("Density", size=LABEL_FONT_SIZE)
            ax[row, col].tick_params(
                axis="both", which="major", labelsize=TICK_FONT_SIZE
            )

            if len(legend_handles) == 0:
                legend_handles = [h1, h2]
                legend_labels = ["Posterior distribution", "Ground truth"]

        if len(media_feature_names) > 1:
            # Add the title of each row
            row_axes = ax[row, :]
            y_top = max(a.get_position().y1 for a in row_axes)
            x_left = min(a.get_position().x0 for a in row_axes)
            fig.text(
                x=x_left,
                y=y_top + 0.05 / len(media_feature_names),
                s=(
                    f"{MEDIA_FEATURE_SUBPLOT_TITLE_PREFIXES[media_feature_name]}"
                    f" Channel {media_feature_id + 1} parameters"
                ),
                fontsize=TITLE_FONT_SIZE,
            )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
        fontsize=LEGEND_FONT_SIZE,
    )
    plt.show()


def plot_experiment_posterior_vs_ground_truth(
    idata_without_experiment: az.InferenceData,
    idata_with_experiment: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (20, 8),
    subplot_adjust_kwargs: dict[str, float] = {"top": 0.90, "hspace": 0.3},
) -> None:
    """Plot the posterior without and with experiment with ground truth.

    Parameters
    ----------
    idata_without_experiment : az.InferenceData
        The inference data without experiment.
    idata_with_experiment : az.InferenceData
        The inference data with experiment.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_names : list[str]
        The list of media feature names to plot against.
    figsize : tuple[int, int], default (20, 9)
        The size of the Matplotlib's figure

    """
    plt.rcParams.update(
        {
            # "font.size": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    media_feature_ids = [
        int(name.split("_")[1]) for name in media_feature_names
    ]

    fig, ax = plt.subplots(
        nrows=len(media_feature_names),
        ncols=3,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(**subplot_adjust_kwargs)

    if len(media_feature_names) == 1:
        ax = ax[np.newaxis, :]

    legend_handles, legend_labels = ([], [])
    for row, (media_feature_id, media_feature_name) in enumerate(
        zip(media_feature_ids, media_feature_names)
    ):
        for col, param in enumerate(["saturation", "shape", "retention_rate"]):
            without_experiment_samples = (
                idata_without_experiment.posterior[f"{param}s"]
                .values[:, :, media_feature_id]
                .flatten()
            )
            without_experiment_kde = gaussian_kde(without_experiment_samples)
            without_experiment_x_space = np.linspace(
                without_experiment_samples.min(),
                without_experiment_samples.max(),
                1000,
            )
            without_experiment_y_prob_density = without_experiment_kde(
                without_experiment_x_space
            )
            h1 = ax[row, col].fill_between(
                without_experiment_x_space,
                without_experiment_y_prob_density,
                color="C0",
                label="without experiment",
                **POSTERIOR_DISTRIBUTION_PLOT_KWARGS,
            )

            with_experiment_samples = (
                idata_with_experiment.posterior[f"{param}s"]
                .values[:, :, media_feature_id]
                .flatten()
            )
            with_experiment_kde = gaussian_kde(with_experiment_samples)
            with_experiment_x_space = np.linspace(
                with_experiment_samples.min(),
                with_experiment_samples.max(),
                1000,
            )
            with_experiment_y_prob_density = with_experiment_kde(
                with_experiment_x_space
            )
            h2 = ax[row, col].fill_between(
                with_experiment_x_space,
                with_experiment_y_prob_density,
                color="C1",
                label="with experiment",
                **POSTERIOR_DISTRIBUTION_PLOT_KWARGS,
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
            ax[row, col].set_xlabel(
                param.capitalize().replace("_", " "), size=LABEL_FONT_SIZE
            )
            ax[row, col].set_ylabel("Density", size=LABEL_FONT_SIZE)
            ax[row, col].tick_params(
                axis="both", which="major", labelsize=TICK_FONT_SIZE
            )

            if len(legend_handles) == 0:
                legend_handles = [h1, h2, h3]
                legend_labels = [
                    "without experiment",
                    "with experiment",
                    "ground truth",
                ]

        if len(media_feature_names) > 1:
            # Add the title of each row
            row_axes = ax[row, :]
            y_top = max(a.get_position().y1 for a in row_axes)
            x_left = min(a.get_position().x0 for a in row_axes)
            fig.text(
                x=x_left,
                y=y_top + 0.05 / len(media_feature_names),
                s=(
                    f"{MEDIA_FEATURE_SUBPLOT_TITLE_PREFIXES[media_feature_name]}"
                    f" Channel {media_feature_id + 1}"
                ),
                fontsize=TITLE_FONT_SIZE,
            )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
        fontsize=LEGEND_FONT_SIZE,
    )
    plt.show()
