"""Functions to plot all response-curves-related figures."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from bedmmm.functions.carryover import adstock_numpy


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
        ax[i].set_title(
            f"Saturation curve of {feature.capitalize().replace('_', ' ')}"
        )
        ax[i].set_xlabel("Investments")
        ax[i].set_ylabel("Diminishing returns")

    plt.plot()


def plot_posterior_response_curve(
    idata: az.InferenceData,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
    figsize: tuple[int, int] = (15, 8),
):
    """Plot the posterior response curves.

    Parameters
    ----------
    idata : az.InferenceData
        The inference data with posterior distributions.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

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
    plt.figure(figsize=figsize)
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

    plt.xlabel("Media investments (x)")
    plt.ylabel("Diminishing returns")
    plt.legend()
    plt.show()


def plot_posterior_response_curve_multiple(
    idata: az.InferenceData,
    data: pl.DataFrame,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (25, 16),
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
    media_feature_names : list[str]
        The media feature name to plot against.
    figsize : tuple[int, int], default (18, 10)
        The size of the Matplotlib's figure

    """
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "legend.fontsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    assert len(media_feature_names) == 4, (
        "This method only supports 4 media channels"
    )

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, hspace=0.25)

    media_feature_ids = [
        int(name.split("_")[1]) for name in media_feature_names
    ]

    legend_handles, legend_labels = ([], [])
    for media_feature_id, media_feature_name, subplot_title_prefix in zip(
        media_feature_ids, media_feature_names, ["(a)", "(b)", "(c)", "(d)"]
    ):
        row = media_feature_id // 2
        col = media_feature_id % 2

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
        y_samples = saturation_samples * (
            1 - np.exp(-x_space_col / shape_samples)
        )
        y_samples = np.apply_along_axis(
            lambda samples: az.hdi(samples, hdi_prob=0.94),
            axis=0,
            arr=y_samples,
        )
        y_lower_hdi = y_samples[0, :]
        y_upper_hdi = y_samples[1, :]

        # Generate the plot
        x_space = np.linspace(0, 8, 1000)
        h1 = ax[row, col].fill_between(
            x=x_space,
            y1=y_lower_hdi,
            y2=y_upper_hdi,
            color="C0",
            alpha=0.2,
            label="94% HDI",
        )
        h2 = ax[row, col].plot(
            x_space,
            true_saturation * (1 - np.exp(-x_space / true_shape)),
            label="Ground truth",
            color="C1",
        )
        h3 = ax[row, col].scatter(
            data["media_0"].to_numpy(),
            true_saturation
            * (1 - np.exp(-data["media_0"].to_numpy() / true_shape)),
            label="Observed data points",
            edgecolors="black",
            s=25,
            linewidths=0.3,
            color="C3",
        )

        if len(legend_handles) == 0:
            legend_handles.extend([h1, h2[0], h3])
            legend_labels.extend(
                ["94% HDI", "Ground truth", "Observed data points"]
            )

        ax[row, col].set_xlabel("Media investments (x)")
        ax[row, col].set_ylabel("Diminishing Returns")
        ax[row, col].set_title(
            f"{subplot_title_prefix} Posterior saturation curve and Ground truth of "
            f"Channel {media_feature_id + 1}"
        )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
    )
    plt.show()


def plot_experiment_posterior_response_curve(
    idata_without_experiment: az.InferenceData,
    idata_with_experiment: az.InferenceData,
    data: pl.DataFrame,
    experiment_data: np.typing.NDArray,
    media_param_df: pl.DataFrame,
    media_feature_name: str,
    figsize: tuple[int, int] = (15, 8),
) -> None:
    """Plot the posterior response curves without and with experiment.

    Parameters
    ----------
    idata_without_experiment : az.InferenceData
        The inference data without experiment.
    idata_with_experiment : az.InferenceData
        The inference data with experiment.
    data : pl.DataFrame
        The generated data.
    experiment_data : NDArray
        The experiment data point.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_name : str
        The media feature name to plot against.

    """
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    media_feature_id = int(media_feature_name.split("_")[1])

    x_space = np.linspace(
        start=0,
        stop=max(
            data[media_feature_name].max(),
            experiment_data[media_feature_id],
        )
        * 1.5,
        num=1000,
    )

    # without experiment
    without_saturation_samples = (
        idata_without_experiment.posterior["saturations"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    without_shape_samples = (
        idata_without_experiment.posterior["shapes"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    without_y_samples = without_saturation_samples * (
        1 - np.exp(-x_space / without_shape_samples)
    )
    without_y_samples = np.apply_along_axis(
        lambda samples: az.hdi(samples, hdi_prob=0.94),
        axis=0,
        arr=without_y_samples,
    )
    without_y_lower_hdi = without_y_samples[0, :]
    without_y_upper_hdi = without_y_samples[1, :]

    # with experiments
    with_saturation_samples = (
        idata_with_experiment.posterior["saturations"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    with_shape_samples = (
        idata_with_experiment.posterior["shapes"]
        .values[:, :, media_feature_id]
        .flatten()[:, np.newaxis]
    )
    with_y_samples = with_saturation_samples * (
        1 - np.exp(-x_space / with_shape_samples)
    )
    with_y_samples = np.apply_along_axis(
        lambda samples: az.hdi(samples, hdi_prob=0.94),
        axis=0,
        arr=with_y_samples,
    )
    with_y_lower_hdi = with_y_samples[0, :]
    with_y_upper_hdi = with_y_samples[1, :]

    # Ground truth
    true_saturation = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["saturation"].item()
    true_shape = media_param_df.filter(
        pl.col("media_feature") == media_feature_name
    )["shape"].item()

    # Final plots
    plt.figure(figsize=figsize)
    plt.fill_between(
        x=x_space,
        y1=without_y_lower_hdi,
        y2=without_y_upper_hdi,
        color="C0",
        alpha=0.2,
        label="94% HDI without experiment",
    )
    plt.fill_between(
        x=x_space,
        y1=with_y_lower_hdi,
        y2=with_y_upper_hdi,
        color="C2",
        alpha=0.2,
        label="94% HDI with experiment",
    )
    plt.plot(
        x_space,
        true_saturation * (1 - np.exp(-x_space / true_shape)),
        label="ground truth",
        color="C1",
    )

    plt.scatter(
        experiment_data[media_feature_id],
        true_saturation
        * (
            1
            - np.exp(
                -adstock_numpy(
                    np.concatenate(
                        [
                            data[media_feature_name].to_numpy()[-11:],
                            [experiment_data[media_feature_id]],
                        ]
                    ),
                    retention_rate=media_param_df.filter(
                        pl.col("media_feature") == media_feature_name
                    )
                    .select(pl.col("retention_rate"))
                    .item(),
                )[-1]
                / true_shape
            )
        ),
        label="Experiment data point",
        color="C4",
        edgecolors="black",
        s=50,
        linewidths=0.3,
    )

    plt.xlabel("Media investments (x)")
    plt.ylabel("Diminishing returns")
    plt.legend()
    plt.show()


def plot_experiment_posterior_response_curve_multiple(
    idata_without_experiment: az.InferenceData,
    idata_with_experiment: az.InferenceData,
    data: pl.DataFrame,
    experiment_data,
    media_param_df: pl.DataFrame,
    media_feature_names: list[str],
    figsize: tuple[int, int] = (25, 16),
) -> None:
    """Plot the posterior response curves without and with experiment.

    Parameters
    ----------
    idata_without_experiment : az.InferenceData
        The inference data without experiment.
    idata_with_experiment : az.InferenceData
        The inference data with experiment.
    data : pl.DataFrame
        The generated data.
    experiment_data : NDArray
        The experiment data point.
    media_param_df : pl.DataFrame
        The media parameters dataframe.
    media_feature_names : list[str]
        The media feature name to plot against.
    figsize : tuple[int, int], default (18, 10)
        The size of the Matplotlib's figure

    """
    assert len(media_feature_names) == 4, (
        "This method only supports 4 media channels"
    )

    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "legend.fontsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.925, hspace=0.25)

    media_feature_ids = [
        int(name.split("_")[1]) for name in media_feature_names
    ]

    legend_handles, legend_labels = ([], [])
    for media_feature_id, media_feature_name, subplot_title_prefix in zip(
        media_feature_ids, media_feature_names, ["(a)", "(b)", "(c)", "(d)"]
    ):
        row = media_feature_id // 2
        col = media_feature_id % 2

        x_space = np.linspace(
            start=0,
            stop=max(
                data[media_feature_name].max(),
                experiment_data[media_feature_id],
            )
            * 1.5,
            num=1000,
        )

        # without experiment
        without_saturation_samples = (
            idata_without_experiment.posterior["saturations"]
            .values[:, :, media_feature_id]
            .flatten()[:, np.newaxis]
        )
        without_shape_samples = (
            idata_without_experiment.posterior["shapes"]
            .values[:, :, media_feature_id]
            .flatten()[:, np.newaxis]
        )
        without_y_samples = without_saturation_samples * (
            1 - np.exp(-x_space / without_shape_samples)
        )
        without_y_samples = np.apply_along_axis(
            lambda samples: az.hdi(samples, hdi_prob=0.94),
            axis=0,
            arr=without_y_samples,
        )
        without_y_lower_hdi = without_y_samples[0, :]
        without_y_upper_hdi = without_y_samples[1, :]

        # with experiments
        with_saturation_samples = (
            idata_with_experiment.posterior["saturations"]
            .values[:, :, media_feature_id]
            .flatten()[:, np.newaxis]
        )
        with_shape_samples = (
            idata_with_experiment.posterior["shapes"]
            .values[:, :, media_feature_id]
            .flatten()[:, np.newaxis]
        )
        with_y_samples = with_saturation_samples * (
            1 - np.exp(-x_space / with_shape_samples)
        )
        with_y_samples = np.apply_along_axis(
            lambda samples: az.hdi(samples, hdi_prob=0.94),
            axis=0,
            arr=with_y_samples,
        )
        with_y_lower_hdi = with_y_samples[0, :]
        with_y_upper_hdi = with_y_samples[1, :]

        # Ground truth
        true_saturation = media_param_df.filter(
            pl.col("media_feature") == media_feature_name
        )["saturation"].item()
        true_shape = media_param_df.filter(
            pl.col("media_feature") == media_feature_name
        )["shape"].item()

        # Final plots
        h1 = ax[row, col].fill_between(
            x=x_space,
            y1=without_y_lower_hdi,
            y2=without_y_upper_hdi,
            color="C0",
            alpha=0.2,
            label="94% HDI without experiment",
        )
        h2 = ax[row, col].fill_between(
            x=x_space,
            y1=with_y_lower_hdi,
            y2=with_y_upper_hdi,
            color="C2",
            alpha=0.2,
            label="94% HDI with experiment",
        )
        h3 = ax[row, col].plot(
            x_space,
            true_saturation * (1 - np.exp(-x_space / true_shape)),
            label="ground truth",
            color="C1",
        )

        h4 = ax[row, col].scatter(
            data[media_feature_name].to_numpy(),
            true_saturation
            * (1 - np.exp(-data[media_feature_name].to_numpy() / true_shape)),
            label="Observed data points",
            color="C3",
            edgecolors="black",
            s=25,
            linewidths=0.3,
        )
        h5 = ax[row, col].scatter(
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

        if len(legend_handles) == 0:
            legend_handles.extend([h1, h2, h3[0], h4, h5])
            legend_labels.extend(
                [
                    "94% HDI without experiment",
                    "94% HDI with experiment",
                    "Ground truth",
                    "Observed data points",
                    "Experiment data point",
                ]
            )

        ax[row, col].set_xlabel("Investments")
        ax[row, col].set_ylabel("Diminishing returns")
        ax[row, col].set_title(
            f"{subplot_title_prefix} Posterior saturation curve and Ground truth of "
            f"Channel {media_feature_id + 1}"
        )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=len(legend_labels),
    )
    plt.show()
