"""Module to generate synthetic data."""

from typing import Literal, Self

from numpy.typing import NDArray


class DataGenerator:
    """Synethetic data generator."""

    def __init__(self: Self, n_time_periods: int):
        """Initialization.

        Parameters
        ----------
        n_time_periods : int
            The number of weeks for the data generation.

        """
        self.T = n_time_periods
        self._baseline: NDArray | None = None
        self._control: NDArray | None = None
        self._media: NDArray | None = None
        self._target: NDArray | None = None

    def generate_baseline(self: Self, base: float, noise: float) -> Self:
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
        return self

    def generate_control(self: Self) -> Self:
        """Generate the effects of one control variable."""
        return self

    def generate_media(self: Self) -> Self:
        """Generate the metrics of one media channel."""
        return self

    def generate_target(self: Self, noise_sigma: float) -> Self:
        """Generate the sales target with noise included.

        Parameters
        ----------
        noise_sigma : float
            The observation noise added to the sales target.
            The noise is generated based on the normal distribution
            with mean zero.

        Returns
        -------
        Self
            DataGenerator with target included.

        """
        return self

    def collect(
        self: Self,
    ) -> dict[
        Literal["baseline", "control", "media", "target"], NDArray | None
    ]:
        """Collect the generated data.

        Returns
        -------
        dict[str, NDArray | None]
            The dictionary containing the collected data.
            If None, the data has not been generated.

        """
        return {
            "baseline": self._baseline,
            "control": self._control,
            "media": self._media,
            "target": self._target,
        }


if __name__ == "__main__":
    generator = DataGenerator(104)
    generated_data = (
        generator.generate_baseline(base=10, noise=0.5)
        .generate_control()
        .generate_control()
        .generate_media()
        .collect()
    )
