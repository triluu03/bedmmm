"""Fit a sample model."""

from bedmmm.data.generator import DataGenerator
from bedmmm.model import MarketingMixModel


def main():
    """Main function."""
    generator = (
        DataGenerator(n_time_periods=52, random_seed=10)
        .generate_baseline(base=10)
        .generate_control(
            control_effect=1,
            noise=1,
        )
        # 1. Low retention rate, high saturation, high shape
        .generate_media(
            retention_rate=0.2,
            saturation=5,
            shape=3,
            # noise=0.05,
        )
        # 2. Low retention rate, low saturation, low shape
        .generate_media(
            retention_rate=0.2,
            saturation=2,
            shape=1,
            # noise=0.05,
        )
        # 3. High retention rate, high saturation, high shape
        .generate_media(
            retention_rate=0.8,
            saturation=6,
            shape=3.5,
            # noise=0.05,
        )
        # 4. High retention rate, low saturation, low shape
        .generate_media(
            retention_rate=0.8,
            saturation=2,
            shape=1,
            # noise=0.05,
        )
        .generate_target(noise=5)
    )

    data = generator.collect()

    model = MarketingMixModel(
        model_config={
            #
            # Media channel 0 priors
            "retention_rate_lb_0": 0.1,
            "retention_rate_ub_0": 0.5,
            "shape_sigma_0": 8,
            "saturation_mu_0": 5,
            "saturation_sigma_0": 1,
            #
            # Media channel 1 priors
            "retention_rate_lb_1": 0.1,
            "retention_rate_ub_1": 0.5,
            "shape_sigma_1": 4,
            "saturation_mu_1": 3,
            "saturation_sigma_1": 1,
            #
            # Media channel 2 priors
            "retention_rate_lb_2": 0.5,
            "retention_rate_ub_2": 0.9,
            "shape_sigma_2": 8,
            "saturation_mu_2": 5,
            "saturation_sigma_2": 1,
            #
            # Media channel 3 priors
            "retention_rate_lb_3": 0.5,
            "retention_rate_ub_3": 0.9,
            "shape_sigma_3": 4,
            "saturation_mu_3": 3,
            "saturation_sigma_3": 1,
            #
            # Control
            "gamma_sigma_0": 1.0,
            #
            # Observation noise
            "obs_sigma": 5,
        }
    )
    idata = model.fit(data.drop(["y"]).to_pandas(), data["y"].to_pandas())


if __name__ == "__main__":
    main()
