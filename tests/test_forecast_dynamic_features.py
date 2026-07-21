import unittest

import torch

from diff_models_multivariate import build_forecast_dynamic_features


class ForecastDynamicFeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "enabled": True,
            "names": ["load_ramp_1h", "net_load"],
            "net_load_scale": {"wind_to_load": 0.25, "solar_to_load": 0.5},
            "normalization": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        }

    def test_builds_ramp_and_physical_net_load_proxy(self):
        forecast = torch.zeros(1, 3, 4)
        forecast[:, 0] = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        forecast[:, 1] = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        forecast[:, 2] = torch.tensor([[0.5, 0.6, 0.8, 0.7]])

        features = build_forecast_dynamic_features(forecast, self.config)

        expected_ramp = torch.tensor([[0.0, 0.1, 0.2, -0.1]])
        expected_net = forecast[:, 2] - 0.25 * forecast[:, 0] - 0.5 * forecast[:, 1]
        torch.testing.assert_close(features[:, 0], expected_ramp)
        torch.testing.assert_close(features[:, 1], expected_net)

    def test_applies_train_normalization(self):
        config = dict(self.config)
        config["normalization"] = {"mean": [0.1, 0.5], "std": [0.2, 0.25]}
        forecast = torch.zeros(2, 3, 168)
        forecast[:, 2] = 0.5

        features = build_forecast_dynamic_features(forecast, config)

        self.assertEqual(tuple(features.shape), (2, 2, 168))
        torch.testing.assert_close(features[:, 0], torch.full((2, 168), -0.5))
        torch.testing.assert_close(features[:, 1], torch.zeros(2, 168))

    def test_disabled_features_return_none(self):
        forecast = torch.zeros(1, 3, 168)
        self.assertIsNone(build_forecast_dynamic_features(forecast, {"enabled": False}))


if __name__ == "__main__":
    unittest.main()
