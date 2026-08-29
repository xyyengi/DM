import unittest

import torch

from src.models.station_conditioned_diffusion import (
    Station24DiffusionModel,
    StationForecastTrustHead,
)


class ForecastTrustHeadTest(unittest.TestCase):
    def test_initial_gate_matches_explicit_lead_day_prior(self) -> None:
        initial = [0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]
        head = StationForecastTrustHead(
            {
                "station_count": 24,
                "recent_error_hours": 24,
                "forecast_trust_channels": 16,
                "forecast_trust_initial_history_fraction": initial,
            },
            torch.zeros(24, 5),
            groups=8,
        )
        batch = 2
        forecast = torch.rand(batch, 24, 168)
        history = torch.rand(batch, 24, 168)
        center, fraction = head(
            forecast,
            history,
            torch.rand_like(forecast),
            torch.zeros(batch, 8, 168),
            torch.zeros(batch, 2, 168),
            torch.zeros_like(forecast),
            torch.zeros_like(forecast),
            torch.zeros(batch, 24, 24),
            torch.zeros(batch, 24, 1),
        )
        expected = torch.tensor(initial).repeat_interleave(24)
        self.assertTrue(
            torch.allclose(fraction, expected[None, None].expand_as(fraction), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(center, forecast + fraction * (history - forecast))
        )

    def test_gate_is_station_and_hour_resolved_after_learning(self) -> None:
        head = StationForecastTrustHead(
            {
                "station_count": 24,
                "recent_error_hours": 24,
                "forecast_trust_channels": 16,
            },
            torch.randn(24, 5),
            groups=8,
        )
        with torch.no_grad():
            head.output.weight.fill_(0.02)
        forecast = torch.rand(1, 24, 168)
        history = torch.rand(1, 24, 168)
        _, fraction = head(
            forecast,
            history,
            torch.rand_like(forecast),
            torch.zeros(1, 8, 168),
            torch.zeros(1, 2, 168),
            torch.zeros_like(forecast),
            torch.ones_like(forecast),
            torch.rand(1, 24, 24),
            torch.ones(1, 24, 1),
        )
        self.assertGreater(float(fraction.std()), 0.0)
        self.assertTrue(bool(torch.all((fraction > 0) & (fraction < 1))))

    def test_wrapper_diffuses_residual_around_dynamic_center(self) -> None:
        features = torch.zeros(24, 5)
        features[:13, 0] = 1.0
        features[13:, 1] = 1.0
        adjacency = torch.eye(24)
        config = {
            "architecture": "station24_resunet",
            "spatial_mode": "none",
            "station_count": 24,
            "sequence_length": 168,
            "base_channels": 4,
            "num_layers": 3,
            "channel_multipliers": [1, 2, 4],
            "group_norm_groups": 4,
            "dropout": 0.0,
            "timestep_embedding_dim": 8,
            "num_steps": 2,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "use_recent_error": True,
            "recent_error_hours": 24,
            "use_forecast_trust_center": True,
            "forecast_trust_channels": 8,
            "forecast_trust_center_loss_weight": 0.2,
            "forecast_trust_oracle_loss_weight": 0.1,
            "forecast_trust_smoothness_weight": 0.01,
        }
        model = Station24DiffusionModel(config, features, adjacency)
        forecast = torch.rand(1, 24, 168)
        history = torch.rand(1, 24, 168)
        actual = torch.rand(1, 24, 168)
        batch = {
            "forecast": forecast,
            "actual": actual,
            "residual": actual - forecast,
            "residual_target": actual - forecast,
            "residual_scale": torch.ones_like(forecast),
            "historical_center": history,
            "historical_dispersion": torch.rand_like(forecast),
            "calendar": torch.zeros(1, 8, 168),
            "lead": torch.zeros(1, 2, 168),
            "valid_mask": torch.ones_like(forecast),
            "forecast_revision": torch.zeros_like(forecast),
            "revision_mask": torch.zeros_like(forecast),
            "recent_error": torch.zeros(1, 24, 24),
            "recent_error_mask": torch.zeros(1, 24, 1),
            "node_state": torch.zeros(1, 24, 4, 168),
            "loss_weight": torch.ones_like(forecast),
            "event_time_weight": torch.ones(1, 168),
            "event_active": torch.zeros(1),
            "event_start": torch.zeros(1, dtype=torch.long),
            "event_window_mask": torch.zeros(1, 168),
            "event_sync_station_weight": torch.zeros(1, 24),
        }
        center, fraction = model.predict_forecast_center(batch)
        self.assertEqual(center.shape, forecast.shape)
        self.assertEqual(fraction.shape, forecast.shape)
        loss = model(
            batch,
            timestep=torch.zeros(1, dtype=torch.long),
            noise=torch.zeros_like(forecast),
            include_auxiliary=False,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.forecast_trust_head.output.weight.grad)


if __name__ == "__main__":
    unittest.main()
