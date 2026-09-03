import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from station_discrete_event_memory import _resample_patch, build_discrete_event_arrays
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from tests.test_station24_pipeline import synthetic_static


class DiscreteEventMemoryTests(unittest.TestCase):
    def test_short_event_resampling_is_finite_and_preserves_endpoints(self):
        one_hour = np.arange(4, dtype=np.float64)[None, :]
        one_resampled = _resample_patch(one_hour, bins=6)
        self.assertEqual(one_resampled.shape, (6, 4))
        self.assertTrue(np.all(np.isfinite(one_resampled)))
        self.assertTrue(np.allclose(one_resampled, np.repeat(one_hour, 6, axis=0)))

        three_hour = np.stack(
            [np.zeros(4), np.ones(4), np.full(4, 2.0)], axis=0
        )
        three_resampled = _resample_patch(three_hour, bins=6)
        self.assertEqual(three_resampled.shape, (6, 4))
        self.assertTrue(np.all(np.isfinite(three_resampled)))
        self.assertTrue(np.allclose(three_resampled[0], three_hour[0]))
        self.assertTrue(np.allclose(three_resampled[-1], three_hour[-1]))

    def test_train_only_local_memory_keeps_separate_sparse_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(7)
            train_count, val_count = 70, 3
            train_forecast = rng.uniform(0.1, 0.9, (train_count, 168, 24)).astype(
                np.float32
            )
            train_residual = rng.normal(0, 0.05, train_forecast.shape).astype(np.float32)
            for issue in range(train_count):
                start = (issue * 7) % 140
                train_residual[issue, start : start + 12, :13] -= 0.25
            np.save(root / "train_forecast.npy", train_forecast)
            np.save(root / "train_residual.npy", train_residual)
            np.save(root / "train_fill_mask.npy", np.zeros_like(train_forecast, dtype=np.uint8))
            val_forecast = rng.uniform(0.1, 0.9, (val_count, 168, 24)).astype(np.float32)
            np.save(root / "val_forecast.npy", val_forecast)
            dates = pd.date_range("2025-01-01", periods=train_count, freq="D")
            pd.DataFrame({"issue_date": dates, "target_start": dates}).to_csv(
                root / "train_issue_dates.csv", index=False
            )
            val_dates = pd.date_range("2026-01-01", periods=val_count, freq="D")
            pd.DataFrame({"issue_date": val_dates, "target_start": val_dates}).to_csv(
                root / "val_issue_dates.csv", index=False
            )
            pd.DataFrame(
                {
                    "channel_index": np.arange(24),
                    "data_type": ["wind"] * 13 + ["solar"] * 11,
                    "capacity_mw": np.linspace(50, 150, 24),
                }
            ).to_csv(root / "station_order.csv", index=False)
            arrays = build_discrete_event_arrays(
                root, "val", top_k=12, event_quantile=0.70, target_stride_hours=6
            )
            self.assertEqual(arrays.residual.shape, (val_count, 12, 24, 168))
            self.assertEqual(arrays.time_mask.shape, (val_count, 12, 168))
            self.assertFalse(arrays.audit["topk_averaging"])
            self.assertFalse(arrays.audit["future_query_actual_used"])
            self.assertTrue(np.allclose(arrays.prior_weight.sum(axis=1), 1.0))
            self.assertTrue(np.all(np.isin(arrays.duration, [6, 12, 24])))
            support = arrays.time_mask.sum(axis=-1)
            self.assertTrue(np.array_equal(support, arrays.duration.astype(np.float32)))
            # Solar candidates remain exactly zero; spatially joint wind patches survive.
            self.assertEqual(float(np.abs(arrays.residual[:, :, 13:]).max()), 0.0)
            self.assertGreater(float(np.abs(arrays.residual[:, :, :13]).max()), 0.0)

    def test_one_selector_and_one_event_expert_forward_backward_generate(self):
        features, adjacency = synthetic_static()
        config = {
            "architecture": "station24_resunet",
            "spatial_mode": "fixed_graph",
            "station_count": 24,
            "sequence_length": 16,
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
            "use_body_tail_experts": True,
            "tail_expert_channels": 4,
            "tail_gate_channels": 4,
            "tail_gate_prior_probability": 0.2,
            "tail_gate_loss_weight": 0.1,
            "tail_epsilon_context_hours": 2,
            "use_discrete_event_memory": True,
            "train_discrete_event_memory_only": True,
            "use_event_transport_transformer": True,
            "event_transformer_d_model": 16,
            "event_transformer_heads": 4,
            "event_transformer_layers": 1,
            "event_transformer_feedforward": 32,
            "event_transformer_dropout": 0.0,
            "event_selector_channels": 8,
            "event_selector_loss_weight": 0.1,
            "event_selector_temperature": 0.75,
            "use_retrieval_mismatch_expert": False,
        }
        source_config = copy.deepcopy(config)
        source_config["use_discrete_event_memory"] = False
        source_config["use_event_transport_transformer"] = False
        source_config["train_discrete_event_memory_only"] = False
        source_config["event_selector_loss_weight"] = 0.0
        source = Station24DiffusionModel(source_config, features, adjacency)
        model = Station24DiffusionModel(config, features, adjacency)
        incompatible = model.load_state_dict(source.state_dict(), strict=False)
        self.assertEqual(
            set(incompatible.missing_keys), set(model.discrete_event_new_state_dict_keys)
        )
        self.assertFalse(incompatible.unexpected_keys)
        trainable = model.configure_discrete_event_training()
        self.assertTrue(any("event_memory_selector" in name for name in trainable))
        self.assertTrue(any("event_prototype_adapter" in name for name in trainable))
        batch_size, candidates, length = 2, 8, 16
        residual = torch.randn(batch_size, 24, length)
        candidate_residual = torch.zeros(batch_size, candidates, 24, length)
        candidate_mask = torch.zeros(batch_size, candidates, length)
        starts = torch.arange(candidates) % 8
        for candidate in range(candidates):
            start = int(starts[candidate])
            candidate_residual[:, candidate, :13, start : start + 6] = torch.randn(
                batch_size, 13, 6
            )
            candidate_mask[:, candidate, start : start + 6] = 1.0
        batch = {
            "residual_target": residual,
            "residual": residual.clone(),
            "residual_scale": torch.ones_like(residual),
            "forecast": torch.rand(batch_size, 24, length),
            "calendar": torch.randn(batch_size, 8, length),
            "lead": torch.rand(batch_size, 2, length),
            "valid_mask": torch.ones_like(residual),
            "forecast_ramps": torch.randn(batch_size, 24, 3, length),
            "forecast_revision": torch.randn(batch_size, 24, length),
            "revision_mask": torch.ones(batch_size, 24, length),
            "recent_error": torch.randn(batch_size, 24, 24),
            "recent_error_mask": torch.ones(batch_size, 24, 1),
            "node_state": torch.zeros(batch_size, 24, 4, length),
            "loss_weight": torch.ones_like(residual),
            "event_time_weight": torch.ones(batch_size, length),
            "event_active": torch.tensor([1.0, 0.0]),
            "event_replay_weight": torch.tensor([2.0, 1.0]),
            "event_start": torch.tensor([2, 0]),
            "event_window_mask": torch.tensor(
                [[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0] * 16],
                dtype=torch.float32,
            ),
            "event_sync_station_weight": torch.zeros(batch_size, 24),
            "retrieval_residual": candidate_residual,
            "retrieval_time_mask": candidate_mask,
            "retrieval_distance": torch.rand(batch_size, candidates),
            "retrieval_prior_weight": torch.full(
                (batch_size, candidates), 1.0 / candidates
            ),
            "retrieval_train_index": torch.arange(candidates).repeat(batch_size, 1),
            "retrieval_event_type": torch.arange(candidates).repeat(batch_size, 1) % 4,
            "retrieval_duration": torch.full((batch_size, candidates), 6),
            "retrieval_target_start": starts.repeat(batch_size, 1),
            "retrieval_source_start": starts.repeat(batch_size, 1),
        }
        model.train()
        logits = model.event_memory_logits(batch)
        self.assertEqual(tuple(logits.shape), (batch_size, candidates))
        self.assertGreater(float(logits.std().detach()), 0.0)
        loss = model(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.eval()
        samples, audit = model.generate(batch, n_samples=3, return_expert_audit=True)
        self.assertEqual(samples.shape, (batch_size, 3, 24, length))
        self.assertEqual(audit["event_memory_index"].shape, (batch_size, 3))
        self.assertTrue(torch.all(audit["event_memory_index"] >= 0))
        self.assertTrue(torch.all(audit["mismatch_route"] == 0))

    def test_multiduration_transport_bank_includes_short_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(13)
            train_count, val_count = 70, 2
            shape = (train_count, 168, 24)
            forecast = rng.uniform(0.1, 0.9, shape).astype(np.float32)
            residual = rng.normal(0, 0.04, shape).astype(np.float32)
            for issue in range(train_count):
                start = (issue * 11) % 150
                duration = (1, 3, 6, 12)[issue % 4]
                residual[issue, start : start + duration, :13] -= 0.30
            np.save(root / "train_forecast.npy", forecast)
            np.save(root / "train_residual.npy", residual)
            np.save(root / "train_fill_mask.npy", np.zeros(shape, dtype=np.uint8))
            np.save(
                root / "val_forecast.npy",
                rng.uniform(0.1, 0.9, (val_count, 168, 24)).astype(np.float32),
            )
            dates = pd.date_range("2025-01-01", periods=train_count, freq="D")
            pd.DataFrame({"issue_date": dates, "target_start": dates}).to_csv(
                root / "train_issue_dates.csv", index=False
            )
            val_dates = pd.date_range("2026-01-01", periods=val_count, freq="D")
            pd.DataFrame({"issue_date": val_dates, "target_start": val_dates}).to_csv(
                root / "val_issue_dates.csv", index=False
            )
            pd.DataFrame(
                {
                    "channel_index": np.arange(24),
                    "data_type": ["wind"] * 13 + ["solar"] * 11,
                    "capacity_mw": np.linspace(50, 150, 24),
                }
            ).to_csv(root / "station_order.csv", index=False)
            arrays = build_discrete_event_arrays(
                root,
                "val",
                top_k=16,
                event_quantile=0.70,
                target_stride_hours=6,
                event_durations=(1, 3, 6, 12),
            )
            self.assertEqual(arrays.audit["durations_hours"], [1, 3, 6, 12])
            self.assertTrue(np.all(np.isin(arrays.duration, [1, 3, 6, 12])))


if __name__ == "__main__":
    unittest.main()
