import tempfile
import unittest
import copy
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml


def synthetic_static():
    features = torch.zeros(24, 5)
    features[:13, 0] = 1.0
    features[13:, 1] = 1.0
    features[:, 2:] = torch.linspace(0, 1, 24)[:, None]
    adjacency = torch.eye(24)
    for index in range(23):
        adjacency[index, index + 1] = 0.5
        adjacency[index + 1, index] = 0.5
    return features, adjacency


class Station24ModelTests(unittest.TestCase):
    def config(self, spatial_mode):
        return {
            "architecture": "station24_resunet",
            "spatial_mode": spatial_mode,
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
        }

    def batch(self, batch_size=2):
        residual_target = torch.randn(batch_size, 24, 16)
        return {
            "residual_target": residual_target,
            "residual": residual_target.clone(),
            "residual_scale": torch.ones(batch_size, 24, 16),
            "forecast": torch.rand(batch_size, 24, 16),
            "calendar": torch.randn(batch_size, 8, 16),
            "lead": torch.rand(batch_size, 2, 16),
            "valid_mask": torch.ones(batch_size, 24, 16),
            "forecast_ramps": torch.randn(batch_size, 24, 3, 16),
            "forecast_revision": torch.randn(batch_size, 24, 16),
            "revision_mask": torch.ones(batch_size, 24, 16),
            "recent_error": torch.randn(batch_size, 24, 24),
            "recent_error_mask": torch.ones(batch_size, 24, 1),
            "node_state": torch.rand(batch_size, 24, 4, 16),
        }

    def test_all_spatial_modes_forward_backward_and_sample(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        counts = {}
        for mode in ["none", "fixed_graph", "type_gated_graph"]:
            model = Station24DiffusionModel(self.config(mode), features, adjacency)
            batch = self.batch()
            loss = model(batch)
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            sample = model.generate(self.batch(batch_size=1), n_samples=2)
            self.assertEqual(tuple(sample.shape), (1, 2, 24, 16))
            counts[mode] = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(counts["fixed_graph"] - counts["none"], 1000)
        self.assertLess(counts["type_gated_graph"] - counts["none"], 1000)

    def test_forecast_condition_dropout_is_parameter_free_and_switchable(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        base_config = self.config("fixed_graph")
        dropout_config = copy.deepcopy(base_config)
        dropout_config.update(
            {
                "forecast_condition_dropout_prob": 0.10,
                "use_state_encoder": True,
                "state_feature_dim": 4,
                "state_channels": [4, 8, 16],
            }
        )
        state_base_config = copy.deepcopy(dropout_config)
        state_base_config["forecast_condition_dropout_prob"] = 0.0
        baseline = Station24DiffusionModel(
            state_base_config, features, adjacency
        )
        candidate = Station24DiffusionModel(
            dropout_config, features, adjacency
        )
        candidate.load_state_dict(baseline.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in candidate.parameters()),
        )

        candidate.train()
        candidate(
            self.batch(),
            timestep=torch.tensor([0, 1]),
            noise=torch.randn(2, 24, 16),
            include_auxiliary=False,
        )
        dropout_audit = candidate.forecast_condition_dropout_statistics
        self.assertEqual(dropout_audit["observed_sample_count"], 2)
        self.assertEqual(dropout_audit["configured_probability"], 0.10)

        candidate.eval()
        batch = self.batch()
        timestep = torch.tensor([0, 1])
        noise = torch.randn_like(batch["residual_target"])
        conditional = candidate(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
            forecast_condition_strength=1.0,
        )
        neutral = candidate(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
            forecast_condition_strength=0.0,
        )
        self.assertTrue(torch.isfinite(conditional))
        self.assertTrue(torch.isfinite(neutral))
        self.assertNotAlmostEqual(
            float(conditional.detach()), float(neutral.detach()), places=7
        )

    def test_forecast_guidance_interpolates_denoiser_predictions(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        config = self.config("fixed_graph")
        config.update(
            {
                "forecast_condition_dropout_prob": 0.10,
                "use_state_encoder": True,
                "state_feature_dim": 4,
                "state_channels": [4, 8, 16],
            }
        )
        model = Station24DiffusionModel(config, features, adjacency)
        model.eval()
        batch = self.batch(batch_size=1)
        noisy = torch.randn(1, 24, 16)
        timestep = torch.zeros(1, dtype=torch.long)
        kwargs = {
            "forecast_ramps": batch["forecast_ramps"],
            "forecast_revision": batch["forecast_revision"],
            "revision_mask": batch["revision_mask"],
            "recent_error": batch["recent_error"],
            "recent_error_mask": batch["recent_error_mask"],
            "node_state": batch["node_state"],
        }
        neutral = model.diffusion.denoise_step(
            noisy,
            timestep,
            batch["forecast"],
            batch["calendar"],
            batch["lead"],
            forecast_guidance_scale=0.0,
            **kwargs,
        )
        halfway = model.diffusion.denoise_step(
            noisy,
            timestep,
            batch["forecast"],
            batch["calendar"],
            batch["lead"],
            forecast_guidance_scale=0.5,
            **kwargs,
        )
        conditional = model.diffusion.denoise_step(
            noisy,
            timestep,
            batch["forecast"],
            batch["calendar"],
            batch["lead"],
            forecast_guidance_scale=1.0,
            **kwargs,
        )
        self.assertTrue(
            torch.allclose(halfway, 0.5 * (neutral + conditional), atol=1e-6)
        )
        with self.assertRaisesRegex(ValueError, "must be in"):
            model.generate(batch, n_samples=1, forecast_guidance_scale=1.1)

    def test_dual_fixed_graph_adds_only_four_shared_projection_logits(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        baseline_config = self.config("fixed_graph")
        baseline_config.update(
            {
                "spatial_mix_levels": ["bottleneck"],
                "parallel_spatial_fusion_levels": ["encoder_0"],
                "parallel_spatial_adjacency_mode": "fixed",
            }
        )
        dual_config = copy.deepcopy(baseline_config)
        dual_config.update(
            {
                "use_dual_fixed_graph": True,
                "dual_graph_primary_logit_init": 2.0,
                "dual_graph_secondary_logit_init": 0.0,
            }
        )
        baseline = Station24DiffusionModel(baseline_config, features, adjacency)
        candidate = Station24DiffusionModel(
            dual_config,
            features,
            adjacency,
            secondary_adjacency=adjacency.clone(),
        )
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(candidate_count - baseline_count, 4)

        dual_parameters = {
            name: parameter
            for name, parameter in candidate.named_parameters()
            if "dual_graph_logits" in name
        }
        self.assertEqual(len(dual_parameters), 2)
        self.assertEqual(sum(value.numel() for value in dual_parameters.values()), 4)
        for parameter in dual_parameters.values():
            weights = torch.softmax(parameter.detach(), dim=0)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
            self.assertGreater(float(weights[0]), float(weights[1]))

        loss = candidate(self.batch())
        loss.backward()
        for parameter in dual_parameters.values():
            self.assertIsNotNone(parameter.grad)

    def test_ramp_auxiliary_loss_is_finite_and_parameter_free(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        base_config = self.config("fixed_graph")
        auxiliary_config = dict(base_config)
        auxiliary_config.update(
            {
                "ramp_auxiliary_loss_weight": 0.05,
                "ramp_auxiliary_lags": [1, 3, 6],
                "ramp_auxiliary_lag_weights": [0.5, 0.3, 0.2],
            }
        )
        baseline = Station24DiffusionModel(base_config, features, adjacency)
        auxiliary = Station24DiffusionModel(auxiliary_config, features, adjacency)
        auxiliary.load_state_dict(baseline.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in auxiliary.parameters()),
        )
        batch = self.batch()
        batch["residual_scale"] = torch.full_like(batch["residual_target"], 0.2)
        timestep = torch.tensor([0, 1])
        noise = torch.randn_like(batch["residual_target"])
        baseline_loss = baseline(batch, timestep=timestep, noise=noise)
        auxiliary_loss = auxiliary(batch, timestep=timestep, noise=noise)
        auxiliary_epsilon_only = auxiliary(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
        )
        self.assertTrue(torch.isfinite(auxiliary_loss))
        self.assertTrue(
            torch.allclose(auxiliary_epsilon_only, baseline_loss, atol=1e-7)
        )
        self.assertGreater(float(auxiliary_loss.detach()), float(baseline_loss.detach()))
        auxiliary_loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in auxiliary.parameters()
            )
        )

    def test_event_replay_x0_loss_is_parameter_free_finite_and_train_only(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        baseline_config = self.config("fixed_graph")
        candidate_config = copy.deepcopy(baseline_config)
        candidate_config.update(
            {
                "event_x0_magnitude_loss_weight": 0.05,
                "event_x0_timing_loss_weight": 0.005,
                "event_x0_sync_loss_weight": 0.025,
                "event_x0_window_hours": 6,
                "event_x0_error_scale": 0.10,
                "event_x0_timing_temperature": 0.05,
            }
        )
        baseline = Station24DiffusionModel(
            baseline_config, features, adjacency, capacities
        )
        candidate = Station24DiffusionModel(
            candidate_config, features, adjacency, capacities
        )
        candidate.load_state_dict(baseline.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in candidate.parameters()),
        )

        batch = self.batch()
        batch["residual_scale"] = torch.full_like(
            batch["residual_target"], 0.2
        )
        batch["event_active"] = torch.tensor([1.0, 0.0])
        batch["event_start"] = torch.tensor([5, 0])
        batch["event_window_mask"] = torch.zeros(2, 16)
        batch["event_window_mask"][0, 5:11] = 1.0
        batch["event_sync_station_weight"] = torch.zeros(2, 24)
        batch["event_sync_station_weight"][0, :4] = 1.0
        timestep = torch.zeros(2, dtype=torch.long)
        noise = torch.randn_like(batch["residual_target"])
        baseline_loss = baseline(batch, timestep=timestep, noise=noise)
        candidate_loss = candidate(batch, timestep=timestep, noise=noise)
        epsilon_only = candidate(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
        )
        self.assertTrue(torch.isfinite(candidate_loss))
        self.assertTrue(torch.allclose(epsilon_only, baseline_loss, atol=1e-7))
        self.assertGreater(float(candidate_loss.detach()), float(baseline_loss.detach()))
        candidate_loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in candidate.parameters()
            )
        )

    def test_body_tail_expert_preserves_body_and_isolates_tail_gradients(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        body_config = self.config("fixed_graph")
        tail_config = copy.deepcopy(body_config)
        tail_config.update(
            {
                "use_body_tail_experts": True,
                "tail_expert_channels": 4,
                "tail_epsilon_context_hours": 2,
                "tail_gate_channels": 4,
                "tail_gate_prior_probability": 0.99,
                "tail_gate_loss_weight": 0.10,
            }
        )
        body = Station24DiffusionModel(
            body_config, features, adjacency, capacities
        )
        candidate = Station24DiffusionModel(
            tail_config, features, adjacency, capacities
        )
        incompatible = candidate.load_state_dict(body.state_dict(), strict=False)
        self.assertEqual(
            set(incompatible.missing_keys),
            set(candidate.body_tail_state_dict_keys),
        )
        self.assertFalse(incompatible.unexpected_keys)

        batch = self.batch()
        timestep = torch.tensor([0, 1])
        noise = torch.randn_like(batch["residual_target"])
        body.eval()
        candidate.eval()
        body_loss = body(
            batch, timestep=timestep, noise=noise, include_auxiliary=False
        )
        tail_at_initialization = candidate(
            batch, timestep=timestep, noise=noise, include_auxiliary=False
        )
        self.assertTrue(
            torch.allclose(body_loss, tail_at_initialization, atol=1e-7)
        )

        trainable = candidate.configure_body_tail_training()
        self.assertEqual(set(trainable), set(candidate.body_tail_trainable_parameter_names))
        self.assertLess(
            sum(
                parameter.numel()
                for parameter in candidate.parameters()
                if parameter.requires_grad
            ),
            sum(parameter.numel() for parameter in candidate.parameters()) * 0.15,
        )
        batch["event_active"] = torch.tensor([1.0, 0.0])
        batch["event_replay_weight"] = torch.tensor([4.0, 1.0])
        batch["event_window_mask"] = torch.zeros(2, 16)
        batch["event_window_mask"][0, 5:11] = 1.0
        epsilon_support = candidate.body_tail_epsilon_weight(batch)
        self.assertEqual(tuple(epsilon_support.shape), (2, 24, 16))
        self.assertTrue(torch.all(epsilon_support[0, :13, 3:13] == 1.0))
        self.assertEqual(float(epsilon_support[0, 13:].sum()), 0.0)
        self.assertEqual(float(epsilon_support[1].sum()), 0.0)
        candidate.train()
        loss = candidate(batch, timestep=timestep, noise=noise)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        tail_names = set(candidate.body_tail_trainable_parameter_names)
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for name, parameter in candidate.named_parameters()
                if name in tail_names
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in candidate.named_parameters()
                if name not in tail_names
            )
        )

        candidate.eval()
        probability = candidate.tail_risk_probability(batch)
        self.assertEqual(tuple(probability.shape), (2,))
        self.assertTrue(torch.all((probability > 0) & (probability < 1)))
        samples, audit = candidate.generate(
            self.batch(batch_size=1), n_samples=5, return_expert_audit=True
        )
        self.assertEqual(tuple(samples.shape), (1, 5, 24, 16))
        self.assertEqual(tuple(audit["tail_route"].shape), (1, 5))
        self.assertEqual(tuple(audit["tail_probability"].shape), (1,))
        self.assertEqual(tuple(audit["tail_condition_attention"].shape), (1, 6))
        self.assertTrue(
            torch.allclose(
                audit["tail_condition_attention"].sum(dim=1),
                torch.ones(1),
                atol=1e-6,
            )
        )

    def test_sampler_energy_score_uses_final_members_and_only_tail_gradients(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        config = self.config("fixed_graph")
        config.update(
            {
                "use_body_tail_experts": True,
                "tail_expert_channels": 4,
                "tail_epsilon_context_hours": 2,
                "tail_gate_channels": 4,
                "tail_gate_prior_probability": 0.08,
                "tail_gate_loss_weight": 0.10,
                "train_sampler_energy_score_only": True,
                "sampler_energy_score_weight": 0.5,
                "sampler_energy_score_members": 3,
                "sampler_energy_score_steps": 2,
                "sampler_energy_score_backprop_steps": 1,
                "sampler_energy_score_max_issues_per_batch": 1,
            }
        )
        model = Station24DiffusionModel(
            config, features, adjacency, capacities
        )
        trainable = model.configure_body_tail_training()
        self.assertEqual(set(trainable), set(model.body_tail_trainable_parameter_names))

        batch = self.batch()
        batch["actual"] = batch["forecast"] + batch["residual_target"]
        torch.manual_seed(0)
        result = model.sampler_energy_score_loss(batch)
        self.assertEqual(result["score"].ndim, 0)
        self.assertTrue(torch.isfinite(result["score"]))
        self.assertGreaterEqual(float(result["truth_attraction"]), 0.0)
        self.assertGreaterEqual(float(result["member_repulsion"]), 0.0)
        self.assertGreater(float(result["tail_route_rate"]), 0.0)
        self.assertEqual(float(result["issue_count"]), 1.0)
        result["score"].backward()
        tail_names = set(model.body_tail_trainable_parameter_names)
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for name, parameter in model.named_parameters()
                if name in tail_names
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name not in tail_names
            )
        )

    def test_tail_time_localizer_reuses_raw_tail_and_localizes_each_member(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        tail_config = self.config("fixed_graph")
        tail_config.update(
            {
                "use_body_tail_experts": True,
                "tail_expert_channels": 4,
                "tail_epsilon_context_hours": 2,
                "tail_gate_channels": 4,
                "tail_gate_prior_probability": 0.08,
                "tail_gate_loss_weight": 0.10,
            }
        )
        raw_tail = Station24DiffusionModel(
            tail_config, features, adjacency, capacities
        )
        localized_config = copy.deepcopy(tail_config)
        localized_config.update(
            {
                "use_tail_time_localizer": True,
                "train_tail_time_localizer_only": True,
                "tail_time_temperature": 1.0,
                "tail_time_mask_radius_hours": 3,
            }
        )
        localized = Station24DiffusionModel(
            localized_config, features, adjacency, capacities
        )
        incompatible = localized.load_state_dict(
            raw_tail.state_dict(), strict=False
        )
        self.assertEqual(
            set(incompatible.missing_keys),
            set(localized.tail_time_state_dict_keys),
        )
        self.assertFalse(incompatible.unexpected_keys)

        trainable = localized.configure_tail_time_training()
        self.assertEqual(
            set(trainable), set(localized.tail_time_trainable_parameter_names)
        )
        self.assertLess(
            sum(
                parameter.numel()
                for parameter in localized.parameters()
                if parameter.requires_grad
            ),
            1000,
        )
        batch = self.batch()
        batch["event_active"] = torch.tensor([1.0, 0.0])
        batch["event_window_mask"] = torch.zeros(2, 16)
        batch["event_window_mask"][0, 5:11] = 1.0
        loss = localized.tail_time_localization_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        time_names = set(localized.tail_time_trainable_parameter_names)
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for name, parameter in localized.named_parameters()
                if name in time_names
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in localized.named_parameters()
                if name not in time_names
            )
        )

        localized.eval()
        probability = localized.tail_time_probability(batch)
        self.assertEqual(tuple(probability.shape), (2, 16))
        self.assertTrue(
            torch.allclose(probability.sum(dim=1), torch.ones(2), atol=1e-6)
        )
        route = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        starts, masks = localized.sample_tail_time_masks(probability, route)
        self.assertEqual(tuple(starts.shape), (2, 3))
        self.assertEqual(tuple(masks.shape), (2, 3, 16))
        self.assertTrue(torch.all(starts[~route.bool()] == -1))
        self.assertTrue(torch.all(masks[~route.bool()] == 0.0))
        self.assertTrue(torch.all((masks >= 0.0) & (masks <= 1.0)))

        with torch.no_grad():
            localized.denoiser.tail_risk_output.bias.fill_(10.0)
        samples, audit = localized.generate(
            self.batch(batch_size=1), n_samples=4, return_expert_audit=True
        )
        self.assertEqual(tuple(samples.shape), (1, 4, 24, 16))
        self.assertEqual(tuple(audit["tail_time_probability"].shape), (1, 16))
        self.assertEqual(tuple(audit["tail_time_start"].shape), (1, 4))
        self.assertTrue(torch.all(audit["tail_time_start"] >= 0))

    def test_retrieval_dual_tail_preserves_inherited_paths_and_routes_members(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        body_config = self.config("fixed_graph")
        body_config.update(
            {
                "use_body_tail_experts": True,
                "tail_expert_channels": 4,
                "tail_gate_channels": 4,
                "tail_gate_loss_weight": 0.1,
                "tail_gate_prior_probability": 0.1,
            }
        )
        body = Station24DiffusionModel(body_config, features, adjacency)
        retrieval_config = copy.deepcopy(body_config)
        retrieval_config.update(
            {
                "use_retrieval_mismatch_expert": True,
                "train_retrieval_mismatch_only": True,
                "retrieval_encoder_channels": 4,
                "mismatch_expert_channels": 4,
                "mismatch_gate_channels": 4,
                "mismatch_gate_loss_weight": 0.1,
                "mismatch_time_loss_weight": 0.05,
            }
        )
        candidate = Station24DiffusionModel(retrieval_config, features, adjacency)
        incompatible = candidate.load_state_dict(body.state_dict(), strict=False)
        self.assertEqual(
            set(incompatible.missing_keys),
            set(candidate.retrieval_mismatch_state_dict_keys),
        )
        self.assertFalse(incompatible.unexpected_keys)
        trainable = candidate.configure_retrieval_mismatch_training()
        self.assertEqual(
            set(trainable), set(candidate.retrieval_mismatch_trainable_parameter_names)
        )
        self.assertTrue(all("retrieval" in name or "mismatch" in name for name in trainable))

        batch = self.batch()
        batch["retrieval_residual"] = torch.randn(2, 4, 24, 16) * 0.1
        batch["retrieval_distance"] = torch.rand(2, 4)
        batch["retrieval_prior_weight"] = torch.softmax(torch.randn(2, 4), dim=1)
        batch["event_active"] = torch.ones(2)
        batch["event_replay_weight"] = torch.full((2,), 2.0)
        batch["event_window_mask"] = torch.zeros(2, 16)
        batch["event_window_mask"][:, 4:10] = 1.0
        batch["event_start"] = torch.full((2,), 4, dtype=torch.long)
        batch["event_sync_station_weight"] = torch.zeros(2, 24)
        batch["event_sync_station_weight"][:, :13] = 1.0
        loss = candidate(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in candidate.named_parameters():
            if name in set(trainable):
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)

        candidate.eval()
        one = {key: value[:1] for key, value in batch.items()}
        samples, audit = candidate.generate(
            one, n_samples=5, return_expert_audit=True
        )
        self.assertEqual(tuple(samples.shape), (1, 5, 24, 16))
        self.assertEqual(tuple(audit["mismatch_route"].shape), (1, 5))
        self.assertEqual(tuple(audit["mismatch_time_probability"].shape), (1, 16))
        self.assertEqual(tuple(audit["retrieval_attention"].shape), (1, 4, 16))
        self.assertTrue(
            torch.allclose(audit["retrieval_attention"].sum(dim=1), torch.ones(1, 16))
        )

    def test_common_wind_head_and_event_loss_are_lightweight_and_trainable(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        baseline = Station24DiffusionModel(
            self.config("fixed_graph"), features, adjacency, capacities
        )
        config = self.config("fixed_graph")
        config.update(
            {
                "use_forecast_revision": True,
                "use_wind_common_residual_head": True,
                "wind_common_channels": 4,
                "wind_common_gate_init": -1.0,
                "wind_common_event_loss_weight": 0.10,
                "wind_common_event_level_fraction": 0.50,
                "ramp_auxiliary_lags": [1, 3, 6],
                "ramp_auxiliary_lag_weights": [0.5, 0.3, 0.2],
            }
        )
        candidate = Station24DiffusionModel(
            config, features, adjacency, capacities
        )
        increment = sum(p.numel() for p in candidate.parameters()) - sum(
            p.numel() for p in baseline.parameters()
        )
        self.assertGreater(increment, 0)
        self.assertLess(increment, 1000)
        batch = self.batch()
        batch["residual_scale"] = torch.full_like(
            batch["residual_target"], 0.2
        )
        batch["loss_weight"] = torch.ones_like(batch["residual_target"])
        batch["event_time_weight"] = torch.ones(2, 16)
        batch["event_time_weight"][:, 8:] = 3.0
        loss = candidate(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(candidate.denoiser.wind_common_head[-1].weight.grad)
        self.assertIsNotNone(candidate.denoiser.wind_common_gate.grad)
        self.assertAlmostEqual(candidate.wind_common_gate_value, 0.2689414, places=5)

    def test_checkpoint_saves_event_weighting_without_run_scope(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel
        from train_station24 import create_ema, save_checkpoint

        features, adjacency = synthetic_static()
        config = self.config("fixed_graph")
        config["use_wind_common_residual_head"] = True
        config["wind_common_channels"] = 4
        model = Station24DiffusionModel(config, features, adjacency)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        event_weighting = {
            "fit_split": "train",
            "future_actual_used_as_condition": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                create_ema(model),
                optimizer,
                {"experiment": {"variant": "checkpoint_smoke"}},
                {"method": "per_station_std", "scale": [1.0] * 24},
                1,
                0.5,
                0.4,
                sum(parameter.numel() for parameter in model.parameters()),
                None,
                event_weighting,
                None,
                {},
            )
            saved = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        self.assertEqual(saved["event_weighting"], event_weighting)
        self.assertEqual(saved["condition_variant"], "checkpoint_smoke")
        self.assertIsNotNone(saved["wind_common_gate_value"])

    def test_generation_checkpoint_state_selection_is_explicit(self):
        from generate_station24 import select_checkpoint_state

        raw = {"weight": torch.tensor([1.0])}
        ema = {"weight": torch.tensor([2.0])}
        checkpoint = {
            "model_state_dict": raw,
            "ema_model_state_dict": ema,
        }
        selected, key = select_checkpoint_state(checkpoint, "raw")
        self.assertIs(selected, raw)
        self.assertEqual(key, "model_state_dict")
        selected, key = select_checkpoint_state(checkpoint, "ema")
        self.assertIs(selected, ema)
        self.assertEqual(key, "ema_model_state_dict")
        selected, key = select_checkpoint_state(
            {"model_state_dict": raw}, "ema"
        )
        self.assertIs(selected, raw)
        self.assertEqual(key, "model_state_dict")
        with self.assertRaises(ValueError):
            select_checkpoint_state(checkpoint, "unknown")

    def test_ema_warmup_tracks_short_trained_adapter(self):
        from train_station24 import ema_decay_for_step

        warmup = {
            "enabled": True,
            "inv_gamma": 1.0,
            "power": 0.75,
            "min_decay": 0.0,
            "update_after_step": 0,
        }
        first = ema_decay_for_step(0.999, 1, warmup)
        short_run = ema_decay_for_step(0.999, 380, warmup)
        long_run = ema_decay_for_step(0.999, 10000, warmup)
        self.assertAlmostEqual(first, 1.0 - 2.0 ** -0.75, places=7)
        self.assertAlmostEqual(short_run, 0.9884, places=3)
        self.assertEqual(long_run, 0.999)
        self.assertEqual(ema_decay_for_step(0.999, 1, None), 0.999)
        with self.assertRaises(ValueError):
            ema_decay_for_step(0.999, 1, {"enabled": True, "power": 0.0})

    def test_type_gated_graph_has_exact_relation_gates(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        model = Station24DiffusionModel(
            self.config("type_gated_graph"), features, adjacency
        )
        self.assertEqual(
            set(model.denoiser.spatial_block.gate_values()),
            {"wind_wind", "solar_solar", "wind_solar"},
        )

    def test_multiscale_graph_mixing_is_lightweight_and_trainable(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        baseline = Station24DiffusionModel(
            self.config("fixed_graph"), features, adjacency
        )
        config = self.config("fixed_graph")
        config["spatial_mix_levels"] = [
            "encoder_0",
            "encoder_1",
            "bottleneck",
        ]
        candidate = Station24DiffusionModel(config, features, adjacency)
        self.assertEqual(baseline.spatial_mix_levels, ("bottleneck",))
        self.assertEqual(
            candidate.spatial_mix_levels,
            ("encoder_0", "encoder_1", "bottleneck"),
        )
        # Each block adds GroupNorm affine, 1x1 projection, and one graph gate:
        # (4x4 + 3x4 + 1) + (8x8 + 3x8 + 1) = 118.
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(candidate_count - baseline_count, 118)
        self.assertEqual(set(baseline.spatial_gate_values), {"all"})
        self.assertEqual(
            set(candidate.spatial_gate_values),
            {"encoder_0/all", "encoder_1/all", "bottleneck/all"},
        )

        loss = candidate(self.batch())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for block in candidate.denoiser.encoder_spatial_blocks.values():
            self.assertIsNotNone(block.graph_gate.grad)
            self.assertIsNotNone(block.projection.weight.grad)
        generated = candidate.generate(self.batch(batch_size=1), n_samples=2)
        self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))

        restored = Station24DiffusionModel(
            self.config("fixed_graph"), features, adjacency
        )
        restored.load_state_dict(baseline.state_dict(), strict=True)

    def test_multiscale_graph_mixing_rejects_unknown_level(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        config = self.config("fixed_graph")
        config["spatial_mix_levels"] = ["encoder_0", "decoder_0"]
        with self.assertRaisesRegex(ValueError, "unsupported spatial_mix_levels"):
            Station24DiffusionModel(config, features, adjacency)

    def test_cdsg_lite_parallel_fusion_is_local_and_lightweight(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        baseline = Station24DiffusionModel(
            self.config("fixed_graph"), features, adjacency
        )
        config = self.config("fixed_graph")
        config["parallel_spatial_fusion_levels"] = ["encoder_0"]
        candidate = Station24DiffusionModel(config, features, adjacency)
        self.assertEqual(baseline.parallel_spatial_fusion_levels, ())
        self.assertEqual(candidate.parallel_spatial_fusion_levels, ("encoder_0",))
        # C=4: norm 8 + projection 20 + prior 1 + local gate (12+1) = 42.
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(candidate_count - baseline_count, 42)

        candidate.reset_parallel_spatial_gate_statistics()
        candidate.eval()
        loss = candidate(self.batch())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        fusion = candidate.denoiser.parallel_spatial_blocks["encoder_0"]
        self.assertIsNotNone(fusion.spatial_projection.weight.grad)
        self.assertIsNotNone(fusion.gate_projection.weight.grad)
        self.assertIsNotNone(fusion.gate_prior.grad)
        statistics = candidate.parallel_spatial_gate_statistics
        self.assertEqual(
            set(statistics),
            {
                "encoder_0/prior",
                "encoder_0/observed_mean",
                "encoder_0/observed_std",
                "encoder_0/observed_min",
                "encoder_0/observed_max",
            },
        )
        generated = candidate.generate(self.batch(batch_size=1), n_samples=2)
        self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))

    def test_cdsg_lite_parallel_fusion_rejects_double_graph_at_same_level(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        config = self.config("fixed_graph")
        config["spatial_mix_levels"] = ["encoder_0", "bottleneck"]
        config["parallel_spatial_fusion_levels"] = ["encoder_0"]
        with self.assertRaisesRegex(ValueError, "cannot share levels"):
            Station24DiffusionModel(config, features, adjacency)

    def test_cdsg_lite_hybrid_dynamic_graph_is_sparse_light_and_trainable(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        fixed_config = self.config("fixed_graph")
        fixed_config["parallel_spatial_fusion_levels"] = ["encoder_0"]
        fixed = Station24DiffusionModel(fixed_config, features, adjacency)

        dynamic_config = self.config("fixed_graph")
        dynamic_config.update(
            {
                "parallel_spatial_fusion_levels": ["encoder_0"],
                "parallel_spatial_adjacency_mode": "hybrid_dynamic",
                "dynamic_graph_embedding_dim": 16,
                "dynamic_graph_top_k": 3,
                "dynamic_graph_temperature": 1.0,
                "dynamic_graph_mix_gate_init": -3.0,
            }
        )
        dynamic = Station24DiffusionModel(dynamic_config, features, adjacency)
        fixed_count = sum(parameter.numel() for parameter in fixed.parameters())
        dynamic_count = sum(parameter.numel() for parameter in dynamic.parameters())
        # C=4 and d=16: dynamic MLP 416, static projection 80,
        # LayerNorm 32, and one residual-mix gate.
        self.assertEqual(dynamic_count - fixed_count, 529)
        self.assertEqual(
            dynamic.parallel_spatial_adjacency_mode, "hybrid_dynamic"
        )

        dynamic.reset_parallel_spatial_gate_statistics()
        dynamic.eval()
        loss = dynamic(self.batch())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        fusion = dynamic.denoiser.parallel_spatial_blocks["encoder_0"]
        self.assertIsNotNone(fusion.dynamic_mix_gate.grad)
        self.assertIsNotNone(fusion.dynamic_node_encoder[0].weight.grad)
        self.assertIsNotNone(fusion.static_node_projection.weight.grad)
        statistics = dynamic.parallel_spatial_gate_statistics
        self.assertIn("encoder_0/dynamic_mix", statistics)
        self.assertIn("encoder_0/off_geographic_mass", statistics)
        moments = dynamic.parallel_spatial_adjacency_moments["encoder_0"]
        self.assertEqual(tuple(moments["mean"].shape), (24, 24))
        self.assertEqual(tuple(moments["std"].shape), (24, 24))

    def test_cdsg_lite_hybrid_dynamic_graph_accepts_state_v1_context(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        config = self.config("fixed_graph")
        config.update(
            {
                "parallel_spatial_fusion_levels": ["encoder_0"],
                "parallel_spatial_adjacency_mode": "hybrid_dynamic",
                "dynamic_graph_embedding_dim": 8,
                "dynamic_graph_top_k": 3,
                "use_state_encoder": True,
                "state_feature_dim": 4,
                "state_channels": [2, 4, 8],
            }
        )
        model = Station24DiffusionModel(config, features, adjacency)
        loss = model(self.batch())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        generated = model.generate(self.batch(batch_size=1), n_samples=2)
        self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))

    def test_condition_variants_forward_backward_and_sample(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        variants = {
            "revision_ramp": (True, False, {"ramp", "revision"}),
            "history_ramp": (False, True, {"ramp", "recent_error"}),
            "revision_history_ramp": (
                True,
                True,
                {"ramp", "revision", "recent_error"},
            ),
        }
        for _, (use_revision, use_history, expected_gates) in variants.items():
            config = self.config("fixed_graph")
            config.update(
                {
                    "use_forecast_ramps": True,
                    "forecast_ramp_lags": [1, 3, 6],
                    "use_forecast_revision": use_revision,
                    "use_recent_error": use_history,
                    "recent_error_hours": 24,
                    "condition_gate_init": -1.0,
                }
            )
            model = Station24DiffusionModel(config, features, adjacency)
            loss = model(self.batch())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            generated = model.generate(self.batch(batch_size=1), n_samples=2)
            self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))
            self.assertEqual(set(model.condition_gate_values), expected_gates)

    def test_forecast_correction_direct_and_decomposed_are_causal_and_trainable(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        counts = {}
        for mode, expected_components in [("direct", 1), ("decomposed", 6)]:
            config = self.config("fixed_graph")
            config.update(
                {
                    "use_forecast_correction": True,
                    "forecast_correction_mode": mode,
                    "forecast_correction_channels": 8,
                    "forecast_correction_use_revision": True,
                    "forecast_correction_use_recent_error": True,
                    "forecast_correction_max_abs": 1.0,
                    "forecast_correction_loss_weight": 0.25,
                    "forecast_correction_huber_beta": 0.05,
                    "recent_error_hours": 24,
                }
            )
            model = Station24DiffusionModel(config, features, adjacency)
            batch = self.batch()
            correction = model.predict_forecast_correction(batch)
            self.assertEqual(tuple(correction.shape), (2, 24, 16))
            self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
            components = model.forecast_correction_head._forecast_components(
                batch["forecast"].reshape(2 * 24, 16)
            )
            self.assertEqual(tuple(components.shape), (2 * 24, expected_components, 16))
            loss = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.forecast_correction_head.output.weight.grad)
            with torch.no_grad():
                model.forecast_correction_head.output.bias.fill_(0.1)
            shifted = model.predict_forecast_correction(batch)
            self.assertGreater(float(shifted.detach().mean()), 0.0)
            generated = model.generate(self.batch(batch_size=1), n_samples=2)
            self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))
            counts[mode] = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(counts["decomposed"], counts["direct"])

    def test_state_v1_lightweight_encoder_shapes_forward_and_zero_init(self):
        from src.models.station_conditioned_diffusion import Station24DiffusionModel

        features, adjacency = synthetic_static()
        capacities = torch.linspace(10.0, 100.0, 24)
        base = Station24DiffusionModel(
            self.config("fixed_graph"), features, adjacency, capacities
        )
        config = self.config("fixed_graph")
        config.update(
            {
                "use_state_encoder": True,
                "state_feature_dim": 4,
                "state_channels": [2, 4, 8],
                "state_global_gate_init": -1.0,
                "state_film_gate_init": -1.0,
            }
        )
        model = Station24DiffusionModel(config, features, adjacency, capacities)
        outputs = model.denoiser.state_encoder(self.batch()["node_state"])
        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [(2, 24, 2, 16), (2, 24, 4, 8), (2, 24, 8, 4)],
        )
        for block in [
            *model.denoiser.encoder_blocks,
            model.denoiser.bottleneck,
            *model.denoiser.decoder_blocks,
        ]:
            self.assertTrue(torch.equal(block.state_affine.weight, torch.zeros_like(block.state_affine.weight)))
            self.assertTrue(torch.equal(block.state_affine.bias, torch.zeros_like(block.state_affine.bias)))
        loss = model(self.batch())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        generated = model.generate(self.batch(batch_size=1), n_samples=2)
        self.assertEqual(tuple(generated.shape), (1, 2, 24, 16))
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters())
            - sum(parameter.numel() for parameter in base.parameters()),
            5000,
        )
        self.assertTrue(model.state_gate_values)


class Station24DatasetTests(unittest.TestCase):
    def make_data(self, root: Path):
        stations = pd.DataFrame(
            {
                "channel_index": np.arange(24),
                "data_type": ["wind"] * 13 + ["solar"] * 11,
                "station_id": np.arange(100, 124),
                "FARM_NAME": [f"s{i}" for i in range(24)],
                "capacity_mw": np.linspace(10, 100, 24),
                "longitude": np.linspace(115, 121, 24),
                "latitude": np.linspace(35, 38, 24),
            }
        )
        stations.to_csv(root / "station_order.csv", index=False)
        features = np.zeros((24, 5), dtype=np.float32)
        features[:13, 0] = 1
        features[13:, 1] = 1
        np.save(root / "station_features.npy", features)
        np.save(root / "station_adjacency.npy", np.eye(24, dtype=np.float32))
        (root / "export_metadata.json").write_text("{}", encoding="utf-8")
        for split, count in [("train", 2), ("val", 1), ("test", 1)]:
            forecast = np.full((count, 168, 24), 0.4, dtype=np.float32)
            residual = np.linspace(-0.1, 0.1, count * 168 * 24, dtype=np.float32)
            residual = residual.reshape(count, 168, 24)
            actual = forecast + residual
            np.save(root / f"{split}_forecast.npy", forecast)
            np.save(root / f"{split}_actual.npy", actual)
            np.save(root / f"{split}_residual.npy", residual)
            np.save(root / f"{split}_time_mark.npy", np.zeros((count, 168, 8), dtype=np.float32))
            np.save(root / f"{split}_lead_mark.npy", np.zeros((count, 168, 2), dtype=np.float32))
            np.save(root / f"{split}_fill_mask.npy", np.zeros((count, 168, 24), dtype=np.uint8))
            issue_dates = pd.date_range("2025-01-01", periods=count, freq="D")
            pd.DataFrame(
                {
                    "issue_date": issue_dates.strftime("%Y-%m-%d"),
                    "target_start": issue_dates.strftime("%Y-%m-%d 00:00:00"),
                    "target_end": (issue_dates + pd.Timedelta(days=6, hours=23)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            ).to_csv(
                root / f"{split}_issue_dates.csv", index=False
            )

    def test_scale_is_train_only_and_item_shapes_are_station_first(self):
        from station_dataset import (
            StationForecastDataset,
            build_station_daylight_mask,
            fit_station_residual_scale,
            fit_station_state_thresholds,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            scale = fit_station_residual_scale(root)
            dataset = StationForecastDataset(root, "val", scale)
            item = dataset[0]
            self.assertEqual(scale["fit_split"], "train")
            self.assertFalse(scale["center"])
            self.assertEqual(tuple(item["forecast"].shape), (24, 168))
            self.assertEqual(tuple(item["calendar"].shape), (8, 168))
            self.assertEqual(tuple(item["lead"].shape), (2, 168))
            self.assertEqual(tuple(item["forecast_ramps"].shape), (24, 3, 168))
            self.assertEqual(tuple(item["forecast_revision"].shape), (24, 168))
            self.assertEqual(tuple(item["recent_error"].shape), (24, 24))
            self.assertEqual(tuple(item["residual_scale"].shape), (24, 168))
            train_dataset = StationForecastDataset(root, "train", scale)
            conditioned = train_dataset[1]
            self.assertEqual(float(conditioned["revision_mask"][:, :144].min()), 1.0)
            self.assertEqual(float(conditioned["revision_mask"][:, 144:].max()), 0.0)
            self.assertEqual(float(conditioned["recent_error_mask"].min()), 1.0)
            self.assertLess(
                float(
                    torch.max(
                        torch.abs(
                            item["actual"] - item["forecast"] - item["residual"]
                        )
                    )
                ),
                1e-6,
            )
            daylight, audit = build_station_daylight_mask(root, "val")
            self.assertEqual(daylight.shape, (1, 168, 24))
            self.assertTrue(daylight[:, :, :13].all())
            self.assertFalse(audit["uses_power_or_actual"])

            thresholds = fit_station_state_thresholds(root, ramp_lags=(3, 6))
            state_config = {
                "use_state_encoder": True,
                "state_ramp_lags": [3, 6],
                "state_clip": 3.0,
            }
            state_dataset = StationForecastDataset(
                root,
                "val",
                scale,
                condition_config=state_config,
                state_thresholds=thresholds,
            )
            state_before = state_dataset[0]["node_state"].clone()
            self.assertEqual(tuple(state_before.shape), (24, 4, 168))
            self.assertTrue(torch.isfinite(state_before).all())
            self.assertEqual(thresholds["fit_split"], "train")
            self.assertEqual(
                thresholds["future_state_source"], "current_issued_forecast_only"
            )
            val_actual = np.load(root / "val_actual.npy")
            changed_actual = val_actual + 0.2
            np.save(root / "val_actual.npy", changed_actual)
            val_forecast = np.load(root / "val_forecast.npy")
            np.save(root / "val_residual.npy", changed_actual - val_forecast)
            state_after = StationForecastDataset(
                root,
                "val",
                scale,
                condition_config=state_config,
                state_thresholds=thresholds,
            )[0]["node_state"]
            self.assertTrue(torch.equal(state_before, state_after))

    def test_wind_conditional_scale_is_train_fitted_and_invertible(self):
        from station_dataset import StationForecastDataset, fit_station_residual_scale

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            train_forecast = np.broadcast_to(
                np.linspace(0.05, 0.95, 168, dtype=np.float32)[None, :, None],
                (2, 168, 24),
            ).copy()
            hourly_wave = (
                0.05
                + 0.10
                * np.sin(np.arange(168, dtype=np.float32) * 2.0 * np.pi / 24.0)
            )[None, :, None]
            train_residual = train_forecast * hourly_wave
            np.save(root / "train_forecast.npy", train_forecast)
            np.save(root / "train_residual.npy", train_residual)
            np.save(root / "train_actual.npy", train_forecast + train_residual)
            scale = fit_station_residual_scale(
                root,
                method="wind_factorized_condition_std",
                condition_config={
                    "ramp_lag": 3,
                    "factor_iterations": 3,
                    "factor_clip": [0.5, 2.0],
                },
            )
            self.assertEqual(scale["fit_split"], "train")
            self.assertEqual(scale["method"], "wind_factorized_condition_std")
            self.assertFalse(scale["future_actual_used_as_condition"])
            self.assertNotIn("Infinity", json.dumps(scale))
            dataset = StationForecastDataset(root, "train", scale)
            item = dataset[1]
            reconstructed = item["residual_target"] * item["residual_scale"]
            self.assertTrue(torch.allclose(reconstructed, item["residual"], atol=1e-6))
            base = torch.tensor(scale["scale"], dtype=torch.float32)
            self.assertTrue(
                torch.allclose(
                    item["residual_scale"][13:],
                    base[13:, None].expand(-1, 168),
                )
            )
            self.assertGreater(
                float(torch.max(item["residual_scale"][:13]) - torch.min(item["residual_scale"][:13])),
                0.0,
            )

    def test_event_weighting_is_train_only_and_bounded(self):
        from station_dataset import (
            StationForecastDataset,
            fit_station_event_weighting,
            fit_station_residual_scale,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            scale = fit_station_residual_scale(root)
            config = {
                "event_max_weight": 3.0,
                "event_negative_quantiles": [0.90, 0.99],
                "event_ramp_quantiles": [0.90, 0.99],
                "event_ramp_lags": [1, 3, 6],
            }
            specification = fit_station_event_weighting(root, config)
            train_item = StationForecastDataset(
                root, "train", scale, event_weighting=specification
            )[0]
            val_item = StationForecastDataset(
                root, "val", scale, event_weighting=specification
            )[0]
            self.assertEqual(specification["fit_split"], "train")
            self.assertFalse(specification["future_actual_used_as_condition"])
            self.assertGreaterEqual(float(train_item["event_time_weight"].min()), 1.0)
            self.assertLessEqual(float(train_item["event_time_weight"].max()), 3.0)
            self.assertAlmostEqual(float(train_item["loss_weight"].mean()), 1.0, places=5)
            self.assertTrue(torch.equal(val_item["loss_weight"], torch.ones_like(val_item["loss_weight"])))
            self.assertTrue(torch.equal(val_item["event_time_weight"], torch.ones_like(val_item["event_time_weight"])))

    def test_independent_event_replay_is_deduplicated_and_train_only(self):
        from station_dataset import (
            StationForecastDataset,
            fit_station_event_replay,
            fit_station_residual_scale,
            get_station_dataloader,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            scale = fit_station_residual_scale(root)
            specification = fit_station_event_replay(
                root,
                {
                    "event_replay_window_hours": 6,
                    "event_replay_merge_gap_hours": 24,
                    "event_replay_quantiles": [0.50, 0.75],
                    "event_replay_weights": [2.0, 4.0],
                },
            )
            self.assertEqual(specification["fit_split"], "train")
            self.assertFalse(specification["future_actual_used_as_condition"])
            self.assertFalse(specification["ordinary_epsilon_loss_reweighted"])
            self.assertEqual(
                specification["representative_issue_count"],
                specification["independent_event_count"],
            )
            self.assertLessEqual(
                specification["representative_issue_count"],
                specification["overlapping_issue_count"],
            )
            train_dataset = StationForecastDataset(
                root, "train", scale, event_replay=specification
            )
            val_dataset = StationForecastDataset(
                root, "val", scale, event_replay=specification
            )
            self.assertGreater(
                sum(float(train_dataset[i]["event_active"]) for i in range(2)),
                0.0,
            )
            self.assertEqual(float(val_dataset[0]["event_active"]), 0.0)
            self.assertFalse(val_dataset.condition_audit["event_replay_applied"])
            loader, _ = get_station_dataloader(
                root,
                "train",
                scale,
                batch_size=1,
                seed=2027,
                event_replay=specification,
            )
            self.assertEqual(loader.sampler.num_samples, 2)

    def test_forecast_mismatch_weighting_targets_event_and_context(self):
        from station_dataset import build_station_event_loss_weights

        actual = np.zeros((24, 168), dtype=np.float32)
        residual = np.zeros_like(actual)
        valid = np.ones_like(actual)
        residual[0, 10] = -1.0
        specification = {
            "method": "train_forecast_mismatch_event_weighting_v1",
            "fit_split": "train",
            "future_actual_used_as_condition": False,
            "applied_to_validation_or_generation": False,
            "max_weight": 3.0,
            "aggregate_level_thresholds": [0.2, 0.8],
            "node_level_thresholds": [[0.2, 0.8] for _ in range(13)],
            "ramp_lags": [1, 3, 6],
            "aggregate_ramp_mismatch_thresholds": {
                str(lag): [0.2, 0.8] for lag in [1, 3, 6]
            },
            "node_ramp_mismatch_thresholds": {
                str(lag): [[0.2, 0.8] for _ in range(13)]
                for lag in [1, 3, 6]
            },
            "context_hours": 2,
            "use_aggregate_events": False,
            "use_node_events": True,
            "wind_station_indices": list(range(13)),
            "wind_capacity_weights": [1.0 / 13.0] * 13,
        }
        train_weight, _ = build_station_event_loss_weights(
            actual, residual, valid, specification, "train"
        )
        val_weight, _ = build_station_event_loss_weights(
            actual, residual, valid, specification, "val"
        )
        self.assertGreater(float(train_weight[0, 10]), float(train_weight[0, 0]))
        self.assertGreater(float(train_weight[0, 8]), float(train_weight[0, 0]))
        self.assertGreater(float(train_weight[0, 12]), float(train_weight[0, 0]))
        self.assertAlmostEqual(float((train_weight * valid).mean()), 1.0, places=5)
        np.testing.assert_array_equal(val_weight, np.ones_like(val_weight))

    def test_generation_cli_smoke_uses_ema_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            train_root = root / "training"
            output_dir = root / "result"
            data_dir.mkdir()
            self.make_data(data_dir)
            model_config = {
                "architecture": "station24_resunet",
                "spatial_mode": "fixed_graph",
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
                "use_forecast_correction": True,
                "forecast_correction_mode": "direct",
                "forecast_correction_channels": 4,
                "forecast_correction_use_revision": True,
                "forecast_correction_use_recent_error": True,
                "forecast_correction_max_abs": 1.0,
                "forecast_correction_loss_weight": 0.25,
                "forecast_correction_huber_beta": 0.05,
                "forecast_condition_dropout_prob": 0.10,
                "use_forecast_ramps": False,
                "forecast_ramp_lags": [3, 6],
                "use_recent_error": True,
                "recent_error_hours": 24,
                "use_state_encoder": True,
                "state_feature_dim": 4,
                "state_channels": [2, 4, 8],
                "state_ramp_lags": [3, 6],
                "state_low_quantile": 0.20,
                "state_high_quantile": 0.90,
                "state_ramp_quantile": 0.90,
                "state_clip": 3.0,
                "use_extreme_event_weighting": True,
                "event_weighting_method": "forecast_mismatch_v1",
                "event_max_weight": 3.0,
                "event_level_quantiles": [0.90, 0.99],
                "event_ramp_mismatch_quantiles": [0.90, 0.99],
                "event_ramp_lags": [1, 3, 6],
                "event_context_hours": 2,
                "event_use_aggregate_events": True,
                "event_use_node_events": True,
                "ramp_auxiliary_loss_weight": 0.05,
                "ramp_auxiliary_lags": [1, 3, 6],
                "ramp_auxiliary_lag_weights": [0.5, 0.3, 0.2],
            }
            config = {
                "experiment": {"name": "smoke", "family": "test"},
                "data": {"data_path": str(data_dir)},
                "target": {
                    "type": "corrected_residual",
                    "residual_scaling": {
                        "method": "wind_factorized_condition_std",
                        "epsilon": 1e-4,
                        "ramp_lag": 3,
                        "factor_iterations": 2,
                        "factor_clip": [0.5, 2.0],
                    },
                },
                "model": model_config,
                "train": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 1,
                    "lr": 1e-3,
                    "epochs": 1,
                    "weight_decay": 0.0,
                    "patience": 1,
                    "validation_every": 1,
                    "min_delta": 0.0,
                    "gradient_clip_norm": 1.0,
                    "ema_decay": 0.9,
                    "mixed_precision": False,
                    "save_every": 50,
                    "seed": 3,
                    "validation_seed": 4,
                    "num_workers": 0,
                },
                "evaluation": {
                    "n_samples": 2,
                    "generation_seed": 9,
                    "issue_batch_size": 1,
                    "member_chunk_size": 1,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    "train_station24.py",
                    "--config",
                    str(config_path),
                    "--data-path",
                    str(data_dir),
                    "--output-root",
                    str(train_root),
                    "--allow-cpu",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            runs = list(train_root.iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            self.assertTrue((run_dir / "checkpoints" / "model_best.pt").is_file())
            self.assertTrue((run_dir / "state_thresholds.json").is_file())
            self.assertTrue((run_dir / "event_weighting.json").is_file())
            subprocess.run(
                [
                    sys.executable,
                    "generate_station24.py",
                    "--run-dir",
                    str(run_dir),
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--split",
                    "val",
                    "--n-samples",
                    "2",
                    "--member-chunk-size",
                    "1",
                    "--forecast-guidance-scale",
                    "0.75",
                    "--allow-cpu",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            metrics = (output_dir / "metrics.json").read_text(encoding="utf-8")
            self.assertIn('"spatial_mode": "fixed_graph"', metrics)
            self.assertIn('"spatial_mix_levels": [', metrics)
            self.assertIn('"parallel_spatial_fusion_levels": []', metrics)
            self.assertIn(
                '"residual_scaling_method": "wind_factorized_condition_std"',
                metrics,
            )
            self.assertIn('"ramp_auxiliary_loss_weight": 0.05', metrics)
            self.assertIn('"forecast_guidance_scale": 0.75', metrics)
            self.assertIn('"forecast_condition_dropout_prob": 0.1', metrics)
            self.assertIn('"forecast_correction_mode": "direct"', metrics)
            self.assertIn('"checkpoint_state_source": "ema"', metrics)
            self.assertIn(
                '"checkpoint_state_key": "ema_model_state_dict"', metrics
            )
            self.assertIn(
                '"checkpoint_validation_objective_type": '
                '"diffusion_epsilon_plus_forecast_correction_huber"',
                metrics,
            )
            self.assertIn(
                '"event_weighting_method": '
                '"train_forecast_mismatch_event_weighting_v1"',
                metrics,
            )
            self.assertTrue((output_dir / "station_daylight_mask.npy").is_file())
            self.assertTrue(
                (output_dir / "forecast_correction_normalized.npy").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "generated_stochastic_residual_standardized.npy"
                ).is_file()
            )
            scale = np.load(output_dir / "generated_residual_normalized.npy")
            scenario = np.load(output_dir / "actual_scenarios_raw_normalized.npy")
            forecast = np.load(output_dir / "forecast_data_normalized.npy")
            np.testing.assert_allclose(
                scenario,
                forecast[:, None, :, :] + scale,
                atol=1e-6,
            )

            comparison_inputs = []
            for mode in ["none", "fixed_graph", "type_gated_graph"]:
                copied = root / f"result_{mode}"
                shutil.copytree(output_dir, copied)
                copied_metrics = json.loads(
                    (copied / "metrics.json").read_text(encoding="utf-8")
                )
                copied_metrics["run"]["spatial_mode"] = mode
                (copied / "metrics.json").write_text(
                    json.dumps(copied_metrics), encoding="utf-8"
                )
                comparison_inputs.append(str(copied))
            comparison_dir = root / "comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_spatial_ablation.py",
                    *comparison_inputs,
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(comparison_dir),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue((comparison_dir / "comparison_report.md").is_file())
            self.assertTrue(
                (comparison_dir / "figures" / "typical_scenario_envelopes.png").is_file()
            )

            condition_inputs = []
            for variant in [
                "revision_ramp",
                "history_ramp",
                "revision_history_ramp",
            ]:
                copied = root / f"condition_{variant}"
                shutil.copytree(output_dir, copied)
                copied_metrics = json.loads(
                    (copied / "metrics.json").read_text(encoding="utf-8")
                )
                copied_metrics["run"]["condition_variant"] = variant
                copied_metrics["run"]["condition_gate_values"] = {}
                (copied / "metrics.json").write_text(
                    json.dumps(copied_metrics), encoding="utf-8"
                )
                condition_inputs.append(str(copied))
            condition_comparison = root / "condition_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_condition_ablation.py",
                    *condition_inputs,
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(condition_comparison),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                (condition_comparison / "comparison_summary.csv").is_file()
            )
            self.assertTrue(
                (
                    condition_comparison
                    / "figures"
                    / "ramp_and_extreme_metrics.png"
                ).is_file()
            )

            state_inputs = []
            for variant in ["ramp36_control", "state_v1_fixed_graph"]:
                copied = root / f"state_v1_{variant}"
                shutil.copytree(output_dir, copied)
                copied_metrics = json.loads(
                    (copied / "metrics.json").read_text(encoding="utf-8")
                )
                copied_metrics["run"]["spatial_mode"] = "fixed_graph"
                copied_metrics["run"]["condition_variant"] = variant
                copied_metrics["run"]["condition_gate_values"] = {}
                copied_metrics["run"]["state_gate_values"] = {}
                (copied / "metrics.json").write_text(
                    json.dumps(copied_metrics), encoding="utf-8"
                )
                state_inputs.append(str(copied))
            state_comparison = root / "state_v1_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_state_v1.py",
                    *state_inputs,
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(state_comparison),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                (state_comparison / "comparison_summary.csv").is_file()
            )
            self.assertTrue(
                (
                    state_comparison
                    / "figures"
                    / "typical_scenario_envelopes.png"
                ).is_file()
            )

            multiscale_inputs = []
            for variant, levels in [
                ("state_v1_fixed_graph", ["bottleneck"]),
                (
                    "state_v1_multiscale_graph",
                    ["encoder_0", "encoder_1", "bottleneck"],
                ),
            ]:
                copied = root / f"multiscale_{variant}"
                shutil.copytree(output_dir, copied)
                copied_metrics = json.loads(
                    (copied / "metrics.json").read_text(encoding="utf-8")
                )
                copied_metrics["run"]["condition_variant"] = variant
                copied_metrics["run"]["spatial_mode"] = "fixed_graph"
                copied_metrics["run"]["spatial_mix_levels"] = levels
                copied_metrics["run"]["spatial_gate_values"] = {"all": 0.25}
                (copied / "metrics.json").write_text(
                    json.dumps(copied_metrics), encoding="utf-8"
                )
                multiscale_inputs.append(str(copied))
            multiscale_comparison = root / "multiscale_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_multiscale_2a.py",
                    *multiscale_inputs,
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(multiscale_comparison),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                (multiscale_comparison / "comparison_summary.csv").is_file()
            )
            self.assertTrue(
                (
                    multiscale_comparison
                    / "figures"
                    / "typical_scenario_envelopes.png"
                ).is_file()
            )

            cdsg_inputs = []
            for variant, parallel_levels in [
                ("state_v1_fixed_graph", []),
                ("state_v1_cdsg_lite_parallel", ["encoder_0"]),
            ]:
                copied = root / f"cdsg_lite_{variant}"
                shutil.copytree(output_dir, copied)
                copied_metrics = json.loads(
                    (copied / "metrics.json").read_text(encoding="utf-8")
                )
                copied_metrics["run"]["condition_variant"] = variant
                copied_metrics["run"]["spatial_mix_levels"] = ["bottleneck"]
                copied_metrics["run"][
                    "parallel_spatial_fusion_levels"
                ] = parallel_levels
                copied_metrics["run"]["parallel_spatial_gate_statistics"] = (
                    {"encoder_0/observed_mean": 0.25} if parallel_levels else {}
                )
                (copied / "metrics.json").write_text(
                    json.dumps(copied_metrics), encoding="utf-8"
                )
                cdsg_inputs.append(str(copied))
            cdsg_comparison = root / "cdsg_lite_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_multiscale_2a.py",
                    *cdsg_inputs,
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(cdsg_comparison),
                    "--candidate-variant",
                    "state_v1_cdsg_lite_parallel",
                    "--candidate-label",
                    "CDSG-lite parallel fusion",
                    "--candidate-spatial-levels",
                    "bottleneck",
                    "--candidate-parallel-levels",
                    "encoder_0",
                    "--figure-prefix",
                    "cdsg_lite_2b",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                (cdsg_comparison / "comparison_summary.csv").is_file()
            )
            self.assertTrue(
                (cdsg_comparison / "figures" / "cdsg_lite_2b_key_metrics.png").is_file()
            )

            dynamic_result = root / "cdsg_lite_state_v1_cdsg_lite_hybrid_dynamic"
            shutil.copytree(output_dir, dynamic_result)
            dynamic_metrics = json.loads(
                (dynamic_result / "metrics.json").read_text(encoding="utf-8")
            )
            dynamic_metrics["run"].update(
                {
                    "condition_variant": "state_v1_cdsg_lite_hybrid_dynamic",
                    "spatial_mix_levels": ["bottleneck"],
                    "parallel_spatial_fusion_levels": ["encoder_0"],
                    "parallel_spatial_adjacency_mode": "hybrid_dynamic",
                    "parallel_spatial_gate_statistics": {
                        "encoder_0/observed_mean": 0.25,
                        "encoder_0/dynamic_mix": 0.05,
                    },
                }
            )
            (dynamic_result / "metrics.json").write_text(
                json.dumps(dynamic_metrics), encoding="utf-8"
            )
            comparison_2b_2c = root / "cdsg_lite_2b_2c_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "tools/compare_station24_multiscale_2a.py",
                    cdsg_inputs[1],
                    str(dynamic_result),
                    "--data-path",
                    str(data_dir),
                    "--output-dir",
                    str(comparison_2b_2c),
                    "--baseline-variant",
                    "state_v1_cdsg_lite_parallel",
                    "--candidate-variant",
                    "state_v1_cdsg_lite_hybrid_dynamic",
                    "--baseline-parallel-levels",
                    "encoder_0",
                    "--candidate-parallel-levels",
                    "encoder_0",
                    "--baseline-parallel-adjacency",
                    "fixed",
                    "--candidate-parallel-adjacency",
                    "hybrid_dynamic",
                    "--baseline-spatial-levels",
                    "bottleneck",
                    "--candidate-spatial-levels",
                    "bottleneck",
                    "--figure-prefix",
                    "cdsg_lite_2b_2c",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                (
                    comparison_2b_2c
                    / "figures"
                    / "cdsg_lite_2b_2c_key_metrics.png"
                ).is_file()
            )


class Station24EvaluationTests(unittest.TestCase):
    def test_metrics_are_finite_and_perfect_interval_covers(self):
        from station_evaluation import evaluate_station_scenarios

        rng = np.random.default_rng(7)
        actual = rng.uniform(0.1, 0.9, size=(2, 168, 24)).astype(np.float32)
        samples = np.repeat(actual[:, None, :, :], 5, axis=1)
        raw = samples.copy()
        forecast = actual.copy()
        stations = pd.DataFrame(
            {
                "channel_index": np.arange(24),
                "station_id": np.arange(24),
                "data_type": ["wind"] * 13 + ["solar"] * 11,
                "FARM_NAME": [f"s{i}" for i in range(24)],
                "capacity_mw": np.ones(24),
            }
        )
        adjacency = np.eye(24, dtype=np.float32)
        adjacency[np.arange(23), np.arange(1, 24)] = 0.5
        adjacency[np.arange(1, 24), np.arange(23)] = 0.5
        metrics, station_frame, lead_frame = evaluate_station_scenarios(
            samples,
            raw,
            actual,
            forecast,
            stations,
            adjacency,
            daylight_mask=np.ones_like(actual, dtype=bool),
        )
        self.assertEqual(len(station_frame), 24)
        self.assertEqual(len(lead_frame), 28)
        self.assertAlmostEqual(metrics["station_average"]["all"]["crps"], 0.0)
        for level in [80, 90, 95]:
            self.assertAlmostEqual(
                metrics["station_average"]["all"][f"coverage_{level}"], 1.0
            )
            self.assertAlmostEqual(
                metrics["station_average"]["solar_daylight"][f"coverage_{level}"],
                1.0,
            )
        self.assertIn("coverage_90_daylight", station_frame.columns)
        self.assertTrue(np.isfinite(metrics["joint"]["energy_score_pu"]))
        self.assertEqual(metrics["joint"]["energy_score_member_count"], 5)


if __name__ == "__main__":
    unittest.main()
