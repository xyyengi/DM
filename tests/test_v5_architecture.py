import unittest

import torch


def tiny_config(architecture="v5_tf"):
    return {
        "architecture": architecture,
        "in_channels": 3,
        "out_channels": 3,
        "sequence_length": 168,
        "base_channels": 8,
        "num_layers": 3,
        "channel_multipliers": [1, 2, 4],
        "group_norm_groups": 4,
        "dropout": 0.0,
        "timestep_embedding_dim": 16,
        "position_embedding_dim": 8,
        "use_sequence_condition": architecture == "v5_tf",
        "target_type": "residual",
        "num_steps": 4,
    }


class V5ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.x_t = torch.randn(2, 3, 168)
        self.forecast = torch.randn(2, 3, 168)
        self.calendar = torch.randn(2, 8, 168)
        self.positions = torch.arange(168, dtype=torch.float32).repeat(2, 1)

    def make_denoiser(self, architecture="v5_tf"):
        from src.models.v5_conditioned_diffusion import V5ConditionalUNet1D

        return V5ConditionalUNet1D(tiny_config(architecture)).eval()

    def forward_tf(self, model, timestep=None, **overrides):
        values = {
            "forecast": self.forecast,
            "calendar": self.calendar,
            "relative_positions": self.positions,
        }
        values.update(overrides)
        return model(
            self.x_t,
            torch.tensor([1, 1]) if timestep is None else timestep,
            **values,
        )

    def test_noisy_state_path_is_exactly_three_channels(self):
        model = self.make_denoiser()

        self.forward_tf(model)

        self.assertEqual(model.state_stem.in_channels, 3)
        self.assertEqual(model.last_noisy_state_shape, (2, 3, 168))
        self.assertFalse(hasattr(model, "get_time_features"))

    def test_timestep_changes_prediction_for_fixed_state_and_conditions(self):
        model = self.make_denoiser()

        at_t0 = self.forward_tf(model, timestep=torch.tensor([0, 0]))
        at_t3 = self.forward_tf(model, timestep=torch.tensor([3, 3]))

        self.assertFalse(torch.allclose(at_t0, at_t3))

    def test_condition_encoder_has_expected_multiscale_shapes(self):
        model = self.make_denoiser()

        output = self.forward_tf(model)

        self.assertEqual(tuple(output.shape), (2, 3, 168))
        self.assertEqual(
            model.last_condition_shapes,
            [(2, 8, 168), (2, 16, 84), (2, 32, 42)],
        )
        self.assertEqual(
            model.condition_encoder.last_input_shapes,
            {
                "forecast": (2, 3, 168),
                "calendar": (2, 8, 168),
                "relative_positions": (2, 168),
            },
        )

    def test_each_resblock_receives_timestep_and_sequence_film(self):
        model = self.make_denoiser()

        self.forward_tf(model)

        self.assertEqual(len(model.residual_blocks), 6)
        for block in model.residual_blocks:
            shapes = block.last_modulation_shapes
            self.assertEqual(shapes["time_gamma"], (2, block.out_channels, 1))
            self.assertEqual(shapes["time_beta"], (2, block.out_channels, 1))
            self.assertIsNotNone(shapes["condition_feature"])
            self.assertEqual(shapes["gamma"][0:2], (2, block.out_channels))
            self.assertGreater(shapes["gamma"][2], 1)

    def test_forecast_calendar_and_position_are_independent_effective_inputs(self):
        model = self.make_denoiser()
        baseline = self.forward_tf(model)

        changed_forecast = self.forward_tf(model, forecast=self.forecast + 0.5)
        changed_calendar = self.forward_tf(model, calendar=self.calendar + 0.5)
        changed_positions = self.forward_tf(
            model, relative_positions=self.positions + 1.0
        )

        self.assertFalse(torch.allclose(baseline, changed_forecast))
        self.assertFalse(torch.allclose(baseline, changed_calendar))
        self.assertFalse(torch.allclose(baseline, changed_positions))

    def test_v5_t_ignores_all_sequence_conditions(self):
        model = self.make_denoiser("v5_t")
        timestep = torch.tensor([1, 1])

        without_conditions = model(self.x_t, timestep)
        with_unused_conditions = model(
            self.x_t,
            timestep,
            forecast=self.forecast,
            calendar=self.calendar,
            relative_positions=self.positions,
        )

        self.assertTrue(torch.equal(without_conditions, with_unused_conditions))
        self.assertEqual(model.last_condition_shapes, [])
        for block in model.residual_blocks:
            self.assertIsNone(block.last_modulation_shapes["condition_feature"])

    def test_num_layers_controls_topology_and_must_match_multipliers(self):
        from src.models.v5_conditioned_diffusion import V5ConditionalUNet1D

        config = tiny_config()
        config["num_layers"] = 4
        with self.assertRaisesRegex(ValueError, "channel_multipliers"):
            V5ConditionalUNet1D(config)


if __name__ == "__main__":
    unittest.main()
