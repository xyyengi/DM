import unittest

import torch


class ModelFactoryCompatibilityTests(unittest.TestCase):
    def legacy_config(self):
        return {
            "input_channels": 14,
            "in_channels": 14,
            "out_channels": 3,
            "base_channels": 8,
            "n_intervals": 10,
            "num_steps": 4,
            "beta_start": 1e-4,
            "beta_end": 0.04,
            "schedule": "linear",
        }

    def test_missing_architecture_is_legacy_only_compatibility_default(self):
        from diff_models_multivariate import MultiChannelCSDI
        from src.models import build_model, resolve_architecture

        config = self.legacy_config()
        model = build_model(config, torch.device("cpu"))

        self.assertEqual(resolve_architecture(config), "v4_legacy")
        self.assertIsInstance(model, MultiChannelCSDI)
        self.assertEqual(model.architecture, "v4_legacy")

    def test_explicit_v5_architecture_is_not_inferred_from_channels(self):
        from src.models import build_model

        config = self.legacy_config()
        config.update({
            "architecture": "v5_t",
            "input_channels": 14,
            "in_channels": 14,
            "sequence_length": 168,
            "num_layers": 3,
            "channel_multipliers": [1, 2, 4],
            "use_sequence_condition": False,
            "target_type": "residual",
        })

        with self.assertRaisesRegex(ValueError, "in_channels=3"):
            build_model(config, torch.device("cpu"))

    def test_v5_t_and_v5_tf_require_matching_condition_contracts(self):
        from src.models import build_model

        base = {
            "in_channels": 3,
            "out_channels": 3,
            "sequence_length": 168,
            "base_channels": 4,
            "num_layers": 3,
            "channel_multipliers": [1, 2, 4],
            "target_type": "residual",
            "num_steps": 3,
        }
        for architecture, condition_enabled in (("v5_t", False), ("v5_tf", True)):
            config = dict(base)
            config.update({
                "architecture": architecture,
                "use_sequence_condition": condition_enabled,
            })
            model = build_model(config, torch.device("cpu"))
            self.assertEqual(model.architecture, architecture)
            self.assertEqual(model.use_sequence_condition, condition_enabled)

        invalid = dict(base)
        invalid.update({
            "architecture": "v5_tf",
            "use_sequence_condition": False,
        })
        with self.assertRaisesRegex(ValueError, "use_sequence_condition=True"):
            build_model(invalid, torch.device("cpu"))

    def test_unknown_architecture_fails_instead_of_guessing(self):
        from src.models import resolve_architecture

        with self.assertRaisesRegex(ValueError, "Unsupported model architecture"):
            resolve_architecture({"architecture": "v5_magic", "in_channels": 3})


if __name__ == "__main__":
    unittest.main()
