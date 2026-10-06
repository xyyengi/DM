import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from station_lightweight_tail import (
    FROZEN_AUXILIARY_ALPHAS,
    expected_config,
    validate_config,
)
from station_dataset import build_station_daylight_mask, load_station_static_data
from src.models.station_conditioned_diffusion import Station24DiffusionModel


class LightweightWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.static = load_station_static_data("diffusion_input_station")

    def build(self, config):
        s = self.static
        return Station24DiffusionModel(config, s["station_features"], s["station_adjacency"], s["station_capacities"], s["station_adjacency"])

    def test_saved_recipe_matches_reference(self):
        c = yaml.safe_load(Path("configs/station24_lightweight_joint_tail_v2_fair_168h.yaml").read_text())
        validate_config(c)
        for section, key, value in [("model", "event_balanced_ramp_loss_weight", .19),
                                    ("model", "joint_multiresidual_channels", 32),
                                    ("train", "lr", .002), ("model", "use_jstd_tail", True)]:
            altered = copy.deepcopy(c)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(altered)

    def test_ramp_selection_recipe_changes_only_declared_selector(self):
        original = expected_config()
        proposed = expected_config(ramp_selection=True)
        saved = yaml.safe_load(Path(
            "configs/station24_lightweight_joint_tail_v2_ramp_selection_168h.yaml"
        ).read_text())
        validate_config(saved)
        self.assertEqual(saved, proposed)
        for section in ("data", "target", "train", "evaluation"):
            self.assertEqual(original[section], proposed[section])
        changed_model = {
            key for key in set(original["model"]) | set(proposed["model"])
            if original["model"].get(key) != proposed["model"].get(key)
        }
        self.assertEqual(
            changed_model, {"event_balanced_ramp_selection_version"}
        )
        old_model = self.build(original["model"])
        new_model = self.build(proposed["model"])
        self.assertEqual(
            {name: tuple(value.shape) for name, value in old_model.state_dict().items()},
            {name: tuple(value.shape) for name, value in new_model.state_dict().items()},
        )
        self.assertEqual(
            sum(value.numel() for value in old_model.parameters() if value.requires_grad),
            sum(value.numel() for value in new_model.parameters() if value.requires_grad),
        )

    def test_frozen_stage1a_alpha_changes_only_declared_fields(self):
        control = expected_config()
        for alpha, weights in FROZEN_AUXILIARY_ALPHAS.items():
            candidate = expected_config(auxiliary_alpha=alpha)
            validate_config(candidate)
            self.assertEqual(
                tuple(candidate["model"][key] for key in (
                    "event_balanced_ramp_loss_weight",
                    "event_balanced_shape_loss_weight",
                    "event_balanced_slow_loss_weight",
                )),
                weights,
            )
            left = copy.deepcopy(control)
            right = copy.deepcopy(candidate)
            for value in (left, right):
                value["experiment"] = {}
                value["model"].pop("auxiliary_strength_alpha", None)
                for key in (
                    "event_balanced_ramp_loss_weight",
                    "event_balanced_shape_loss_weight",
                    "event_balanced_slow_loss_weight",
                ):
                    value["model"].pop(key)
            self.assertEqual(left, right)
        with self.assertRaises(ValueError):
            expected_config(auxiliary_alpha=.80)

    def test_daylight_mask_uses_no_actual_and_keeps_wind_valid(self):
        mask, audit = build_station_daylight_mask("diffusion_input_station", "train")
        wind = self.static["station_features"][:, 0].bool().numpy()
        solar = self.static["station_features"][:, 1].bool().numpy()
        self.assertTrue(mask[..., wind].all())
        self.assertGreater(mask[..., solar].mean(), 0.0)
        self.assertLess(mask[..., solar].mean(), 1.0)
        self.assertFalse(audit["uses_power_or_actual"])

    def test_version_defaults_preserve_legacy(self):
        legacy = yaml.safe_load(Path("configs/station24_joint_multiresidual_tail_v1_168h.yaml").read_text())["model"]
        m = self.build(legacy)
        self.assertFalse(m.lightweight_v2)
        self.assertEqual(m.diffusion.jstd_decomposition_loss_weight, .20)
        legacy["joint_multiresidual_decomposition_loss_weight"] = 0
        with self.assertRaises(ValueError):
            self.build(legacy)

    def test_network_layout_unchanged_and_frozen_train_state(self):
        c = expected_config()["model"]
        m = self.build(c)
        legacy = self.build(yaml.safe_load(Path("configs/station24_joint_multiresidual_tail_v1_168h.yaml").read_text())["model"])
        left = {n: tuple(v.shape) for n, v in m.denoiser.joint_multiresidual_tail.state_dict().items()}
        right = {n: tuple(v.shape) for n, v in legacy.denoiser.joint_multiresidual_tail.state_dict().items()}
        self.assertEqual(left, right)
        self.assertEqual(sum(v.numel() for v in m.parameters() if v.requires_grad), 20588)
        m.train()
        self.assertFalse(m.denoiser.training)
        self.assertTrue(m.denoiser.joint_multiresidual_tail.training)
        self.assertTrue(all(not x.training for n, x in m.named_modules() if "joint_multiresidual_tail" not in n))
        m.eval()
        self.assertFalse(m.denoiser.joint_multiresidual_tail.training)

    def test_no_event_epsilon_mask(self):
        m = self.build(expected_config()["model"])
        b = {"forecast": torch.zeros(2, 24, 168), "jstd_event_active": torch.zeros(2)}
        self.assertTrue(torch.equal(m.body_tail_epsilon_weight(b), torch.ones_like(b["forecast"])))
        self.assertTrue(torch.equal(m.tail_risk_probability(b), torch.ones(2)))

    def test_existing_merger_exact_400_100_order(self):
        # Exercise the real merger with cheap synthetic arrays; only metric
        # calculation is mocked. No model training or formal generation.
        from tools import merge_station24_independent_tail_members as merge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = [root / "body", root / "tail"]
            for folder, count, value in zip(folders, (500, 100), (1., 2.)):
                folder.mkdir()
                for name in merge.MEMBER_ARRAYS:
                    np.save(folder / name, np.full((1, count, 8, 24), value, dtype=np.float32))
                for name in merge.STATIC_ARRAYS:
                    np.save(folder / name, np.zeros((1, 8, 24), dtype=np.float32))
                meta = dict(split="val", generation_seed=424242, physical_projection="test", architecture="station24_resunet", spatial_mode="fixed_graph", n_samples=count)
                (folder / "generation_metadata.json").write_text(json.dumps(meta))
            output = root / "merged"
            args = ["merge", "--body-results", str(folders[0]), "--tail-results", str(folders[1]), "--output-dir", str(output), "--body-member-limit", "400", "--tail-member-limit", "100"]
            with patch.object(sys, "argv", args), patch.object(merge, "evaluate_station_scenarios", return_value=({}, None, None)) as evaluate_mock, patch.object(merge, "save_evaluation"):
                merge.main()
                # Mock call history otherwise retains Windows mmap file handles.
                evaluate_mock.reset_mock()
            for name in merge.MEMBER_ARRAYS:
                a = np.load(output / name)
                self.assertEqual(a.shape[1], 500)
                np.testing.assert_array_equal(a[:, :400], np.load(folders[0] / name)[:, :400])
                np.testing.assert_array_equal(a[:, 400:], np.load(folders[1] / name))
            np.testing.assert_array_equal(np.load(output / "tail_expert_route.npy"), np.concatenate([np.zeros((1,400)), np.ones((1,100))], axis=1))


if __name__ == "__main__":
    unittest.main()
