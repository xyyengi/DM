import copy
from pathlib import Path
import unittest

import torch
import yaml

from station_lightweight_tail import expected_config, validate_config
from station_dataset import load_station_static_data
from src.models.station_conditioned_diffusion import Station24DiffusionModel


class AuxiliaryTimestepCompensationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.static = load_station_static_data("diffusion_input_station")

    def build(self, config):
        s = self.static
        return Station24DiffusionModel(
            config, s["station_features"], s["station_adjacency"],
            s["station_capacities"], s["station_adjacency"]
        )

    def test_candidate_changes_only_declared_mechanism(self):
        control = expected_config()
        candidate = expected_config(auxiliary_timestep_compensation=True)
        saved = yaml.safe_load(Path(
            "configs/station24_lightweight_joint_tail_aux_timestep_compensation_168h.yaml"
        ).read_text(encoding="utf-8"))
        validate_config(saved)
        self.assertEqual(saved, candidate)
        left, right = copy.deepcopy(control), copy.deepcopy(candidate)
        left["experiment"] = right["experiment"] = {}
        right["model"].pop("auxiliary_timestep_compensation")
        right["model"].pop("auxiliary_timestep_compensation_version")
        self.assertEqual(left, right)

    def test_schedule_is_fixed_mean_one_and_checkpoint_compatible(self):
        control = self.build(expected_config()["model"])
        candidate = self.build(
            expected_config(auxiliary_timestep_compensation=True)["model"]
        )
        weight = candidate.diffusion.auxiliary_timestep_weight
        self.assertAlmostEqual(float(weight.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weight.min()), 0.3480401, places=5)
        self.assertAlmostEqual(float(weight.max()), 2.7843205, places=5)
        self.assertFalse(control.diffusion.auxiliary_timestep_compensation)
        self.assertTrue(candidate.diffusion.auxiliary_timestep_compensation)
        self.assertEqual(set(control.state_dict()), set(candidate.state_dict()))
        self.assertEqual(
            sum(p.numel() for p in candidate.parameters() if p.requires_grad),
            20588,
        )


if __name__ == "__main__":
    unittest.main()
