from __future__ import annotations

import unittest

import torch

from src.models.shandong91_conditioned_diffusion import Shandong91HeterogeneousRawBody


class Shandong91RawBodyContractTests(unittest.TestCase):
    def test_lead_and_tail_are_rejected(self) -> None:
        features = torch.zeros(91, 10)
        features[:, 4:7] = 1
        adjacency = torch.eye(91, dtype=torch.bool)
        base = {
            "node_count": 91, "resource_count": 3, "sequence_length": 168,
            "base_channels": 8, "num_layers": 3, "channel_multipliers": [1, 2, 4],
            "group_norm_groups": 8,
        }
        with self.assertRaisesRegex(ValueError, "lead"):
            Shandong91HeterogeneousRawBody({**base, "use_lead_condition": True}, features, adjacency)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            Shandong91HeterogeneousRawBody({**base, "use_body_tail_experts": True}, features, adjacency)


if __name__ == "__main__":
    unittest.main()
