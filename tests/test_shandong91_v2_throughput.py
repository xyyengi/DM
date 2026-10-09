import unittest

import torch

from tools.tune_shandong91_v2_cuda_throughput import parse_candidates, select_fastest
from train_shandong91_v2 import loader_kwargs


class Shandong91V2ThroughputTests(unittest.TestCase):
    def test_candidate_parser_is_positive_sorted_unique(self):
        self.assertEqual(parse_candidates("16,2,8,2,4"), [2, 4, 8, 16])
        with self.assertRaises(ValueError):
            parse_candidates("2,0")

    def test_selection_uses_throughput_and_memory_headroom(self):
        rows = [
            {"status": "PASS", "samples_per_second": 3.0, "peak_reserved_gb": 10.0},
            {"status": "PASS", "samples_per_second": 9.0, "peak_reserved_gb": 30.0},
            {"status": "OOM", "samples_per_second": 99.0, "peak_reserved_gb": 0.0},
        ]
        selected = select_fastest(rows, "samples_per_second", total_gb=32.0, max_fraction=0.9)
        self.assertEqual(selected["samples_per_second"], 3.0)

    def test_loader_options_preserve_cpu_compatibility(self):
        cpu = loader_kwargs({"num_workers": 0}, torch.device("cpu"))
        self.assertEqual(cpu, {"num_workers": 0, "pin_memory": False})
        cuda = loader_kwargs(
            {"num_workers": 8, "persistent_workers": True, "prefetch_factor": 3},
            torch.device("cuda:0"),
        )
        self.assertEqual(cuda["num_workers"], 8)
        self.assertTrue(cuda["pin_memory"])
        self.assertTrue(cuda["persistent_workers"])
        self.assertEqual(cuda["prefetch_factor"], 3)


if __name__ == "__main__":
    unittest.main()
