import unittest
from pathlib import Path

import torch
import yaml

from tools.tune_shandong91_v2_cuda_throughput import parse_candidates, select_fastest
from train_shandong91_v2 import loader_kwargs


class Shandong91V2ThroughputTests(unittest.TestCase):
    def test_station24_runtime_parity_config(self):
        config = yaml.safe_load(Path(
            "configs/shandong91/raw_body_v2_faithful24_solar_nonnegative_station24_runtime.yaml"
        ).read_text("utf-8"))
        training = config["training"]
        self.assertEqual(training["batch_size"], 8)
        self.assertEqual(training["gradient_accumulation_steps"], 2)
        self.assertEqual(training["effective_batch_size"], 16)
        self.assertEqual(training["num_workers"], 4)
        self.assertTrue(training["mixed_precision"])
        self.assertEqual(config["generation"]["n_samples"], 500)
        self.assertEqual(config["generation"]["member_chunk"], 10)

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
