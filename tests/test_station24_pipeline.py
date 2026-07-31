import tempfile
import unittest
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
        return {
            "residual_target": torch.randn(batch_size, 24, 16),
            "forecast": torch.rand(batch_size, 24, 16),
            "calendar": torch.randn(batch_size, 8, 16),
            "lead": torch.rand(batch_size, 2, 16),
            "valid_mask": torch.ones(batch_size, 24, 16),
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
            pd.DataFrame(
                {
                    "issue_date": ["2025-01-01"] * count,
                    "target_start": ["2025-01-01 00:00:00"] * count,
                    "target_end": ["2025-01-07 23:00:00"] * count,
                }
            ).to_csv(
                root / f"{split}_issue_dates.csv", index=False
            )

    def test_scale_is_train_only_and_item_shapes_are_station_first(self):
        from station_dataset import (
            StationForecastDataset,
            build_station_daylight_mask,
            fit_station_residual_scale,
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
            }
            config = {
                "experiment": {"name": "smoke", "family": "test"},
                "data": {"data_path": str(data_dir)},
                "target": {
                    "type": "residual",
                    "residual_scaling": {"epsilon": 1e-4},
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
                    "--allow-cpu",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            metrics = (output_dir / "metrics.json").read_text(encoding="utf-8")
            self.assertIn('"spatial_mode": "fixed_graph"', metrics)
            self.assertTrue((output_dir / "station_daylight_mask.npy").is_file())

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
            samples, raw, actual, forecast, stations, adjacency
        )
        self.assertEqual(len(station_frame), 24)
        self.assertEqual(len(lead_frame), 21)
        self.assertAlmostEqual(metrics["station_average"]["all"]["crps"], 0.0)
        self.assertAlmostEqual(
            metrics["station_average"]["all"]["coverage_90"], 1.0
        )
        self.assertTrue(np.isfinite(metrics["joint"]["energy_score_pu"]))


if __name__ == "__main__":
    unittest.main()
