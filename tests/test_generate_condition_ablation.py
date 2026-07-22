import unittest

import torch

from generate import apply_condition_ablation, generate_scenarios


class _CaptureModel:
    def __init__(self):
        self.batch = None

    def eval(self):
        return self

    def generate(self, batch, n_samples):
        self.batch = batch
        batch_size, _, length = batch["forecast_3ch"].shape
        return torch.zeros(batch_size, n_samples, 3, length)


class GenerateConditionAblationTests(unittest.TestCase):
    def make_batch(self):
        return {
            "forecast_3ch": torch.ones(1, 3, 168),
            "calendar_8ch": torch.full((1, 8, 168), 2.0),
            "time_encoding": torch.full((1, 8, 168), 2.0),
            "residual_3ch": torch.full((1, 3, 168), 3.0),
            "actual_3ch": torch.full((1, 3, 168), 4.0),
        }

    def test_forecast_ablation_does_not_mutate_original_batch(self):
        batch = self.make_batch()

        ablated = apply_condition_ablation(batch, "forecast")

        self.assertTrue(torch.equal(batch["forecast_3ch"], torch.ones(1, 3, 168)))
        self.assertEqual(torch.count_nonzero(ablated["forecast_3ch"]).item(), 0)
        self.assertTrue(torch.equal(ablated["calendar_8ch"], batch["calendar_8ch"]))

    def test_calendar_ablation_zeros_both_compatibility_keys(self):
        batch = self.make_batch()

        ablated = apply_condition_ablation(batch, "calendar")

        self.assertEqual(torch.count_nonzero(ablated["calendar_8ch"]).item(), 0)
        self.assertEqual(torch.count_nonzero(ablated["time_encoding"]).item(), 0)
        self.assertTrue(torch.equal(ablated["forecast_3ch"], batch["forecast_3ch"]))

    def test_generation_collects_original_forecast_while_model_sees_ablation(self):
        batch = self.make_batch()
        model = _CaptureModel()

        samples, forecast, residual, actual = generate_scenarios(
            model,
            [batch],
            torch.device("cpu"),
            n_samples=2,
            condition_ablation="forecast_calendar",
        )

        self.assertEqual(samples.shape, (1, 2, 3, 168))
        self.assertEqual(torch.count_nonzero(model.batch["forecast_3ch"]).item(), 0)
        self.assertEqual(torch.count_nonzero(model.batch["calendar_8ch"]).item(), 0)
        self.assertTrue((forecast == 1.0).all())
        self.assertTrue((residual == 3.0).all())
        self.assertTrue((actual == 4.0).all())

    def test_invalid_ablation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported condition ablation"):
            apply_condition_ablation(self.make_batch(), "risk")


if __name__ == "__main__":
    unittest.main()
