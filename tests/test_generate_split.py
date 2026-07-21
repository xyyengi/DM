import unittest

from generate import resolve_result_folder


class GenerateSplitTests(unittest.TestCase):
    def test_validation_gets_separate_default_folder(self):
        self.assertEqual(
            resolve_result_folder("outputs/run", None, "val", 20, 2026),
            "outputs/run_val_n20_seed2026",
        )

    def test_test_keeps_legacy_default_folder(self):
        self.assertEqual(
            resolve_result_folder("outputs/run", None, "test", 20, 2026),
            "outputs/run",
        )

    def test_explicit_output_wins_for_any_split(self):
        self.assertEqual(
            resolve_result_folder("outputs/run", "outputs/custom", "val", 50, 9),
            "outputs/custom",
        )


if __name__ == "__main__":
    unittest.main()
