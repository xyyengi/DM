import unittest

import numpy as np

from tools.audit_station24_jstd_mechanism import (
    _average_precision,
    _binary_auc,
    _correction_localization,
    _event_localization,
)


class JSTDMechanismAuditMetricTests(unittest.TestCase):
    def test_gate_metrics_identify_perfect_ranking(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.float32)
        scores = np.asarray([0.1, 0.8, 0.2, 0.9], dtype=np.float32)
        self.assertAlmostEqual(_binary_auc(labels, scores), 1.0)
        self.assertAlmostEqual(_average_precision(labels, scores), 1.0)

    def test_localization_separates_inside_and_outside(self):
        support = np.asarray([[False, True, True, False]])
        mask = np.asarray([[0.1, 0.8, 0.6, 0.1]], dtype=np.float32)
        correction = np.asarray([[1.0, 4.0, 4.0, 1.0]], dtype=np.float32)
        mask_metrics = _event_localization(mask, support)
        correction_metrics = _correction_localization(correction, support)
        self.assertGreater(mask_metrics["mask_inside_outside_ratio"], 5.0)
        self.assertAlmostEqual(
            correction_metrics["correction_event_energy_fraction"], 0.8
        )
        self.assertAlmostEqual(
            correction_metrics["correction_outside_event_fraction"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
