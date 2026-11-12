"""Regression tests for the recovered BTB bootstrap calculation."""
import unittest

import numpy as np
import pandas as pd

from tools import reproduce_btb_sensitivity as sensitivity


class BTBSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(sensitivity.INPUT)

    def test_all_indices_and_error_bars_reproduced(self):
        result, report = sensitivity.reproduce()
        self.assertEqual(result.shape, (3, 8))
        self.assertEqual(report['bootstrap_draws'], 2000)
        self.assertEqual(report['random_seed'], 20260901)
        self.assertEqual(report['checked_values'], 18)
        self.assertLess(report['maximum_absolute_difference'], 1e-12)

    def test_input_order_does_not_change_bootstrap(self):
        result = sensitivity.calculate(self.frame.sample(frac=1, random_state=17))
        reference = pd.read_csv(sensitivity.REFERENCE)
        np.testing.assert_allclose(result[sensitivity.COLUMNS], reference[sensitivity.COLUMNS],
                                   atol=1e-12, rtol=0)

    def test_invalid_counts_and_probabilities_rejected(self):
        index = self.frame.index[self.frame.system == 'btb_mol'][0]
        for column, value in [('trace_count', 0), ('target_count', 1.5),
                              ('target_count', 33), ('persistent_assembly_probability', -1)]:
            frame = self.frame.copy()
            frame[column] = frame[column].astype(float)
            frame.loc[index, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                sensitivity.calculate(frame)

    def test_duplicate_and_incomplete_grid_rejected(self):
        btb = self.frame.loc[self.frame.system == 'btb_mol']
        for frame in (btb.iloc[:-1], pd.concat([btb.iloc[:-1], btb.iloc[:1]])):
            with self.assertRaises(ValueError):
                sensitivity.calculate(frame)

    def test_variance_indices_for_single_active_axis(self):
        module = sensitivity.source_module()
        cube = np.broadcast_to(np.arange(4.)[:, None, None], (4, 3, 2))
        first, total = module.variance_indices(cube)
        np.testing.assert_allclose(first, [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(total, [1, 0, 0], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
