import unittest

import numpy as np

from tools import audit_fig5d_lineage as audit


class Fig5DLineageTests(unittest.TestCase):
    def test_pre_event_to_post_event_block_offset(self):
        self.assertEqual(audit.raw_location(1), (0, 'all_molecules_0.0.mol2', 0))
        self.assertEqual(audit.raw_location(2), (1, 'all_molecules_1.0.mol2', 0))
        self.assertEqual(audit.raw_location(10001), (10000, 'all_molecules_1.0.mol2', 9999))
        self.assertEqual(audit.raw_location(10002), (10001, 'all_molecules_2.0.mol2', 0))
        with self.assertRaises(ValueError):
            audit.raw_location(0)

    def test_composition_must_match(self):
        hist = {'Zr': np.array([1.0, 0.0])}
        self.assertIsNone(audit.first_matching_uid((1, 2, hist), {1: (2, 1, hist)}))

    def test_original_rule_is_first_not_best_match(self):
        hist = {'C': np.array([1.0, 0.0])}
        earlier = {'C': np.array([0.9, 0.1])}
        self.assertEqual(audit.first_matching_uid((0, 1, hist), {2: (0, 1, hist), 1: (0, 1, earlier)}), 1)

    def test_empty_histogram_similarity(self):
        self.assertEqual(audit.similarity({}, {}), 0)

    def test_all_figure_snapshots_replay_offline(self):
        report = audit.verify()
        self.assertEqual(report['classification_rows_linked'], 284)
        self.assertEqual(report['replayed_cells_this_run'], 3688)
        self.assertEqual(report['representatives_byte_matched_to_group04'], 90)


if __name__ == '__main__':
    unittest.main()
