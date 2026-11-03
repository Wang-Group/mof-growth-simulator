"""S20 reanalysis, completeness checks and finite-window statistics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools import reproduce_s20 as s20


class S20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = s20.read_csv(s20.FOLDER / 'source_trace_summary.csv')

    def altered(self, field, value):
        rows = list(self.records)
        rows[0] = dict(rows[0], **{field: value})
        return rows

    def test_full_s20_reproduction(self):
        report, summaries, matched, printed = s20.reproduce()
        self.assertEqual(report['trajectory_records'], 27648)
        self.assertEqual(report['condition_groups'], 864)
        self.assertEqual(report['outcome_counts'], {'target_reached': 15040, 'observation_window': 12608})
        self.assertEqual(report['high_precision_reference']['checked_values'], 90)
        self.assertEqual(report['display_rows_matched'], 9)
        self.assertEqual(printed[6], ['4-tBu', '0.1', '0/32', '0/32', 'Not reached'])
        self.assertIsNone(matched[6]['median_growth_passage_time'])
        self.assertIsNone(matched[6]['50_over_median_growth_time'])
        self.assertEqual(sum(row['n'] for row in summaries), 27648)

    def test_duplicate_run_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate or empty run_id'):
            s20.summarize(self.records + [self.records[0]])

    def test_duplicate_replicate_with_different_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate replicate'):
            s20.summarize(self.records + [dict(self.records[0], run_id='different_id')])

    def test_missing_replicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete replicate set'):
            s20.summarize(self.records[1:])

    def test_missing_condition_rejected(self):
        first = self.records[0]
        rows = [r for r in self.records if any(r[k] != first[k] for k in s20.KEYS)]
        with self.assertRaisesRegex(ValueError, 'Incomplete assay-condition grid'):
            s20.summarize(rows)

    def test_invalid_record_rejected(self):
        for field, value in [('status', 'implementation_failure'), ('model_time', 'nan'),
                             ('model_time', '-1'), ('model_time', '101'),
                             ('valid_physical_outcome', 'False'), ('failure_reason', 'crash'),
                             ('final_N', '29'), ('replicate', '32')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                s20.summarize(self.altered(field, value))

    def test_window_endpoint_kept_out_of_successful_median(self):
        rows = [{'status': 'target_reached', 'model_time': t} for t in (1, 3)]
        rows += [{'status': 'observation_window', 'model_time': 20}]
        result = s20.summarize_group(rows, 'growth')
        self.assertEqual(result['target_fraction'], 2 / 3)
        self.assertEqual(result['window_count'], 1)
        self.assertEqual(result['median_success_passage_time'], 2)
        self.assertEqual(result['50_over_median_growth_time'], 25)
        self.assertNotEqual(result['50_over_median_growth_time'], (50 + 50 / 3) / 2)

    def test_all_window_endpoints_have_no_passage_time(self):
        rows = [{'status': 'observation_window', 'model_time': 20}] * 32
        result = s20.summarize_group(rows, 'growth')
        self.assertEqual(result['target_fraction'], 0)
        self.assertIsNone(result['median_success_passage_time'])
        self.assertIsNone(result['50_over_median_growth_time'])
        self.assertGreater(result['wilson95_high'], 0)

    def test_wilson_boundaries_and_invalid_counts(self):
        self.assertEqual(s20.wilson_interval(0, 32)[0], 0)
        self.assertAlmostEqual(s20.wilson_interval(0, 32)[1], 0.1071791982550706)
        self.assertEqual(s20.wilson_interval(32, 32)[1], 1)
        self.assertAlmostEqual(s20.wilson_interval(32, 32)[0], 0.8928208017449293)
        for successes, n in ((0, 0), (-1, 32), (33, 32)):
            with self.assertRaises(ValueError):
                s20.wilson_interval(successes, n)

    def test_changed_reference_rejected(self):
        matched = s20.matched_rows(s20.summarize(self.records))
        reference = s20.read_csv(s20.FOLDER / 'steric_matched_nine_conditions.csv')
        for field, value in [('target_nucleation', '32'), ('median_growth_passage_time', '3')]:
            changed = copy.deepcopy(reference)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                s20.compare_reference(matched, changed)

    def test_source_output_folder_rejected(self):
        for path in (s20.FOLDER, s20.ROOT, s20.ROOT / 'metadata'):
            with self.assertRaisesRegex(ValueError, 'reproduced_local'):
                s20.output_directory(path)

    def test_cli_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'results'
            run = subprocess.run([sys.executable, str(s20.ROOT / 'tools/reproduce_s20.py'),
                                  '--output-dir', str(output)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((output / 's20_verification.json').read_text())
            self.assertEqual(report['display_rows_matched'], 9)
            self.assertEqual(len(s20.read_csv(output / 's20_all_conditions.csv')), 864)
            self.assertEqual(len(s20.read_csv(output / 's20_recalculated.csv')), 9)
            self.assertEqual(s20.read_csv(output / 's20_recalculated.csv')[6]['median_growth_passage_time'], '')
            self.assertEqual(s20.read_csv(output / 's20_display.csv')[6]['median_growth_passage_time'], 'Not reached')


if __name__ == '__main__':
    unittest.main()
