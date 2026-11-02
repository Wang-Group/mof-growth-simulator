import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.verify_submission_sources import ROOT, identity_checks, package_path, read_csv, regression, verify


class SubmissionSourcesTests(unittest.TestCase):
    def test_complete_local_supplement(self):
        report = verify()
        self.assertEqual(report['files_verified'], 99)
        self.assertEqual(report['nmr_rows'], 10)
        self.assertEqual(report['nmr_source_files'], 90)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            package_path(ROOT, '../not-in-package')
        with self.assertRaises(ValueError):
            package_path(ROOT, 'README.md')

    def test_regression_known_line(self):
        result = regression([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(result['slope'], 2)
        self.assertAlmostEqual(result['intercept'], 1)
        self.assertAlmostEqual(result['r_squared'], 1)

    def test_identity_samples(self):
        report = identity_checks(ROOT/'17_submission_sources')
        self.assertEqual(report['mol_constants'], 25)
        self.assertEqual(report['recomputed_comparisons'], 7)

    def test_corrupt_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root/'17_submission_sources'
            group.mkdir()
            (group/'bad.csv').write_text('modified', encoding='utf-8')
            manifest = {'files':[{'release_path':'17_submission_sources/bad.csv',
                                 'bytes':8,'sha256':'0'*64}]}
            (group/'source_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Changed source'):
                verify(root)

    def test_duplicate_comparison_rejected(self):
        group = ROOT/'17_submission_sources'
        def duplicate_row(path):
            rows = read_csv(path)
            if path.name == 'recomputed_correlations.csv':
                rows[1] = rows[0].copy()
            return rows
        with patch('tools.verify_submission_sources.read_csv', side_effect=duplicate_row):
            with self.assertRaisesRegex(ValueError, 'Duplicate correlation row'):
                identity_checks(group)


if __name__ == '__main__':
    unittest.main()
