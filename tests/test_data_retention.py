from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import data_retention as retention


class DataRetentionTests(unittest.TestCase):
    def test_compact_policy(self):
        report = retention.verify()
        self.assertEqual(report['excluded_raw_files'], 11)
        self.assertEqual(report['classifier_examples'], 4)

    def test_changed_source_hash_not_silently_excluded(self):
        row = retention.policy()['removed_payloads'][0]
        with self.assertRaises(AssertionError):
            retention.excluded_payload(retention.ROOT / row['path'], row['bytes'], '0' * 64)

    def test_unknown_missing_file_not_excluded(self):
        self.assertFalse(retention.excluded_payload(retention.ROOT / 'unknown.csv', 1, '0' * 64))

    def test_reintroduced_raw_file_fails(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            path = root / 'raw.zip'
            path.write_bytes(b'x')
            data = {'removed_payloads': [{'path': 'raw.zip', 'bytes': 1, 'sha256': 'abc'}]}
            with patch.object(retention, 'ROOT', root), patch.object(retention, 'policy', return_value=data):
                with self.assertRaises(AssertionError):
                    retention.excluded_payload(path, 1, 'abc')

    def test_path_escape_fails(self):
        with self.assertRaises(ValueError):
            retention.excluded_payload(retention.ROOT.parent / 'outside.zip', 1, '0' * 64)

    def test_corrupt_classifier_example_fails(self):
        data = retention.policy()
        data['classifier_fixtures'][0]['sha256'] = '0' * 64
        with patch.object(retention, 'policy', return_value=data):
            with self.assertRaises(AssertionError):
                retention.classifier_examples()


if __name__ == '__main__':
    unittest.main()
