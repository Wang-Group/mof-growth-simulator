"""Tests for the new offline verifier, without deserializing any research input."""
import gzip
import unittest
from unittest.mock import Mock, patch

from tools import audit_runtime_and_tables as audit


class SourceIntegrityTests(unittest.TestCase):
    def store(self, raw=b'example', encoding='gzip'):
        packed = gzip.compress(raw, mtime=0) if encoding == 'gzip' else raw
        obj = audit.Sources.__new__(audit.Sources)
        obj.cache = {}
        obj.index = {'source.csv': {'release_path': 'example.csv.gz', 'encoding': encoding,
                                    'stored_bytes': len(packed), 'stored_sha256': audit.digest(packed),
                                    'source_bytes': len(raw), 'source_sha256': audit.digest(raw)}}
        return obj, packed

    def test_gzip_recovers_exact_bytes(self):
        obj, packed = self.store(b'a,b\r\n1,2\r\n')
        with patch.object(audit, 'local_path', return_value=Mock(read_bytes=lambda: packed)):
            self.assertEqual(obj.read('source.csv'), b'a,b\r\n1,2\r\n')

    def test_corrupt_stored_file_rejected(self):
        obj, packed = self.store()
        with patch.object(audit, 'local_path', return_value=Mock(read_bytes=lambda: packed + b'x')):
            with self.assertRaisesRegex(AssertionError, 'Stored hash'):
                obj.read('source.csv')

    def test_wrong_uncompressed_hash_rejected(self):
        obj, packed = self.store()
        obj.index['source.csv']['source_sha256'] = '0' * 64
        with patch.object(audit, 'local_path', return_value=Mock(read_bytes=lambda: packed)):
            with self.assertRaisesRegex(AssertionError, 'Source hash'):
                obj.read('source.csv')

    def test_unknown_encoding_rejected(self):
        obj, packed = self.store(encoding='unknown')
        with patch.object(audit, 'local_path', return_value=Mock(read_bytes=lambda: packed)):
            with self.assertRaisesRegex(AssertionError, 'Encoding'):
                obj.read('source.csv')

    def test_path_cannot_leave_package(self):
        with self.assertRaisesRegex(AssertionError, 'outside package'):
            audit.local_path('../not-in-this-package')

    def test_table_scope_and_full_audit(self):
        report = audit.verify()
        self.assertFalse(report['publication_ready'])
        self.assertEqual(report['runtime']['lineage_rows'], 1327)
        self.assertEqual(report['runtime']['paired_independent_source_particles'], 32)
        self.assertEqual(report['runtime']['unmatched_version_references'], [])
        self.assertEqual(report['tables']['s1']['distinct_publication_groups'], 68)
        self.assertEqual(report['tables']['s6']['matched_values'], 18)
        self.assertIn('NOT_LOCATED', report['tables']['s8']['status'])

    def test_packaged_pickle_allowlist(self):
        from verify_package import serialized_input_policy
        self.assertEqual(serialized_input_policy()['packaged_seed_host_files'], 28)

    def test_unlisted_pickle_is_rejected(self):
        from pathlib import Path
        from verify_package import serialized_input_policy
        def candidates(path, pattern):
            return iter([] if pattern == '*.docx' else [audit.ROOT / 'unlisted.pkl'])
        with patch.object(Path, 'rglob', candidates):
            with self.assertRaisesRegex(AssertionError, 'Unlisted pickle'):
                serialized_input_policy()


if __name__ == '__main__':
    unittest.main()
