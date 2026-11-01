"""Tests for the local S6 calculation without invoking archived batch writers."""
import json
import unittest
from unittest.mock import patch

from tools import reproduce_entropy as entropy
from tools.package_entropy_sources import clean_notebook, cell_digest


class EntropyTests(unittest.TestCase):
    def test_full_s6_calculation(self):
        report = entropy.verify()
        self.assertEqual(report['values'], 18)
        self.assertEqual(report['species'], 6)
        self.assertEqual(report['source_files_verified'], 39)
        self.assertLessEqual(report['maximum_absolute_difference_si'], 5e-9)

    def test_wrong_btb_variant_rejected(self):
        with patch.dict(entropy.MAPPING, {'BTB': ('BTB', 'BTBa.xyz')}):
            with self.assertRaisesRegex(AssertionError, 'Geometry input mismatch: BTB'):
                entropy.verify()

    def test_notebook_outputs_removed_code_preserved(self):
        nb = {'cells': [{'cell_type': 'code', 'source': ['print(1)'],
                         'metadata': {'local': 'discard'}, 'execution_count': 5,
                         'outputs': [{'output_type': 'stream', 'text': 'discard'}]}],
              'metadata': {'local': 'discard'}, 'nbformat': 4, 'nbformat_minor': 5}
        raw, fingerprint = clean_notebook(json.dumps(nb).encode())
        cleaned = json.loads(raw)
        self.assertEqual(fingerprint, cell_digest(nb))
        self.assertEqual(cleaned['cells'][0]['source'], nb['cells'][0]['source'])
        self.assertEqual(cleaned['cells'][0]['outputs'], [])
        self.assertEqual(cleaned['metadata'], {})

    def test_main_and_top_level_calls_not_run(self):
        env = entropy.geometry_helpers('def helper():\n    return 7\ndef main():\n    raise RuntimeError()\nmain()\n')
        self.assertNotIn('main', env)
        self.assertEqual(env['helper'](), 7)


if __name__ == '__main__':
    unittest.main()
