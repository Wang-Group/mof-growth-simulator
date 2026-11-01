import unittest

from tools import audit_fig5_sources as audit


class Fig5ProvenanceTests(unittest.TestCase):
    def test_partial_path_cannot_become_exact_row_match(self):
        self.assertIsNone(audit.conditions('Zr_35_FA_225_L_1.29548_index0/assembly.mol2'))
        self.assertEqual(audit.matching_rows({'Zr': '35'}, [{'Zr': '35'}]), [])

    def test_repeat_number_is_part_of_match(self):
        values = dict(zip(audit.FIELDS, ['1', '2', '3', '0.6', '1e15.0', '0', '1', '1']))
        rows = [values, {**values, 'Index': '1'}]
        self.assertEqual(audit.matching_rows(values, rows), [1])
        self.assertEqual(audit.matching_rows(values, rows, ignore_index=True), [1, 2])

    def test_step_label_and_conditions_are_not_coerced(self):
        values = dict(zip(audit.FIELDS, ['1', '2', '3', '0.6', '1e15.0', '0', '1', '1']))
        self.assertEqual(audit.matching_rows(values, [{**values, 'Step': '1e15'}]), [])
        self.assertEqual(audit.matching_rows(values, [{**values, 'Ratio': '0'}]), [])

    def test_offline_evidence_is_explicitly_partial(self):
        report = audit.verify()
        self.assertEqual(report['fully_linked_classification_rows'], 0)
        self.assertFalse(report['fig5c_original_module_found'])
        self.assertEqual(report['row_status_counts']['STRUCTURE_CONDITIONS_ONLY'], 8)
        self.assertEqual(report['row_status_counts']['AMBIGUOUS_TRACE_INDEX'], 4)


if __name__ == '__main__':
    unittest.main()
