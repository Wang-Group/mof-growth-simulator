"""Check archived framework sources without executing growth trajectories."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / '18_hkust1_pcn240'
MODELS = ROOT / 'KMC_example/input_monomers'
SYSTEMS = (('HKUST-1', 'HKUST_1', 'Cu2.mol', 'BTC.mol', 4, 3, 200),
           ('PCN-240', 'PCN_240', 'Fe_Co3.mol', 'BDC_OH.mol', 6, 2, 500))


def model_calls(tree):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'read_mol_file_v5']


class FrameworkSourceTests(unittest.TestCase):
    def test_source_hashes_and_syntax(self):
        manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['files']), 9)
        for row in manifest['files']:
            path = (ROOT / row['path']).resolve()
            self.assertTrue(path.is_relative_to(ROOT.resolve()))
            raw = path.read_bytes()
            self.assertEqual(len(raw), row['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
            if path.suffix == '.py':
                compile(raw, str(path), 'exec')

    def test_all_preloaded_models_included(self):
        for folder, prefix, node, linker, *_ in SYSTEMS:
            tree = ast.parse((ROOT / folder / f'{prefix}_Assembly_Large_Correction_conc.py').read_bytes())
            models = [Path(ast.literal_eval(call.args[0])).name for call in model_calls(tree)]
            self.assertCountEqual(models, [node, linker, 'Zr12_AA.mol'])
            for name in models:
                self.assertTrue((MODELS / name).is_file())

    def test_engine_models_and_initial_assembly(self):
        for folder, prefix, _, _, node_sites, linker_sites, _ in SYSTEMS:
            path = ROOT / folder / f'{prefix}_Assembly_Large_Correction_conc.py'
            tree = ast.parse(path.read_bytes())
            # Redirect only model paths in memory; archived code stays byte-identical.
            for call in model_calls(tree):
                name = Path(ast.literal_eval(call.args[0])).name
                call.args[0] = ast.copy_location(ast.Constant(str(MODELS / name)), call.args[0])
            namespace = {'__name__': f'_test_{prefix}', '__file__': str(path)}
            exec(compile(tree, str(path), 'exec'), namespace)
            for data in namespace['PRELOADED_DATA'].values():
                self.assertGreater(len(data[0]), 0)
                self.assertEqual(np.asarray(data[1]).shape, (len(data[0]), 3))
                self.assertTrue(np.isfinite(data[1]).all())
            node = namespace['Zr6_AA']()
            linker = namespace['BDC']()
            self.assertEqual(len(node.carboxylates), node_sites)
            self.assertEqual(len(linker.carboxylates), linker_sites)
            assembly = namespace['Assembly'](node, 1, 30.9, 2)
            self.assertEqual(len(assembly.entities), 1)

    def test_runner_settings_and_local_engine_import(self):
        for folder, prefix, _, _, _, _, maximum in SYSTEMS:
            path = ROOT / folder / f'{prefix}_growth_main_conc.py'
            tree = ast.parse(path.read_bytes())
            settings = {}
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            settings[target.id] = statement.value.value
            self.assertEqual(settings['max_entities'], maximum)
            self.assertEqual(settings['Linker_conc'], 34.57984361265395)
            self.assertEqual(settings['Capping_agent_conc'], 200 if prefix == 'PCN_240' else 1473.06341756944)
            self.assertIsNone(settings['pkl_path'])
            self.assertTrue(any(isinstance(n, ast.ImportFrom) and
                                n.module == f'{prefix}_Assembly_Large_Correction_conc'
                                for n in tree.body))


if __name__ == '__main__':
    unittest.main()
