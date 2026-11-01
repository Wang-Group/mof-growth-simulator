"""Reproduce S6 from reviewed local sources without running notebook batch cells."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import N_A, R, h, k, pi

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / '08_entropy_inputs'
MAPPING = {'SBU': ('SBU', 'Hf6(SBU).xyz'), 'BTB': ('BTB', 'BTBa_new.xyz'),
           'MOL1': ('MOL_1', 'Hf6(MOL_1).xyz'), "MOL1'": ("MOL_1'", "Hf6(MOL_1').xyz"),
           'MOL2': ('MOL_2', 'Hf6(MOL_2).xyz'), 'MOL3': ('MOL_3', 'Hf6(MOL_3).xyz')}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_sources():
    manifest = json.loads((FOLDER / 'calculation_manifest.json').read_text(encoding='utf-8'))
    require(len(manifest['files']) == 39, 'Expected 39 recovered entropy sources')
    for record in manifest['files']:
        path = (ROOT / record['release_path']).resolve()
        require(path.is_relative_to(FOLDER / 'source'), 'Entropy path escapes source folder')
        raw = path.read_bytes()
        require(len(raw) == record['stored_bytes'] and digest(raw) == record['stored_sha256'],
                f'Changed entropy source: {path.name}')
        if path.suffix == '.ipynb':
            nb = json.loads(raw)
            cells = json.dumps([(c['cell_type'], c['source']) for c in nb['cells']],
                               ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            require(digest(cells) == record['cell_sources_sha256'], 'Changed notebook code cells')
            require(not nb['metadata'] and all(not c['metadata'] for c in nb['cells']), 'Notebook metadata retained')
            require(all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells']),
                    'Notebook execution output retained')
        else:
            require(record['stored_sha256'] == record['source_sha256'], 'Non-notebook source was transformed')
    return manifest


def literal_assignment(code, name):
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'Missing notebook input: {name}')


def geometry_helpers(code):
    # The reviewed first cell ends by calling main(), which writes tables.
    # Load only helper definitions/constants, never main or the final invocation.
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
             or isinstance(n, ast.FunctionDef) and n.name != 'main']
    env = {'__name__': '_entropy_geometry_helpers'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'geometry_helpers', 'exec'), env)
    return env


def verify():
    manifest = verify_sources()
    nb = json.loads((FOLDER / 'source/calculate_entropy.ipynb').read_text(encoding='utf-8'))
    code = lambda i: ''.join(nb['cells'][i]['source'])
    function = next(n for n in ast.parse(code(1)).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'calculate_entropy')
    env = {'np': np, 'N_A': N_A, 'R_con': R, 'k': k, 'h': h, 'pi': pi}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'entropy_function', 'exec'), env)
    solvent = literal_assignment(code(5), 'solvent_property')
    solutes = literal_assignment(code(5), 'solute_property')
    geom_nb = json.loads((FOLDER / 'source/calculate_molecular_properties.ipynb').read_text(encoding='utf-8'))
    geom = geometry_helpers(''.join(geom_nb['cells'][0]['source']))
    documents = json.loads((ROOT / '13_table_source_audit/selected_si_tables_20261031.json').read_text(encoding='utf-8'))
    table = next(t for t in documents['documents'][0]['tables'] if t['caption'].startswith('Table S6.'))
    with (FOLDER / 'data/entropy_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = {r[0]: list(map(float, r[1:])) for r in list(csv.reader(stream))[1:]}
    rows, geometry = [], []
    for row in table['rows'][1:]:
        name, filename = MAPPING[row[0]]
        path = FOLDER / 'source/mol_xyz_files' / filename
        lines = path.read_text(encoding='utf-8').splitlines()
        atoms_and_xyz = [s.split() for s in lines[2:] if s.strip()]
        require(len(atoms_and_xyz) == int(lines[0]) and all(len(r) == 4 for r in atoms_and_xyz), 'Invalid XYZ')
        atoms = np.array([r[0] for r in atoms_and_xyz])
        xyz = np.array([[float(v) for v in r[1:]] for r in atoms_and_xyz])
        mass = geom['calculate_molecular_mass'](atoms)
        volume = geom['calculate_vdw_volume'](str(path))
        rg = np.sqrt(geom['calculate_rg_squared'](atoms, xyz))
        com = geom['calculate_com'](atoms, xyz)
        inertia = np.prod(np.linalg.eigvals(geom['calculate_inertia_tensor'](atoms, xyz, com))) ** (1 / 3) * 1e-3 / N_A * 1e-20
        expected = solutes[name]
        for actual, target, tolerance in zip((mass, volume, rg), expected[:3], (5.1e-4, 5.1e-8, 5.1e-10)):
            require(abs(actual - target) <= tolerance, f'Geometry input mismatch: {name}')
        require(float(f'{inertia:.3g}') == expected[4], f'Rounded inertia mismatch: {name}')
        calculated = env['calculate_entropy'](solvent, expected)
        # Report the actual calculation convention, not false full-precision agreement.
        geometry.append({'species': name, 'file': filename, 'atoms': len(atoms),
                         'mass_g_mol': mass, 'volume_A3': volume, 'rg_A': rg,
                         'inertia_kg_m2': inertia, 'sigma_from_notebook': expected[3],
                         'rounded_notebook_inputs': expected})
        for i, label in enumerate(('St/R', 'Sr/R', 'S/R')):
            difference = abs(float(calculated[i]) - float(row[i+1]))
            require(difference <= 5.0001e-9, f'S6 displayed value mismatch: {name} {label}')
            require(abs(float(calculated[i]) - original[name][i]) < 1e-9, f'Original CSV mismatch: {name}')
            rows.append({'species': name, 'quantity': label, 'calculated': float(calculated[i]),
                         'si_value': row[i+1], 'absolute_difference': difference})
    require(len(rows) == 18 and len(geometry) == 6, 'Unexpected S6 coverage')
    return {'status': 'PASS_S6_CALCULATION_AND_GEOMETRY_MAPPING',
            'source_files_verified': len(manifest['files']),
            'stored_source_bytes': sum(r['stored_bytes'] for r in manifest['files']),
            'species': 6, 'values': 18, 'maximum_absolute_difference_si': max(r['absolute_difference'] for r in rows),
            'solvent_M_gmol_V_A3_density_gcm3_T_K': solvent, 'geometry': geometry, 'values_compared': rows,
            'scope': 'Original model with original rounded inputs and symmetry numbers; six geometry mappings checked. No notebook batch cells, full-folder preprocessing, production KMC or new physical-model validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_local/entropy')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.is_relative_to(FOLDER):
        raise ValueError('Outputs must not overwrite entropy sources')
    report = verify()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 's6_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (destination / 's6_recalculated.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['values_compared'][0]))
        writer.writeheader()
        writer.writerows(report['values_compared'])
    print(json.dumps({k: report[k] for k in ('status', 'species', 'values', 'maximum_absolute_difference_si')}))


if __name__ == '__main__':
    main()
