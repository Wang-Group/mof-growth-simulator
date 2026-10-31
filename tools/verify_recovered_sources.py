"""Offline checks for the 5090, plot-version and S21 source additions."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def mean_trace_check(path, expected_hash):
    require(digest(path) == expected_hash, f'Trace identity mismatch: {path}')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    runs, mean = data['replay_runs'], data['mean_trace']
    require(len(runs) == 4 and len(mean) == 400, f'Unexpected cohort: {path}')
    grid = np.asarray([r['sim_time_seconds'] for r in mean])
    require(np.isfinite(grid).all() and (np.diff(grid) >= 0).all(), 'Invalid mean grid')
    errors = {}
    for species in ('Zr12_AA', 'Zr6_AA', 'BDC'):
        sampled = []
        for run in runs:
            events = run['traces']
            times = np.asarray([r['sim_time_seconds'] for r in events])
            require(np.isfinite(times).all() and (np.diff(times) >= 0).all(), 'Invalid event times')
            require(times[0] <= grid[0], 'Mean grid precedes an initial state')
            # Previous-value sampling retains completed runs at their terminal state.
            indices = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
            sampled.append(np.asarray([r[species] for r in events])[indices])
        calculated = np.mean(sampled, axis=0)
        saved = np.asarray([r[species + '_mean'] for r in mean])
        np.testing.assert_allclose(calculated, saved, rtol=0, atol=1e-12)
        errors[species] = float(np.max(np.abs(calculated - saved)))
    zero = next((r['sim_time_seconds'] for r in mean if r['Zr12_AA_mean'] == 0), None)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': expected_hash,
            'repeats': len(runs), 'grid_points': len(mean), 'mean_max_absolute_error': errors,
            'first_mean_zr12_zero_time_s_as_saved': zero}


def verify():
    manifest = json.loads((ROOT / 'metadata/recovered_sources_20261031.json').read_text(encoding='utf-8'))
    rows = manifest['files']
    require(len({r['release_path'] for r in rows}) == len(rows), 'Duplicate recovered paths')
    counts = Counter(r['release_path'].split('/')[0] for r in rows)
    require(counts == {'09_recovered_runtime': 10, '10_plot_versions': 8, '11_s21_april03': 24},
            'Unexpected recovered-file selection')
    syntax_count = 0
    for row in rows:
        path = (ROOT / row['release_path']).resolve()
        require(path.is_relative_to(ROOT), f'Escaping manifest path: {path}')
        require(path.stat().st_size == row['bytes'] <= 25_000_000, f'Unexpected source size: {path}')
        require(digest(path) == row['sha256'], f'Changed source: {path}')
        require(path.suffix.lower() not in {'.docx', '.pkl', '.pickle', '.pem', '.key'}, 'Forbidden payload')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            syntax_count += 1

    version_pairs = [
        ('historical/figure6/build_figure7_kinetic_window_v2.py',
         'shared_20261031/MS_Figure_6/original_code/build_figure7_kinetic_window_v2.py',
         'b2a288c3096c7da83da0ef2008336ebf67913d667b7f8e813e971f86443e421e',
         '2f1f6ce0bfc1b0c4c94ce2a86e2b32b96ff3172504673d27e75aa5cb2ae5ec36'),
        ('historical/common/panel_runtime.py', 'shared_20261031/MS_Figure_6/panel_runtime.py',
         '15d48882e1c9c7b2aedaf2c25e12282c2caac3781c32516b34d41aafe479a90f',
         '113e81d846f2cb1059e18fa8ee6347599e80739c02c108fd6fb63c181af169c6'),
        ('historical/common/panel_runtime.py', 'shared_20261031/SI_Figure_S7/panel_runtime.py',
         '15d48882e1c9c7b2aedaf2c25e12282c2caac3781c32516b34d41aafe479a90f',
         'db715f6e27474340cd0e1b6ce3195184e8fc7dcf08d8dc9890d1133452e9ff64'),
    ]
    for old, new, old_hash, new_hash in version_pairs:
        require(digest(ROOT / '10_plot_versions' / old) == old_hash, f'Old version changed: {old}')
        require(digest(ROOT / '10_plot_versions' / new) == new_hash, f'New version changed: {new}')

    models = {}
    for path in sorted((ROOT / '09_recovered_runtime/models').rglob('*.mol')):
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        require('V2000' in lines[3], f'Unexpected MOL format: {path}')
        count = int(lines[3][:3])
        elements = [line[31:34].strip() for line in lines[4:4 + count]]
        require(len(elements) == count, f'Truncated atom block: {path}')
        models[path.name] = {'atoms': count, 'explicit_hydrogens': elements.count('H')}
    require(len(models) == 8 and models['BDC_explicit_H.mol'] == {'atoms': 16, 'explicit_hydrogens': 4},
            'Wrong explicit-H BDC/model set')

    archive = ROOT / '11_s21_april03/source_archive'
    cohorts = {
        'april03_zr6_zr12_mix': mean_trace_check(
            archive / 'source_data/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned.summary_trace.json',
            'd7be8f68b070266eb327d84e2d62d3cb00b7cb4cfa4008a6652b29e9e7391a89'),
        'april03_initialize_structures1': mean_trace_check(
            archive / 'other_versions/initialize_structures1/mixed_old_trace_schema.json',
            'f08ad2680b27bf1f337aa424de66abb3279d14bbf1e650bc5cdde03cb69fca03'),
        'april10_rerun': mean_trace_check(
            ROOT / '05_mixed_zr6_zr12/data/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned_rerun_20260410.summary_trace.json',
            '827fa5f219fad17a6602fe50193996b80c020756b810afb5620f2397fadc3be6'),
    }
    require(len({v['sha256'] for v in cohorts.values()}) == 3, 'Mixed-seed variants were conflated')
    return {'status': 'PASS', 'copied_files': len(rows), 'copy_bytes': sum(r['bytes'] for r in rows),
            'groups': dict(counts), 'python_syntax_checks': syntax_count, 'version_pairs_verified': 3,
            'models': models, 'mixed_seed_variants': cohorts,
            'limits': ['No KMC execution', 'No independent MS/SI artwork matching',
                       'Stored time conventions are not newly calibrated physical clocks']}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
