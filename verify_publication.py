"""Check the publication candidate locally without running KMC or archived plotters."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from tools.verify_recovered_sources import verify as verify_recovered_sources
from tools.audit_runtime_and_tables import verify as verify_runtime_and_tables
from tools.reproduce_entropy import verify as verify_entropy

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / '07_publication_sources'
TABLES = SOURCES / 'SI_tables_sources_20261023'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def manifest_checks():
    with (ROOT/'metadata/publication_sources.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    included = [r for r in records if r['decision'] == 'COPY']
    variants = [r for r in records if r['decision'] == 'VERSION_VARIANT_ARCHIVED']
    require(len(variants) == 3, 'Expected three resolved plot-version records')
    historical_hashes = {digest(p) for p in (ROOT/'10_plot_versions/historical').rglob('*.py')}
    for row in variants:
        require(row['expected_sha256'] in historical_hashes, 'Historical plot version not retained')
    for row in included + variants:
        path = (ROOT/row['release_path']).resolve()
        require(path.is_relative_to(ROOT), f'Escaping path: {path}')
        require(path.is_file(), f'Missing packaged input: {path}')
        require(path.stat().st_size == int(row['bytes']), f'Changed size: {path}')
        require(digest(path) == row['sha256'], f'Changed contents: {path}')
        require(path.stat().st_size <= 25_000_000, f'Oversize source: {path}')
    forbidden = [str(p.relative_to(ROOT)) for p in SOURCES.rglob('*') if p.is_file() and
                 (p.suffix.lower() in {'.docx', '.pkl', '.pickle', '.pem', '.key'} or p.name.startswith('id_rsa'))]
    require(not forbidden, f'Forbidden publication payloads: {forbidden}')
    return {'verified_copies': len(included), 'version_variants_verified': len(variants),
            'decisions': dict(Counter(r['decision'] for r in records)),
            'copy_bytes': sum(int(r['bytes']) for r in included)}


def variance_checks():
    folder = SOURCES/'MS_Figure_5/content_checked_20261023/source_data'
    frame = pd.read_csv(folder/'common_response_surface_v2_final_condition_summary.csv')
    frame = frame.loc[frame.system == 'btb_mol']
    axes = ['SC', 'q', 'off_weight']
    levels = [sorted(frame[c].unique()) for c in axes]
    require(not frame.duplicated(axes).any(), 'Duplicate response conditions')
    index = pd.MultiIndex.from_product(levels, names=axes)
    p = frame.set_index(axes).reindex(index).persistent_assembly_probability
    require(not p.isna().any(), 'Incomplete response grid')
    require(len(frame) == 1260 and frame.trace_count.eq(32).all(), 'Unexpected BTB cohort')
    np.testing.assert_allclose(frame.persistent_assembly_probability, frame.target_count/frame.trace_count,
                               atol=1e-14, rtol=0)
    cube = p.to_numpy().reshape(tuple(map(len, levels)))
    expected = pd.read_csv(folder/'figure6bc_current_parameter_phase_v1_sensitivity.csv').set_index('parameter')
    results = []
    for i, name in enumerate(['s_close', 'atomic_metal_linker_ratio', 'rho_off']):
        first = float(np.var(cube.mean(axis=tuple(j for j in range(3) if j != i)))/np.var(cube))
        total = float(np.var(cube, axis=i).mean()/np.var(cube))
        np.testing.assert_allclose([first, total], expected.loc[name, ['first_order', 'total_effect']].astype(float),
                                   atol=1e-12, rtol=0)
        results.append({'parameter': name, 'first_order': first, 'total_effect': total})
    return {'conditions': len(frame), 'repeats_per_condition': 32, 'results': results,
            'bootstrap_intervals_recomputed': False}


def repair_checks():
    folder = SOURCES/'SI_Figure_S25/content_checked_20261023/source_data'
    result = {}
    for panel, rows, endpoint, expected_hash in [
        ('A', 130, 3651, '1837a4e5a8b5690b7f1ff30d0fa1f789a4f5b15d18e640d6e8a996adb70d6b4d'),
        ('B', 103, 100001, '1c8a667365327b334887f1f1c29b7c112ceb59c36701ea032d6a90e5b52b3bfc')]:
        path = folder/panel/'repair_progress_vs_growth.csv'
        require(digest(path) == expected_hash, f'S25{panel} provenance changed')
        data = pd.read_csv(path)
        require(len(data) == rows and data.total_entities.iloc[0] == 2124 and data.total_entities.iloc[-1] == endpoint,
                f'Unexpected S25{panel} trace')
        result[panel] = {'rows': rows, 'endpoint_entities': endpoint,
                         'final_BDC_fill': float(data.bdc_fill.iloc[-1]), 'final_Zr_fill': float(data.zr_fill.iloc[-1])}
    return result


def selector_smoke_checks():
    path = TABLES/'Table_S28/code/common_kmc_selector.py'
    require(digest(path) == '91de30bd57bd24472cff4c326d1436cc40da3ea29ed026d9ec47f56a50a524ed',
            'Selector differs from inspected source')
    spec = importlib.util.spec_from_file_location('_publication_selector', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    counts = module.OpportunityCounts(2, 4, 1, 3)
    coordinates = module.PhysicalCoordinates(2.0, 0.25, 3.0, 0.0)
    p = module.compute_propensities(counts, coordinates)
    np.testing.assert_allclose([p.add_node, p.add_linker, p.close_internal, p.remove_bond], [1, 6, 1, 9])
    require(math.isclose(p.normalized().total, 1.0), 'Propensity normalization')
    require(math.isclose(module.ideal_node_fraction(1.0), 1/7), 'Atomic-to-node fraction mapping')
    expected_wait = -math.log(random.Random(17).random())/17
    require(math.isclose(module.gillespie_waiting_time(17, random.Random(17)), expected_wait), 'Clock smoke test')
    try:
        module.gillespie_waiting_time(0, random.Random(17))
    except ValueError:
        pass
    else:
        raise AssertionError('Zero propensity should be rejected')
    return {'status': 'PASS', 'scope': 'Pure selector weights and clock smoke checks, not full geometry or KMC validation'}


def dependency_audit():
    by_hash = {}
    for folder in (SOURCES, ROOT/'09_recovered_runtime'):
        for path in folder.rglob('*.py'):
            by_hash.setdefault(digest(path), []).append(path.relative_to(ROOT).as_posix())
    results = []
    records = sorted((TABLES/'Table_S7/matched_source_files').glob('uio66*.json'))
    require(len(records) == 6, 'Expected six executed UiO-66 result records')
    for path in records:
        data = json.loads(path.read_text(encoding='utf-8'))
        for kind in ['engine', 'runner', 'selector']:
            expected = data.get('provenance', {}).get(kind+'_sha256')
            require(expected and len(expected) == 64, f'Missing {kind} provenance: {path.name}')
            if expected:
                results.append({'result': path.name, 'component': kind, 'recorded_sha256': expected,
                                'matching_packaged_paths': by_hash.get(expected, []),
                                'status': 'EXACT_HASH_PRESENT' if expected in by_hash else 'EXECUTED_VERSION_NOT_PACKAGED'})
    require(all(r['status'] == 'EXACT_HASH_PRESENT' for r in results),
            'A recorded UiO-66 engine/runner/selector version is missing')
    return results


def additional_source_checks():
    entropy = json.loads((ROOT/'08_entropy_inputs/source_manifest.json').read_text(encoding='utf-8'))
    require(len(entropy['files']) == 13, 'Expected thirteen entropy tables')
    for row in entropy['files']:
        path = (ROOT/row['path']).resolve()
        require(path.is_relative_to(ROOT/'08_entropy_inputs/data'), f'Escaping entropy path: {path}')
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                f'Changed entropy table: {path}')
    btb = {
        'mol_growth_main_20250811.py': 'a07480434d679249a4a353c108fc6a503678261dddb906fa41c3efa3b42957a2',
        'MOL_Assembly_Large_Correction_20250811.py': '27799313b3a11f16a6f99da210b5e1d58d20ee9ef5d641df3518a90352fe5490',
    }
    for name, expected in btb.items():
        require(digest(ROOT/'03_btb_trajectory/source'/name) == expected, f'Changed recovered BTB source: {name}')
    return {'entropy_tables_verified': 13, 'btb_run_source_files_verified': 2,
            'scope': 'Byte verification only; not Gaussian recalculation or historical KMC rerun'}


def main():
    from tools.audit_fig5_sources import verify as verify_fig5_sources
    from tools.audit_fig5d_lineage import verify as verify_fig5d_lineage
    from tools.audit_fig5c import verify as verify_fig5c
    from tools.verify_submission_sources import verify as verify_submission_sources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    checks = {'sources': manifest_checks(), 'figure5b': variance_checks(), 's25': repair_checks(),
              'selector_smoke': selector_smoke_checks(), 'additional_sources': additional_source_checks(),
              'recovered_sources': verify_recovered_sources(),
              'entropy_calculation': verify_entropy(),
              'figure5_source_followup_historical_20261101': verify_fig5_sources(),
              'figure5d_trajectory_lineage': verify_fig5d_lineage(),
              'figure5c_independent_reconstruction': verify_fig5c(),
              'runtime_and_table_provenance': verify_runtime_and_tables(),
              'submission_source_supplement': verify_submission_sources()}
    dependencies = dependency_audit()
    decisions = json.loads((ROOT/'metadata/publication_decisions_20261031.json').read_text(encoding='utf-8'))
    require(decisions['S7']['time_coordinate'] == 'raw algorithm time u', 'Wrong S7 publication coordinate')
    require((ROOT/decisions['S7']['entry_point']).is_file(), 'S7 publication entry missing')
    release = json.loads((ROOT/'metadata/release_preparation.json').read_text(encoding='utf-8'))
    require(release['license_code'] == 'MIT' and release['license_project_data'] == 'CC-BY-4.0',
            'Release licenses differ from author confirmation')
    require(digest(ROOT/'LICENSE-DATA.txt') == release['cc_by_legalcode_sha256'],
            'Changed CC BY 4.0 legal text')
    report = {'status': 'PASS_FOR_DECLARED_LOCAL_CHECKS', 'publication_ready': False,
              'checks': checks, 'executed_code_coverage': dependencies,
              'user_confirmed_release_decisions': decisions,
              'release_preparation': release,
              'publication_ready_reason': 'Formal tag and selected external dataset deposit are not yet published',
              'limits': ['No production KMC rerun', 'Not every archived plotter is portable',
                         'Verification covers the numerical and provenance checks listed above']}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
