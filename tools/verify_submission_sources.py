"""Verify the November 2 supplement entirely from packaged local files."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT/'17_submission_sources'


def package_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to((root/'17_submission_sources').resolve()):
        raise ValueError(f'Unsafe supplement path: {relative}')
    return path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def regression(x, y):
    fit = linregress(x, y)
    rank = spearmanr(x, y)
    return dict(n=len(x), slope=float(fit.slope), intercept=float(fit.intercept),
                r_squared=float(fit.rvalue**2), p_value=float(fit.pvalue),
                spearman_rho=float(rank.statistic), spearman_p=float(rank.pvalue))


def identity_checks(group):
    folder = group/'identity'
    mol = read_csv(folder/'mol_fitted_constants_checked.csv')
    chen = read_csv(folder/'chen_exact_selected.csv')
    expected = read_csv(folder/'recomputed_correlations.csv')
    comparisons = {}
    for label, subset in [('acidic_all', [r for r in mol if r['Experiment']=='acidic']),
                          ('basic_all', [r for r in mol if r['Experiment']=='basic']),
                          ('basic_excluding_TFA_NFA', [r for r in mol if r['Experiment']=='basic' and r['Acid'] not in {'TFA','NFA'}])]:
        comparisons[(label, 'ln fitted K_effective')] = regression(
            [float(r['pKa']) for r in subset], [math.log(float(r['K_effective'])) for r in subset])
    included = {'BA','4-F','4-OMe','4-CN','4-Me','4-Cl','4-tBu'}
    for dose in [10, 50]:
        subset = [r for r in chen if r['identity'] in included and int(r['dose_equiv_per_Zr'])==dose]
        if len(subset) != 7 or {r['identity'] for r in subset} != included:
            raise ValueError(f'Changed Chen comparison sample: {dose}')
        x = [float(r['pKa_source_used']) for r in subset]
        y = np.array([float(r['LVol_IB_nm']) for r in subset])
        for response, values in [('LVol_IB_nm', y), ('ln_LVol_IB_nm', np.log(y))]:
            comparisons[(f'chen_para_matched_{dose}eq', response)] = regression(x, values)
    errors = []
    matched = set()
    for row in expected:
        key = (row['comparison'], row['response'])
        if key not in comparisons:
            if row['comparison'] != 'gibbons_50eq_four_acids':
                raise ValueError(f'Unexpected comparison: {key}')
            continue
        if key in matched:
            raise ValueError(f'Duplicate correlation row: {key}')
        matched.add(key)
        actual = comparisons[key]
        if actual['n'] != int(row['n']):
            raise ValueError(f'Sample mismatch: {key}')
        for field in ['slope','intercept','r_squared','p_value','spearman_rho','spearman_p']:
            reference = float(row[field])
            # The supplied fitted constants are rounded before this independent check.
            np.testing.assert_allclose(actual[field], reference, rtol=2e-6, atol=2e-8,
                                       err_msg=f'{key} {field}')
            errors.append(abs(actual[field]-reference))
    if len(expected) != 8 or len(comparisons) != 7 or matched != set(comparisons):
        raise ValueError('Changed correlation-table coverage')
    return dict(mol_constants=len(mol), chen_source_rows=len(chen), recomputed_comparisons=len(comparisons),
                maximum_absolute_difference=max(errors),
                archived_literature_only_comparison='gibbons_50eq_four_acids',
                scope='Regression of supplied source values; not refitting the original adsorption experiments')


def verify(root=ROOT):
    group = root/'17_submission_sources'
    manifest = json.loads((group/'source_manifest.json').read_text(encoding='utf-8'))
    seen = set()
    for item in manifest['files']:
        path = package_path(root, item['release_path'])
        if path in seen:
            raise ValueError(f'Duplicate manifest path: {path}')
        seen.add(path)
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'Changed source: {item["release_path"]}')
    if len(seen) != 99 or manifest['documents_included_or_edited']:
        raise ValueError('Changed supplement scope')
    original = json.loads((group/'nmr/Table_S4_acquisition_mapping.original.json').read_text(encoding='utf-8'))
    portable = json.loads((group/'nmr/portable_mapping.json').read_text(encoding='utf-8'))
    if [r['table_entry'] for r in portable] != list(range(1, 11)):
        raise ValueError('Expected all ten S4 rows')
    for old, row in zip(original['rows'], portable, strict=True):
        for key in row.keys()-{'files'}:
            if row[key] != old[key]:
                raise ValueError(f'Changed S4 field: {key}')
        if set(row['files']) != set(old['files']):
            raise ValueError('Changed NMR acquisition file set')
        for kind, relative in row['files'].items():
            path = package_path(root, relative)
            if path not in seen or hashlib.sha256(path.read_bytes()).hexdigest() != old['files'][kind]['sha256']:
                raise ValueError(f'NMR mapping mismatch: {relative}')
        rate = row['exchange_rate_s_minus_1_as_reported']
        if rate is not None and abs(math.pi*(row['linewidth_Hz_as_reported_in_Table_S4']-5.22)-rate) >= .005:
            raise ValueError('S4 rate/linewidth arithmetic differs beyond displayed rounding')
    row268 = next(r for r in portable if r['temperature_K_nominal_in_SI']==268)
    if row268['experiment_number'] != 2:
        raise ValueError('268 K must use author-confirmed acquisition 2')
    baseline = json.loads((root/'metadata/release_document_baseline.json').read_text(encoding='utf-8'))
    if baseline['documents'] != manifest['documents']:
        raise ValueError('Current document baseline differs from imported supplement')
    fig6 = next(d for d in baseline['documents'] if d['role']=='MS')['adopted_figure6']
    if (fig6['N'], fig6['repeats_per_condition']) != (2000, 4):
        raise ValueError('Wrong Fig. 6 cohort')
    return dict(status='PASS_SUBMISSION_SUPPLEMENT', files_verified=len(seen),
                bytes=sum(r['bytes'] for r in manifest['files']), nmr_rows=10,
                nmr_source_files=sum(len(r['files']) for r in portable),
                linewidths='Author/software values; no independent spectral refit',
                identity=identity_checks(group), documents=manifest['documents'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = verify()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
