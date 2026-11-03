"""Recompute Table S20 from packaged per-trajectory endpoints, without running KMC."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / '07_publication_sources/SI_tables_sources_20261023/Table_S20/matched_source_files'
REFERENCE = ROOT / 'metadata/s20_reference.json'
IDENTITIES = ('benzoate', '4-F', '4-OMe', '4-CN', '4-Me', '4-Cl', '4-tBu',
              '4-CF3', '2-F', '2-Me', '2-OMe', '2-NO2')
ASSAYS = {'nucleation': (1, 30, 100.0), 'growth': (76, 126, 20.0)}
RHOS = (0.1, 1.0, 3.0, 10.0)
RATIOS = (0.3, 1.0, 3.0)
ETAS = (0.1, 0.5, 0.9)
MATCHED_IDENTITIES = ('benzoate', '4-OMe', '4-tBu')
KEYS = ('identity', 'assay', 'rho_off', 'q_atomic_Zr_BDC', 'eta_S')
REPLICATES = set(range(32))
Z95 = 1.959963984540054
DISPLAY_FIELDS = ('cap', 'etaS', 'nucleation_fraction', 'growth_fraction',
                  'median_growth_passage_time')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def wilson_interval(successes, n):
    require(isinstance(successes, int) and isinstance(n, int) and n > 0
            and 0 <= successes <= n, 'Invalid binomial counts')
    p = successes / n
    z2 = Z95 ** 2
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n ** 2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_group(records, assay):
    times = [r['model_time'] for r in records if r['status'] == 'target_reached']
    successes, n = len(times), len(records)
    low, high = wilson_interval(successes, n)
    passage_time = median(times) if times else None
    initial, target, window = ASSAYS[assay]
    return {'initial_entities': initial, 'target_entities': target,
            'observation_window_model_time': window, 'n': n,
            'target_count': successes, 'window_count': n - successes,
            'target_fraction': successes / n, 'wilson95_low': low, 'wilson95_high': high,
            'median_success_passage_time': passage_time,
            '50_over_median_growth_time': 50 / passage_time
            if assay == 'growth' and passage_time is not None else None}


def summarize(records):
    """Validate the complete production grid before calculating any summaries."""
    required = set(KEYS) | {'run_id', 'replicate', 'status', 'model_time',
                           'valid_physical_outcome', 'failure_reason', 'final_N', 'maximum_N'}
    groups, run_ids, replicate_keys = defaultdict(list), set(), set()
    for index, raw in enumerate(records, 2):
        require(required <= raw.keys(), f'Missing trajectory columns at CSV row {index}')
        row = dict(raw)
        for field in ('rho_off', 'q_atomic_Zr_BDC', 'eta_S', 'model_time'):
            row[field] = float(row[field])
            require(math.isfinite(row[field]), f'Non-finite {field} at CSV row {index}')
        for field in ('replicate', 'final_N', 'maximum_N'):
            row[field] = int(row[field])
        key = tuple(row[field] for field in KEYS)
        run_id = row['run_id']
        require(run_id and run_id not in run_ids, f'Duplicate or empty run_id: {run_id}')
        run_ids.add(run_id)
        require((key, row['replicate']) not in replicate_keys, f'Duplicate replicate: {key}')
        replicate_keys.add((key, row['replicate']))
        require(row['identity'] in IDENTITIES and row['assay'] in ASSAYS
                and row['rho_off'] in RHOS and row['q_atomic_Zr_BDC'] in RATIOS
                and row['eta_S'] in ETAS, f'Unexpected condition: {key}')
        require(row['replicate'] in REPLICATES, f'Invalid replicate: {run_id}')
        require(row['status'] in {'target_reached', 'observation_window'},
                f'Unexpected terminal status: {run_id}')
        require(row['valid_physical_outcome'] == 'True' and not row['failure_reason'],
                f'Invalid physical outcome: {run_id}')
        _, target, window = ASSAYS[row['assay']]
        require(0 < row['model_time'] <= window, f'Invalid model_time: {run_id}')
        require(0 < row['final_N'] <= row['maximum_N'], f'Invalid entity counts: {run_id}')
        if row['status'] == 'target_reached':
            require(row['final_N'] == row['maximum_N'] == target,
                    f'Target status disagrees with entity counts: {run_id}')
        else:
            require(row['maximum_N'] < target and row['model_time'] == window,
                    f'Window status disagrees with endpoint: {run_id}')
        groups[key].append(row)
    expected = set(product(IDENTITIES, ASSAYS, RHOS, RATIOS, ETAS))
    require(set(groups) == expected, 'Incomplete assay-condition grid (expected 864 groups)')
    result = []
    for key in sorted(groups):
        group = groups[key]
        require({r['replicate'] for r in group} == REPLICATES and len(group) == 32,
                f'Incomplete replicate set: {key}')
        result.append(dict(zip(KEYS, key), **summarize_group(group, key[1])))
    return result


def matched_rows(summaries):
    indexed = {tuple(row[field] for field in KEYS): row for row in summaries}
    result = []
    for cap, eta in product(MATCHED_IDENTITIES, ETAS):
        n = indexed[(cap, 'nucleation', 1.0, 1.0, eta)]
        g = indexed[(cap, 'growth', 1.0, 1.0, eta)]
        result.append({'cap': cap, 'etaS': eta,
                       'n_nucleation': n['n'], 'target_nucleation': n['target_count'],
                       'n_growth': g['n'], 'target_growth': g['target_count'],
                       'median_growth_passage_time': g['median_success_passage_time'],
                       '50_over_median_growth_time': g['50_over_median_growth_time'],
                       'nucleation_wilson95_low': n['wilson95_low'],
                       'nucleation_wilson95_high': n['wilson95_high'],
                       'growth_wilson95_low': g['wilson95_low'],
                       'growth_wilson95_high': g['wilson95_high']})
    return result


def display_rows(matched):
    return [[r['cap'], f"{r['etaS']:.1f}",
             f"{r['target_nucleation']}/{r['n_nucleation']}",
             f"{r['target_growth']}/{r['n_growth']}",
             'Not reached' if r['median_growth_passage_time'] is None
             else f"{r['median_growth_passage_time']:.3f}"] for r in matched]


def compare_reference(matched, reference):
    expected = {(r['cap'], float(r['etaS'])): r for r in reference}
    require(len(expected) == len(reference) == len(matched) == 9, 'Invalid nine-row reference')
    require(set(expected) == {(r['cap'], r['etaS']) for r in matched}, 'Reference keys differ')
    differences, checked, undefined = defaultdict(list), 0, 0
    count_fields = {'n_nucleation', 'target_nucleation', 'n_growth', 'target_growth'}
    for row in matched:
        target = expected[(row['cap'], row['etaS'])]
        require(set(row) == set(target), 'Reference columns differ')
        for field, actual in row.items():
            if field in {'cap', 'etaS'}:
                continue
            wanted = target[field]
            checked += 1
            if actual is None:
                require(wanted == '', f'Undefined value mismatch: {row["cap"]} {field}')
                undefined += 1
            elif field in count_fields:
                require(actual == int(wanted), f'Count mismatch: {row["cap"]} {field}')
            else:
                difference = abs(actual - float(wanted))
                require(math.isclose(actual, float(wanted), rel_tol=0, abs_tol=1e-12),
                        f'Numeric mismatch: {row["cap"]} {field}')
                differences[field].append(difference)
    return {'rows': len(matched), 'checked_values': checked, 'undefined_values': undefined,
            'absolute_tolerance': 1e-12,
            'maximum_absolute_difference_by_field': {k: max(v) for k, v in differences.items()}}


def reproduce():
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    source = FOLDER / 'source_trace_summary.csv'
    saved = FOLDER / 'steric_matched_nine_conditions.csv'
    for path in (source, saved):
        require(digest(path) == reference['input_sha256'][path.name],
                f'Changed packaged S20 input: {path.name}')
    records = read_csv(source)
    summaries = summarize(records)
    matched = matched_rows(summaries)
    comparison = compare_reference(matched, read_csv(saved))
    printed = display_rows(matched)
    require(printed == reference['display_rows'], 'Displayed SI Table S20 values differ')
    report = {'status': 'PASS_S20_FROM_TRAJECTORY_RECORDS', 'trajectory_records': len(records),
              'condition_groups': len(summaries), 'replicates_per_group': 32,
              'outcome_counts': dict(sorted(Counter(r['status'] for r in records).items())),
              'source': source.relative_to(ROOT).as_posix(),
              'input_sha256': reference['input_sha256'],
              'high_precision_reference': comparison,
              'si_reference': reference['document'], 'display_rows_matched': len(printed),
              'definitions': {
                  'time': 'Unscaled model_time from the trajectory endpoint records',
                  'probability': 'target_reached / all 32 trajectories; window endpoints retained',
                  'interval': f'Pointwise Wilson 95% interval; z={Z95}; no continuity correction',
                  'passage_time': 'Median model_time among target_reached trajectories only',
                  'conditional_growth_measure': '50 / median(successful growth time), not median(50/time)',
                  'undefined': 'No successes: time and conditional growth measure are null, not zero'},
              'scope': 'Reanalysis of archived per-trajectory terminal records. All 864 groups computed; '
                       'nine matched rows checked against saved high-precision results and SI display. '
                       'No KMC rerun, event-history reconstruction, time rescaling or MS/SI edit.'}
    return report, summaries, matched, printed


def verify():
    return reproduce()[0]


def output_directory(path):
    destination = path.resolve()
    require(not destination.is_relative_to(ROOT)
            or destination.is_relative_to(ROOT / 'reproduced_local'),
            'Outputs inside the repository must stay in reproduced_local')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_local/S20')
    args = parser.parse_args()
    destination = output_directory(args.output_dir)
    report, summaries, matched, printed = reproduce()
    tables = {'s20_all_conditions.csv': summaries, 's20_recalculated.csv': matched,
              's20_display.csv': [dict(zip(DISPLAY_FIELDS, row)) for row in printed]}
    for name in [*tables, 's20_verification.json']:
        output_directory((destination / name).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (destination / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (destination / 's20_verification.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'trajectory_records', 'condition_groups',
                                          'display_rows_matched', 'outcome_counts')}))


if __name__ == '__main__':
    main()
