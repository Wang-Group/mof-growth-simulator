"""Recompute BTB variance indices and bootstrap intervals from condition counts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / '16_fig5c_reconstruction'
SOURCE = GROUP / 'source/analyze_figure6_current_parameter_phase_v1.py'
INPUT = GROUP / 'data/common_response_surface_v2_final_condition_summary.csv'
REFERENCE = (ROOT / '07_publication_sources/MS_Figure_5/content_checked_20261023'
             / 'source_data/figure6bc_current_parameter_phase_v1_sensitivity.csv')
SOURCE_HASH = '2346424154dbc0e6c4eaf3bfc877ea564c9593bccd6450e9c9c0ad054d9ee0c7'
INPUT_HASH = '866fee5ec5a3b0e4f80bc776b4f1880bdfb0f0694cc0bc0e02ac0ddb95e31763'
REFERENCE_HASH = '6ae0aa62a0314aba80fcca088a61f9de45036452cd384ce3381977d2abefe6af'
PARAMETERS = ['s_close', 'atomic_metal_linker_ratio', 'rho_off']
COLUMNS = ['first_order', 'first_order_ci_low', 'first_order_ci_high',
           'total_effect', 'total_effect_ci_low', 'total_effect_ci_high']


def checked_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Changed source: {path.name}')
    return actual


def source_module():
    checked_hash(SOURCE, SOURCE_HASH)
    spec = importlib.util.spec_from_file_location('_btb_sensitivity_source', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.BOOTSTRAP_DRAWS != 2000 or module.RNG_SEED != 20260901:
        raise ValueError('Unexpected bootstrap settings')
    return module


def calculate(frame):
    frame = frame.loc[frame.system == 'btb_mol'].copy()
    keys = ['SC', 'q', 'off_weight']
    numeric = keys + ['trace_count', 'target_count', 'persistent_assembly_probability']
    if len(frame) != 1260 or frame.duplicated(keys).any():
        raise ValueError('Expected 1260 distinct BTB conditions')
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError('Nonfinite condition data')
    n, k = frame.trace_count.to_numpy(), frame.target_count.to_numpy()
    if not ((n == 32).all() and (k >= 0).all() and (k <= n).all()
            and (k == np.floor(k)).all()):
        raise ValueError('Invalid trial or success counts')
    if not np.allclose(frame.persistent_assembly_probability, k / n, atol=1e-14, rtol=0):
        raise ValueError('Probability differs from success fraction')
    if tuple(frame[key].nunique() for key in keys) != (9, 14, 10):
        raise ValueError('Expected a complete 9 x 14 x 10 grid')
    module = source_module()
    _, _, _, cube, counts, _ = module.build_probability_cube(frame)
    first, total = module.variance_indices(cube)
    first_low, first_high, total_low, total_high = module.bootstrap_indices(cube, counts)
    values = np.stack([first, first_low, first_high, total, total_low, total_high], axis=1)
    result = pd.DataFrame(values, columns=COLUMNS)
    result.insert(0, 'parameter', PARAMETERS)
    result['scope'] = 'uniform weighting over the sampled 9 x 14 x 10 BTB grid'
    return result


def reproduce():
    checked_hash(INPUT, INPUT_HASH)
    checked_hash(REFERENCE, REFERENCE_HASH)
    result = calculate(pd.read_csv(INPUT))
    expected = pd.read_csv(REFERENCE).set_index('parameter').loc[PARAMETERS, COLUMNS]
    actual = result.set_index('parameter').loc[PARAMETERS, COLUMNS]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    report = {
        'status': 'PASS', 'conditions': 1260, 'replicates_per_condition': 32,
        'bootstrap_draws': 2000, 'random_seed': 20260901,
        'checked_values': 18, 'bootstrap_intervals_recomputed': True,
        'maximum_absolute_difference': float(np.max(np.abs(actual.to_numpy() - expected.to_numpy()))),
        'results': result.drop(columns='scope').to_dict(orient='records'),
        'source_sha256': SOURCE_HASH, 'input_sha256': INPUT_HASH,
        'reference_sha256': REFERENCE_HASH,
    }
    return result, report


def verify():
    return reproduce()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'reproduced_local/btb_sensitivity')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'reproduced_local'):
        parser.error('Use reproduced_local/ or a directory outside the repository for outputs')
    result, report = reproduce()
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'btb_sensitivity.csv', index=False)
    (output / 'calculation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
