"""Portable Fig. 5C reconstruction; never executes a KMC simulation."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'data/common_response_surface_v2_final_condition_summary.csv'
DEFAULT_GRID = (67, 75, 43)
INPUT_HASH = '866fee5ec5a3b0e4f80bc776b4f1880bdfb0f0694cc0bc0e02ac0ddb95e31763'
SOURCE_HASH = '69e3cbf879e267e093e5ec81741006517cfb7fa70f559eab0e845f732d41888d'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_module():
    path = HERE / 'source/rebuild_figure5c_standalone.py'
    if digest(path) != SOURCE_HASH:
        raise ValueError('The archived independent implementation has changed')
    spec = importlib.util.spec_from_file_location('_fig5c_independent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(path=INPUT):
    if digest(path) != INPUT_HASH:
        raise ValueError('Input differs from the verified response-surface table')
    module = original_module()
    frame, axes, cube = module.load_conditions(Path(path))
    if not np.array_equal(frame.P, frame.persistent_assembly_probability):
        raise ValueError('Probabilities must exactly match the archived column')
    return frame, axes, cube


def mesh(axes, cube, grid=DEFAULT_GRID):
    if len(grid) != 3 or any(type(n) is not int or n < 3 for n in grid):
        raise ValueError('Grid must contain three integers >= 3')
    if np.prod(grid) > 2_000_000:
        raise ValueError('Grid exceeds the bounded-memory validation limit')
    module = original_module()
    module.FINE_GRID = tuple(grid)
    return module.boundary_triangles(axes, cube)


def exact_roots(q_sc, axes, cube):
    """Return the rightmost bracketed P=0.5 root of the fixed interpolant."""
    q_sc = np.asarray(q_sc, dtype=float)
    if q_sc.ndim != 2 or q_sc.shape[1] != 2:
        raise ValueError('Expected an N x 2 array of log10(q), s_close')
    if (np.diff(cube, axis=0) > 0).any():
        raise ValueError('Boundary-root audit requires nonincreasing P along rho')
    interpolator = RegularGridInterpolator(axes, cube, bounds_error=True)
    result = []
    for start in range(0, len(q_sc), 10000):
        pairs = q_sc[start:start+10000]
        xyz = np.c_[np.tile(axes[0], len(pairs)), np.repeat(pairs, len(axes[0]), axis=0)]
        values = interpolator(xyz).reshape(len(pairs), -1)
        indices = (values >= .5).sum(axis=1) - 1
        if ((indices < 0) | (indices >= len(axes[0])-1)).any():
            raise ValueError('No bracketed P=0.5 root in the sampled domain')
        rows = np.arange(len(pairs))
        left, right = values[rows, indices], values[rows, indices+1]
        result.append(axes[0][indices] + (.5-left)/(right-left) *
                      (axes[0][indices+1]-axes[0][indices]))
    return np.concatenate(result) if result else np.array([])


def mesh_errors(triangles, axes, cube):
    centers = triangles.mean(axis=1)
    error = np.abs(centers[:, 0] - exact_roots(centers[:, 1:], axes, cube))
    interpolator = RegularGridInterpolator(axes, cube, bounds_error=True)
    probability_error = np.abs(interpolator(centers)-.5)
    return {
        'centroid_log10_rho_error_p95': float(np.quantile(error, .95)),
        'centroid_log10_rho_error_max': float(error.max()),
        'centroid_rho_relative_error_bound_max': float(10**error.max()-1),
        'centroid_error_max_fraction_full_logrho_axis': float(error.max()/np.ptp(axes[0])),
        'centroid_P_error_p95': float(np.quantile(probability_error, .95)),
        'centroid_P_error_max': float(probability_error.max()),
        'vertex_P_error_max': float(np.abs(interpolator(triangles.reshape(-1, 3))-.5).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, nargs=3, default=DEFAULT_GRID,
                        metavar=('RHO', 'Q', 'SC'))
    parser.add_argument('--output', type=Path,
                        default=HERE.parent/'reproduced_local/Fig5C/Figure5C')
    args = parser.parse_args()
    frame, axes, cube = load_data()
    triangles = mesh(axes, cube, tuple(args.grid))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    module = original_module()
    # Fix the archived legend's inclusive middle interval; the data are unchanged.
    original_line = module.Line2D
    def line(*a, **kw):
        if kw.get('label') == r'Calculated: $0.1 \leq P \leq 0.9$':
            kw['label'] = r'Calculated: $0.1 < P < 0.9$'
        return original_line(*a, **kw)
    module.Line2D = line
    shown = module.plot_panel(frame, triangles, args.output, 'outlined', 1.2)
    report = {'implementation': 'Independent reconstruction, not the missing historical module',
              'input_sha256': digest(INPUT), 'archived_script_sha256': SOURCE_HASH,
              'entry_sha256': digest(__file__), 'grid': list(args.grid),
              'triangles': len(triangles), 'conditions': len(frame),
              'displayed': shown, 'mesh_errors': mesh_errors(triangles, axes, cube),
              'coordinate_system': 'log10(rho_off), log10(M/BTB), linear s_close',
              'interpolation': 'Trilinear; no smoothing, fitting, or extrapolation',
              'historical_module_recovered': False, 'MS_SI_modified': False}
    args.output.with_name(args.output.name+'_method.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
