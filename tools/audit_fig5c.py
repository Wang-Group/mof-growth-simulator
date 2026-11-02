"""Check Fig. 5C reference geometry, fixed data, and mesh refinement offline."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / '16_fig5c_reconstruction'
NS = {'s': 'http://www.w3.org/2000/svg'}
TOKEN = re.compile(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
GRIDS = [(17, 19, 11), (34, 38, 22), (67, 75, 43), (133, 149, 85)]
# Figure-level engineering tolerances, not statistical equivalence margins.
CRITERIA = {'historical_iou_min': .99, 'edge_p95_pt_max': .5,
            'edge_max_pt_max': 1., 'refinement_iou_min': .995,
            'centroid_logrho_p95_max': .005, 'centroid_full_axis_fraction_max': .01,
            'centroid_P_p95_max': .005, 'camera_residual_pt_max': 1e-5,
            'author_coordinate_error_pt_max': .001, 'mask_iou_resolution_delta_max': .001}


def core():
    spec = importlib.util.spec_from_file_location('_fig5c_render', GROUP/'render.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_points(d):
    """Read the M/m/L/l/Z/z subset used by the archived triangle/grid paths."""
    tokens = TOKEN.findall(d)
    position = np.zeros(2)
    points, command, i = [], None, 0
    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
            if command in 'zZ':
                break
            if command not in 'MmLl':
                raise ValueError(f'Unsupported SVG path command: {command}')
        if command is None or i+1 >= len(tokens):
            raise ValueError('Incomplete SVG path')
        xy = np.array([float(tokens[i]), float(tokens[i+1])])
        position = position+xy if command.islower() else xy
        points.append(position.copy())
        i += 2
        command = 'l' if command == 'm' else 'L' if command == 'M' else command
    return np.asarray(points)


def project(xyz, camera):
    homogeneous = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], axis=-1)
    p = homogeneous @ camera.T
    return p[..., :2]/p[..., 2, None]


def reference():
    svg_bytes = gzip.decompress((GROUP/'reference/historical_panel_C.svg.gz').read_bytes())
    root = ET.fromstring(svg_bytes)
    collection = root.find('.//s:g[@id="Poly3DCollection_1"]', NS)
    triangles = np.asarray([path_points(p.get('d')) for p in collection])
    if triangles.shape != (14488, 3, 2):
        raise ValueError('Unexpected historical triangle collection')
    author = json.loads(gzip.decompress((GROUP/'reference/author_surface_paths.json.gz').read_bytes()))
    author_triangles = np.asarray([path_points(d) for d in author['path_d']])
    if author_triangles.shape != triangles.shape:
        raise ValueError('Author composite has a different mesh count')
    author_error = float(np.linalg.norm(author_triangles-triangles, axis=-1).max())
    # Calibrate only from axis/grid lines, never from the surface being tested.
    xmin, xmax, ymin, ymax, zmin, zmax = -2, np.log10(1500), -2, 2, 0, 2
    world, svg = [], []
    for axis, ticks in enumerate([[-2,-1,0,1,2,xmax],[-1,0,1,2],[0,.5,1,1.5,2]]):
        paths = root.find(f'.//s:g[@id="Line3DCollection_{axis+1}"]', NS)
        if len(paths) != len(ticks):
            raise ValueError('Historical tick/grid layout has changed')
        for t, path in zip(ticks, paths):
            coords = ([(t,ymin,zmin),(t,ymax,zmin),(t,ymax,zmax)] if axis == 0 else
                      [(xmin,t,zmax),(xmin,t,zmin),(xmax,t,zmin)] if axis == 1 else
                      [(xmax,ymax,t),(xmin,ymax,t),(xmin,ymin,t)])
            world.extend(coords)
            svg.extend(path_points(path.get('d')))
    world, svg = np.array(world), np.array(svg)
    a, b = [], []
    for xyz, xy in zip(world, svg):
        h = np.r_[xyz, 1]
        a.extend([np.r_[h, np.zeros(4), -xy[0]*xyz],
                  np.r_[np.zeros(4), h, -xy[1]*xyz]])
        b.extend(xy)
    fitted = np.linalg.lstsq(a, b, rcond=None)[0]
    camera = np.r_[fitted, 1].reshape(3, 4)
    error = float(np.linalg.norm(project(world, camera)-svg, axis=-1).max())
    return root, triangles, camera, {'grid_control_points': len(world),
            'camera_residual_max_pt': error, 'author_vertex_difference_max_pt': author_error,
            'author_composite_sha256': author['source_sha256'],
            'historical_svg_sha256': hashlib.sha256(svg_bytes).hexdigest()}


def mask(triangles, lower, scale, shape):
    image = Image.new('1', tuple(shape), 0)
    draw = ImageDraw.Draw(image)
    coords = (triangles-lower)*scale
    if (coords < 0).any() or (coords > (np.asarray(shape)-1)).any():
        raise ValueError('Comparison viewport would clip a triangle')
    for triangle in coords:
        draw.polygon([tuple(p) for p in triangle], fill=1)
    return np.asarray(image, dtype=bool)


def overlap(a, b, lower, scale, shape):
    ma, mb = mask(a, lower, scale, shape), mask(b, lower, scale, shape)
    ea, eb = ma ^ binary_erosion(ma), mb ^ binary_erosion(mb)
    distances = np.r_[distance_transform_edt(~ea)[eb], distance_transform_edt(~eb)[ea]]/scale
    return {'iou': float((ma & mb).sum()/(ma | mb).sum()),
            'edge_p95_svg_pt': float(np.quantile(distances, .95)),
            'edge_max_svg_pt': float(distances.max())}


def source_checks():
    manifest = json.loads((GROUP/'source_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = (GROUP/row['path']).resolve()
        if not path.is_relative_to(GROUP):
            raise ValueError('Manifest path escapes group')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Changed packaged source: {row["path"]}')
        if row['operation'] == 'gzip':
            if hashlib.sha256(gzip.decompress(data)).hexdigest() != row['source_sha256']:
                raise ValueError('Reference decompression hash mismatch')
    old = ROOT/'07_publication_sources/MS_Figure_5/content_checked_20261023/source_data/common_response_surface_v2_final_condition_summary.csv'
    if core().digest(old) != core().INPUT_HASH:
        raise ValueError('Group 07 and group 16 inputs are not identical')
    return len(manifest['files'])


def verify(output=None):
    sources = source_checks()
    mod = core()
    frame, axes, cube = mod.load_data()
    root, old, camera, ref = reference()
    lower, upper = old.reshape(-1, 2).min(0)-10, old.reshape(-1, 2).max(0)+10
    scale = 8
    shape = np.ceil((upper-lower)*scale).astype(int)+1
    records, projections = [], []
    for grid in GRIDS:
        tri = mod.mesh(axes, cube, grid)
        projected = project(tri, camera)
        record = {'grid': list(grid), 'triangles': len(tri),
                  'historical': overlap(old, projected, lower, scale, shape),
                  **mod.mesh_errors(tri, axes, cube)}
        records.append(record)
        projections.append(projected)
    for record, projected in zip(records, projections):
        record['finest'] = overlap(projected, projections[-1], lower, scale, shape)
    selected = records[2]
    mask_resolution = {}
    for res in (4, 16):
        sh = np.ceil((upper-lower)*res).astype(int)+1
        mask_resolution[str(res)] = overlap(old, projections[2], lower, res, sh)
    trend = {}
    for axis, name in enumerate(['rho_off', 'M/BTB', 's_close']):
        diff = np.diff(cube, axis=axis)
        trend[name] = {'increasing_adjacent_pairs': int((diff > 0).sum()),
                       'decreasing_adjacent_pairs': int((diff < 0).sum()),
                       'equal_adjacent_pairs': int((diff == 0).sum()),
                       'mean_P_at_axis_levels': cube.mean(axis=tuple(i for i in range(3) if i != axis)).tolist()}
    q_sc = np.stack(np.meshgrid(axes[1], axes[2], indexing='ij'), axis=-1).reshape(-1, 2)
    roots = mod.exact_roots(q_sc, axes, cube)
    criteria = {
        'camera_from_axes_not_surface': ref['camera_residual_max_pt'] < CRITERIA['camera_residual_pt_max'],
        'author_composite_matches_reference': ref['author_vertex_difference_max_pt'] < CRITERIA['author_coordinate_error_pt_max'],
        'historical_projected_area': selected['historical']['iou'] >= CRITERIA['historical_iou_min'],
        'historical_projected_edge_p95': selected['historical']['edge_p95_svg_pt'] <= CRITERIA['edge_p95_pt_max'],
        'historical_projected_edge_max': selected['historical']['edge_max_svg_pt'] <= CRITERIA['edge_max_pt_max'],
        'refinement_projected_area': selected['finest']['iou'] >= CRITERIA['refinement_iou_min'],
        'centroid_rho_p95': selected['centroid_log10_rho_error_p95'] <= CRITERIA['centroid_logrho_p95_max'],
        'centroid_rho_max_full_axis_fraction': selected['centroid_error_max_fraction_full_logrho_axis'] <= CRITERIA['centroid_full_axis_fraction_max'],
        'centroid_probability_p95': selected['centroid_P_error_p95'] <= CRITERIA['centroid_P_p95_max'],
        'root_error_improves_each_refinement': bool(np.all(np.diff([r['centroid_log10_rho_error_p95'] for r in records]) < 0)),
        'raster_resolution_stable': abs(mask_resolution['4']['iou']-mask_resolution['16']['iou']) < CRITERIA['mask_iou_resolution_delta_max'],
        'rho_monotonic_all_126_slices': trend['rho_off']['increasing_adjacent_pairs'] == 0,
        'exact_probability_identity': bool(np.array_equal(frame.P, frame.persistent_assembly_probability)),
    }
    report = {'status': 'PASS' if all(criteria.values()) else 'FAIL',
              'scope': 'Projected historical boundary and new interpolant mesh convergence, not exact historical 3D-mesh recovery',
              'source_files_verified': sources, 'conditions': len(frame), 'traces_per_condition': 32,
              'probability_classes': [int((frame.P <= .1).sum()), int(((frame.P > .1) & (frame.P < .9)).sum()), int((frame.P >= .9).sum())],
              'reference': ref, 'selected_grid': list(mod.DEFAULT_GRID), 'grids': records,
              'mask_pixels_per_svg_pt': scale, 'mask_resolution_check': mask_resolution,
              'criteria_type': 'Figure-level engineering checks, not statistical equivalence or uncertainty bounds',
              'tolerances': CRITERIA, 'checks': criteria, 'trends': trend,
              'limitations': ['Only one historical camera projection is available; full original 3D coordinates are unavailable.',
                  'Pointwise centroid errors are unweighted across triangles; worst-case local rho error is reported.',
                  'P uses 32 launched trajectories per condition; no bootstrap or physical-model uncertainty analysis here.',
                  'The missing historical module remains unrecovered. No MS/SI or author artwork is replaced.']}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        pd.json_normalize(records).to_csv(output/'grid_convergence.csv', index=False)
        pd.DataFrame({'q': 10**q_sc[:, 0], 's_close': q_sc[:, 1], 'rho_at_P_half': 10**roots}).to_csv(
            output/'boundary_at_sampled_q_sc.csv', index=False)
        draw_summary(output, root, old, projections, lower, upper, records, mod, axes, cube)
    if not all(criteria.values()):
        raise AssertionError({k:v for k,v in criteria.items() if not v})
    return report


def draw_summary(output, root, old, projections, lower, upper, records, mod, axes, cube):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    for ax, entries, title in [
        (axs[0, 0], [(old, '#5185c0', 'Historical'), (projections[2], '#d26d4f', 'Reconstructed')], 'Projected Boundary'),
        (axs[0, 1], [(projections[i], c, str(GRIDS[i])) for i,c in [(0,'#999999'),(2,'#d26d4f'),(3,'#267b70')]], 'Grid Refinement')]:
        scale = 8
        shape = np.ceil((upper-lower)*scale).astype(int)+1
        for triangles, color, label in entries:
            m = mask(triangles, lower, scale, shape)
            ax.contour(lower[0]+np.arange(shape[0])/scale, lower[1]+np.arange(shape[1])/scale,
                       m, levels=[.5], colors=[color], linewidths=1.3)
            ax.plot([], [], color=color, label=label)
        for i in range(1, 4):
            xy = path_points(root.find(f'.//s:g[@id="pane3d_{i}"]//s:path', NS).get('d'))
            xy = np.r_[xy, xy[:1]]
            ax.plot(*xy.T, color='.85', lw=.7, zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(lower[0], upper[0]); ax.set_ylim(upper[1], lower[1])
        ax.axis('off'); ax.set_title(title); ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax = axs[1, 0]
    q = np.linspace(axes[1][0], axes[1][-1], 401)
    for sc, color in [(0, '#5185c0'),(.5, '#d26d4f'),(2, '#267b70')]:
        rho = 10**mod.exact_roots(np.c_[q, np.full(len(q), sc)], axes, cube)
        ax.plot(10**q, rho, color=color, label=rf'$s_{{close}} = {sc:g}$')
    ax.set(xscale='log', yscale='log', xlabel='M/BTB', ylabel=r'$\rho_{off}$ at $P = 0.5$',
           title='Boundary Sections (Fixed Interpolant)')
    ax.legend(frameon=False); ax.grid(alpha=.18)
    ax = axs[1, 1]
    x = np.arange(len(records))
    for key, label, c in [('centroid_log10_rho_error_p95','95th Percentile','#5185c0'),('centroid_log10_rho_error_max','Maximum','#d26d4f')]:
        ax.plot(x, [r[key] for r in records], 'o-', color=c, label=label)
    ax.set_xticks(x, ['17/19/11','34/38/22','67/75/43','133/149/85'], rotation=15)
    ax.set(yscale='log', xlabel='Grid Dimensions (Rho / Q / SC)', ylabel=r'Centroid Error in $\log_{10}(\rho_{off})$',
           title='Triangle Approximation Error')
    ax.legend(frameon=False); ax.grid(alpha=.18)
    fig.savefig(output/'boundary_validation.png', dpi=200)
    fig.savefig(output/'boundary_validation.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced_local/Fig5C_validation')
    args = parser.parse_args()
    report = verify(args.output)
    print(json.dumps({'status':report['status'], 'checks':report['checks'], 'output':str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
