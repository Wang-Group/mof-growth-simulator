"""Figure-specific adapter derived from the previously verified runtime."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
import types

def load(path, overrides=None, definitions_only=False):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    overrides = dict(overrides or {})
    body = []
    for node in tree.body:
        if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
            continue
        if definitions_only and (not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))):
            continue
        names = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if names and names[0] in overrides:
            node = ast.Assign(targets=[ast.Name(names[0], ast.Store())], value=ast.Subscript(ast.Name('_overrides', ast.Load()), ast.Constant(names[0]), ast.Load()))
        body.append(node)
    tree.body = body
    ast.fix_missing_locations(tree)
    name = '_native_' + path.stem
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__['_overrides'] = overrides
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    module.__dict__.update(overrides)
    return module

def save(fig, output, preview=True):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for artist in ax.collections:
            artist.set_rasterized(False)
    fig.savefig(output, format='svg', facecolor='white', bbox_inches='tight', pad_inches=0.035)
    if preview:
        fig.savefig(output.with_suffix('.png'), dpi=150, facecolor='white', bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

def run(group_dir, panel, config_override=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np
    import pandas as pd
    group = Path(group_dir)
    config = config_override or json.loads((group / 'config.json').read_text(encoding='utf-8'))
    kind = config['kind']
    code = group / 'original_code'
    data = group / 'source_data'
    output = group / f'panel_{panel}' / f"{config['figure']}_{panel}.svg"
    output.parent.mkdir(exist_ok=True)
    style = code / 'kernel.py'
    mpl.rcParams['svg.fonttype'] = 'none'
    if kind == 'surface':
        files = {k: data / n for k, n in {'mismatch': 'figure5_surface_vs_seeded_growth_sc1_v1_mismatched_surface.csv', 'matched': 'figure5_surface_vs_seeded_growth_sc1_v1_matched_surface_control.csv', 'seed': 'figure5_surface_vs_seeded_growth_sc1_v1_seeded.csv'}.items()}
        m = load(code / 'plot_figure5_surface_nucleation_v2.py', {'FILES': files, 'HERE': group / '_temporary'})
        m.HERE.mkdir(exist_ok=True)
        mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
        frame = m.load_data()
        fig, ax = plt.subplots(figsize=(4.6, 3.55))
        getattr(m, config['function'])(ax, frame)
        save(fig, output)
    else:
        raise ValueError('This runner only supports the verified figure job')
