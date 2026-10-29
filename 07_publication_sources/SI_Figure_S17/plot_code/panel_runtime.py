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

def export_one(module, function, token, output):
    """Select a drawing save, never crop/read the composite artwork."""
    import matplotlib.figure
    import matplotlib.pyplot as plt
    original = matplotlib.figure.Figure.savefig
    captured = []

    def selected(fig, path, *args, **kwargs):
        if Path(path).suffix.lower() == '.svg' and token in Path(path).stem:
            kwargs.pop('format', None)
            original(fig, output, format='svg', **kwargs)
            original(fig, Path(output).with_suffix('.png'), dpi=150, facecolor='white', bbox_inches='tight')
            captured.append(str(output))
    matplotlib.figure.Figure.savefig = selected
    try:
        getattr(module, function)()
    finally:
        matplotlib.figure.Figure.savefig = original
        plt.close('all')
    if len(captured) != 1:
        raise RuntimeError(f'Expected exactly one independently drawn panel, got {captured}')

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
    if kind == 'native_exporter':
        safe_functions = {'export_s26_standalone', 'export_s27_standalone', 'export_s29_standalone', 'export_s58_s59_standalone', 'export_s24', 'export_s50'}
        if config['function'] not in safe_functions:
            raise ValueError('Unapproved exporter')
        m = load(code / 'export_si_subpanels_svg.py', {'BASE': data, 'WORK': data, 'OUT': group / '_temporary'})
        replacements = {'pxrd_dir': data, 'data_dir': data, 'analysis_dir': data, 'source_dir': data, 'geometry_json': data / 's29_geometry_centers_protocol5.json'}
        if config['function'] == 'export_s24':
            replacements['path'] = data / 'figureS_btb_complete_metric_grid_source.csv'
        if config['function'] == 'export_s50':
            replacements['path'] = data / 'figure8_condition_source_v2.csv'

        class Remap(ast.NodeTransformer):

            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id in replacements):
                    node.value = ast.Subscript(ast.Name('_local_inputs', ast.Load()), ast.Constant(node.targets[0].id), ast.Load())
                return self.generic_visit(node)
        tree = ast.parse((code / 'export_si_subpanels_svg.py').read_text(encoding='utf-8-sig'))
        fun = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == config['function']))
        remapped = ast.fix_missing_locations(ast.Module(body=[Remap().visit(fun)], type_ignores=[]))
        m.__dict__['_local_inputs'] = replacements
        exec(compile(remapped, str(code / 'export_si_subpanels_svg.py'), 'exec'), m.__dict__)
        m.setup_style()
        export_one(m, config['function'], config['tokens'][panel], output)
    else:
        raise ValueError('This runner only supports the verified figure job')
