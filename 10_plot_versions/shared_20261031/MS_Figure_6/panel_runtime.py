"""Run selected native drawing functions against local, copied inputs only."""
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
        if definitions_only and not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        names = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if names and names[0] in overrides:
            node = ast.Assign(targets=[ast.Name(names[0], ast.Store())],
                              value=ast.Subscript(ast.Name('_overrides', ast.Load()), ast.Constant(names[0]), ast.Load()))
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
            kwargs.pop('format',None)
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


def remap_function(module, filename, function, replacements):
    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
                node.value=ast.Subscript(ast.Name('_function_inputs',ast.Load()),ast.Constant(node.targets[0].id),ast.Load())
            return self.generic_visit(node)
    tree=ast.parse(Path(filename).read_text(encoding='utf-8-sig'))
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    new=ast.fix_missing_locations(ast.Module(body=[Replace().visit(fun)],type_ignores=[]))
    module.__dict__['_function_inputs']=replacements
    exec(compile(new,str(filename),'exec'),module.__dict__)

def run(group_dir, panel, config_override=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np
    import pandas as pd
    group = Path(group_dir)
    config = config_override or json.loads((group/'config.json').read_text(encoding='utf-8'))
    kind = config['kind']
    code = group/'original_code'
    data = group/'source_data'
    output = group/f'panel_{panel}'/f'{config["figure"]}_{panel}.svg'
    output.parent.mkdir(exist_ok=True)
    style = code/'kernel.py'
    mpl.rcParams['svg.fonttype']='none'
    if kind == 'figure1D':
        m = load(code/'plot_figure1D_code_generated.py', {'DATA':data/'figure1D_data.csv','OUT':output,'PNG':output.with_suffix('.png')})
    elif kind == 'figure4':
        m = load(code/'build_figure3_revision_blue_teal_CDE_v4.py', {
            'BASE_SCRIPT':code/'plot_figure3_prototype_v1.py','STYLE_KERNEL':style,
            'RESPONSE_SOURCE':data/'figure3_response_surface_source_v11.csv',
            'SENSITIVITY_SOURCE':data/'figure3_local_sensitivity_source_v11.csv'})
        m.STYLE.apply_figure_style(frame='open',font='Arial',sizes=(8.6,7.6,6.8),grid=False)
        response,sensitivity,rate_max=m.prepare_data()
        if panel in 'AB':
            fig,ax=plt.subplots(figsize=(4.0,4.1) if panel=='A' else (4.8,3.5))
            (m.definition_panel if panel=='A' else lambda ax:m.anchor_comparison(ax,response))(ax)
            m.STYLE.panel_letter(ax,panel,case='upper',dx=-.06 if panel=='A' else -.10,dy=1.03 if panel=='A' else 1.05,fontsize=11)
        else:
            value,cmap,norm,title,cbar={
                'C':('P_2_to_30',m.RESPONSE_BLUE,Normalize(0,1),'Nucleation probability\n'+r'$P_{2\rightarrow30}$','Probability'),
                'D':('P_76_to_126',m.RESPONSE_BLUE_TEAL,Normalize(0,1),'Crystal growth probability\n'+r'$P_{76\rightarrow126}$','Probability'),
                'E':('G_76_to_126_entities_per_model_time',m.RESPONSE_TEAL,Normalize(0,rate_max),'Growth rate\n'+r'$G_{76\rightarrow126}$','Entities per model time')}[panel]
            fig=plt.figure(figsize=(4.8,4.45));grid=fig.add_gridspec(2,1,height_ratios=[1,.13],hspace=.015)
            ax=fig.add_subplot(grid[0,0],projection='3d');cax=fig.add_subplot(grid[1,0]);cax.set_position([.20,.075,.60,.018])
            m.stacked_response(ax,cax,response,value,cmap_name=cmap,norm=norm,title=title,colorbar_label=cbar,show_y_label=True,show_z_label=True)
            ax.text2D(-.08,1.03,panel,transform=ax.transAxes,fontsize=11,fontweight='bold',va='top',ha='left')
        save(fig,output)
    elif kind == 'sensitivity':
        m=load(code/'build_S54_subpanels.py',{'FIGURE3_SCRIPT':code/'build_figure3_revision_v11.py','STYLE_KERNEL':style,'S54_DIR':group/'_temporary'})
        fig3=load(code/'build_figure3_revision_v11.py',{'BASE_SCRIPT':code/'plot_figure3_prototype_v1.py','STYLE_KERNEL':style})
        fig3.STYLE.apply_figure_style(frame='open',font='Arial',sizes=(8.6,7.6,6.8),grid=False)
        sensitivity=pd.read_csv(data/'figure3_local_sensitivity_source_v11.csv')
        idx=int(panel[1:])-1;process='assembly_J_1_to_30' if idx<3 else 'seed_growth_J_76_to_126';label='Nucleation' if idx<3 else 'Growth';sc=[.1,1.,2.][idx%3]
        fig,ax=plt.subplots(figsize=(3.25,2.85))
        norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=.5);cmap=mpl.colormaps['RdBu_r'].copy();cmap.set_bad(fig3.MISSING)
        m.vector_sensitivity_heatmap(fig3,ax,sensitivity,process,sc,norm=norm,cmap=cmap,show_y=True,show_x=True,row_label=label)
        ax.set_title(rf'{label}: $S_{{\mathrm{{close}}}}={sc:g}$',fontsize=9.2,fontweight='bold',pad=5)
        fig.subplots_adjust(left=.22,right=.98,bottom=.25,top=.84);save(fig,output)
    elif kind == 's7':
        m=load(code/'export_figure_s7_panels.py')
        original_argv=sys.argv[:]
        sys.argv=[str(code/'export_figure_s7_panels.py'),'--data-dir',str(data),'--output-dir',str(group/'_temporary'),'--qa-dir',str(group/'_temporary'/'qa')]
        # The historical script's export name S8 is only a name; final numbering is S7.
        def selected_save(fig, output_dir, qa_dir, name, source):
            chosen=name.startswith(f'Figure_S8{panel}_')
            if chosen:save(fig,output)
            else:plt.close(fig)
            return {'file':output.name if chosen else name+'.svg','written':chosen,'vector_only':True}
        m.save=selected_save
        try:m.main()
        finally:sys.argv=original_argv
    elif kind == 'native_exporter':
        # Remap assignments inside the selected drawing function too. No filesystem
        # links or original-library paths are consulted by this runtime.
        safe_functions={'export_s26_standalone','export_s27_standalone','export_s29_standalone','export_s58_s59_standalone','export_s24','export_s50'}
        if config['function'] not in safe_functions: raise ValueError('Unapproved exporter')
        m=load(code/'export_si_subpanels_svg.py',{'BASE':data,'WORK':data,'OUT':group/'_temporary'})
        replacements={'pxrd_dir':data,'data_dir':data,'analysis_dir':data,'source_dir':data,'geometry_json':data/'s29_geometry_centers_protocol5.json'}
        if config['function']=='export_s24':replacements['path']=data/'figureS_btb_complete_metric_grid_source.csv'
        if config['function']=='export_s50':replacements['path']=data/'figure8_condition_source_v2.csv'
        class Remap(ast.NodeTransformer):
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
                    node.value=ast.Subscript(ast.Name('_local_inputs',ast.Load()),ast.Constant(node.targets[0].id),ast.Load())
                return self.generic_visit(node)
        tree=ast.parse((code/'export_si_subpanels_svg.py').read_text(encoding='utf-8-sig'))
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==config['function'])
        remapped=ast.fix_missing_locations(ast.Module(body=[Remap().visit(fun)],type_ignores=[]))
        m.__dict__['_local_inputs']=replacements
        exec(compile(remapped,str(code/'export_si_subpanels_svg.py'),'exec'),m.__dict__)
        m.setup_style()
        export_one(m,config['function'],config['tokens'][panel],output)
    elif kind == 'repair':
        m=load(code/'redraw_S49_and_S29_legend.py',{'S49_A_CSV':data/'A'/'repair_progress_vs_growth.csv','S49_B_CSV':data/'B'/'repair_progress_vs_growth.csv'})
        m.set_style()
        export_one(m,'draw_s49',f'Figure_S49_{panel}',output)
    elif kind == 'steric':
        m=load(code/'redraw_S49_and_S29_legend.py',{'S29_SOURCE':data})
        m.set_style();summary=m.load_s29();k='ABC'.index(panel)
        fig,ax=plt.subplots(figsize=(4.8,3.7));fig.subplots_adjust(left=.20,bottom=.18,right=.98,top=.64)
        m.add_s29_curves(ax,summary,k);m.format_s29_axis(ax,k,panel)
        h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.54,.995),ncol=1,frameon=False,fontsize=8.2,handlelength=2.8)
        save(fig,output)
    elif kind == 'pxrd39':
        m=load(code/'draw_available_panels.py',{'ROOT':data,'HERE':group,'SVG_DIR':group/'_temporary','PREVIEW_DIR':group/'_temporary'})
        # Original draw_s51 expects the archived relative directory names.
        remap_function(m,code/'draw_available_panels.py','draw_s51',{'source':data/'spectra','projection':data/'n2000_projection.csv'})
        export_one(m,'draw_s51',f'Figure_S51_{panel}',output)
    elif kind == 'btb_growth':
        m=load(code/'build_figure7_kinetic_window_v2.py',{'STYLE_KERNEL':style,'SUMMARY_PATH':data/'figure7_btb_final_morphology_sc1_v1.csv'})
        m.STYLE.apply_figure_style(frame='open',font='Arial',sizes=(8.6,7.6,6.8),grid=False)
        # These are the original renderer's own exported per-panel numeric tables.
        q1=pd.read_csv(data/'figure7_q1_kinetic_source_v2.csv')
        traces=pd.read_csv(data/'figure7_q1_successful_trajectory_source_v2.csv')
        if panel=='A':
            fig,ax=plt.subplots(figsize=(4.2,3.2))
            m.panel_a(ax,q1)
        else:
            layout=load(group/'figure6B_reference_style.py')
            fig,ax=layout.new_figure()
            artist=m.panel_b(ax,q1,traces)
            layout.finish(fig,ax,artist,q1)
        save(fig,output)
    elif kind in ('growth_count','retention'):
        m=load(code/'build_figure8_growth_window_v2.py')
        m.configure_style();summary=pd.read_csv(data/'figure8_absolute_outcomes_v1_condition_summary.csv')
        fig,ax=plt.subplots(figsize=(5.5,3.7))
        if kind=='growth_count':m.draw_count_panel(ax,m.build_count_table(summary))
        else:m.draw_retention_panel(ax,m.build_retention_table(summary,pd.read_csv(data/'figure8_absolute_outcomes_v1_trace_source.csv')))
        save(fig,output)
    elif kind == 'variance':
        m=load(code/'build_figure5_AC_redraw_v2.py',{'SENSITIVITY_TABLE':data/'figure6bc_current_parameter_phase_v1_sensitivity.csv','B_OUTPUT':output,'B_PREVIEW':output.with_suffix('.png')})
        m.panel_b()
    elif kind == 'pxrd6C':
        filename=code/'build_figures.py'
        tree=ast.parse(filename.read_text(encoding='utf-8-sig'))
        m=types.ModuleType('_pxrd')
        m.__dict__.update(np=np,pd=pd,plt=plt,matplotlib=mpl)
        unit=ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pxrd'],type_ignores=[]))
        exec(compile(unit,str(filename),'exec'),m.__dict__)
        m.np=np;m.pd=pd;m.plt=plt;m.matplotlib=mpl
        records=[]
        for rho in (10,30,50,100):
            for rep in range(4):
                p=data/'spectra'/f'btb_extend_rho{rho}_rep{rep:02d}_milestone_N02000.csv'
                records.append((rho,json.loads(p.with_suffix('.json').read_text()),pd.read_csv(p)))
        m.records=records;m.refs=pd.read_csv(data/'spectra/kgd_reference_positions.csv');m.colors={10:'#4477AA',30:'#228833',50:'#AA3377',100:'#BB8800'}
        fig,ax=plt.subplots(figsize=(3.35,2.55));fig.subplots_adjust(left=.17,right=.98,bottom=.20,top=.86);m.pxrd(ax,False);save(fig,output)
    elif kind == 'trajectory':
        m=load(code/config.get('core','plot_entities_broken_axis.py'))
        m.process_single_pkl(data/config['input'],output.parent,x_range=tuple(config['x_range']),y_range=tuple(config['y_range']),figsize=(9,6),dpi=150,x_label='Growth Time (source coordinate)',y_label='Number of Entities',zero_fraction=.03,output_name=output.name,y_ticks=range(0,int(config['y_range'][1])+1,int(config['tick'])))
        # The original selected window and raw time coordinate are retained.
        # Structural insets and annotation lines in the assembled artwork are manual.
    elif kind == 'surface':
        files={k:data/n for k,n in {'mismatch':'figure5_surface_vs_seeded_growth_sc1_v1_mismatched_surface.csv','matched':'figure5_surface_vs_seeded_growth_sc1_v1_matched_surface_control.csv','seed':'figure5_surface_vs_seeded_growth_sc1_v1_seeded.csv'}.items()}
        m=load(code/'plot_figure5_surface_nucleation_v2.py',{'FILES':files,'HERE':group/'_temporary'})
        m.HERE.mkdir(exist_ok=True)
        mpl.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
        frame=m.load_data();fig,ax=plt.subplots(figsize=(4.6,3.55))
        getattr(m,config['function'])(ax,frame);save(fig,output)
    elif kind == 'linkage':
        # Keep the original equations and plotting statements, but construct one
        # axis instead of the old two-panel canvas. No SVG or image is read.
        filename=code/'render_uio66_entity13_dls_linkage.py'
        m=load(filename,{'FIRST_PASSAGE_CURVES_CSV':data/'linkage/uio66_induction_vs_dls_first_passage_curves.csv','DLS_TRACE_CSV':data/'linkage/uio66_dls_size_gualtieri_overlay_v2_trace.csv','DLS_FIT_CSV':data/'linkage/uio66_dls_size_gualtieri_overlay_v2_fit_curve.csv','DLS_PARAMS_CSV':data/'linkage/uio66_dls_size_gualtieri_overlay_v2_parameters.csv'})
        tree=ast.parse(filename.read_text(encoding='utf-8-sig'))
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render')
        selected=[];axis='ax_sim' if panel=='D' else 'ax_dls'
        for node in fun.body:
            s=ast.unparse(node)
            if 'plt.subplots(' in s:
                selected.extend(ast.parse('fig, '+axis+' = plt.subplots(figsize=(3.65,3.65))').body)
                continue
            if any(t in s for t in ['OUTPUT_DIR.mkdir','fig.savefig','plt.close','fig.subplots_adjust']):continue
            if axis=='ax_sim' and 'ax_dls' in s:continue
            if axis=='ax_dls' and 'ax_sim' in s:continue
            selected.append(node)
        unit=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
        env=m.__dict__.copy();exec(compile(unit,str(filename),'exec'),env)
        save(env['fig'],output)
    elif kind == 'motif_bars':
        # Source notebook cells 19-21, selected entity only; original fillna(0)
        # denominator is deliberately retained rather than the later dropna table.
        from collections import Counter
        import os
        notebook=json.loads((code/'unique_structures.ipynb').read_text(encoding='utf-8'))
        env={'pd':pd,'plt':plt,'Counter':Counter,'os':os,'rcParams':mpl.rcParams}
        exec(''.join(notebook['cells'][19]['source']),env)
        frame=pd.read_csv(data/'unique_structures_result.csv').fillna(0)
        col='entity_'+panel[1:];counts=Counter(frame[col].astype(int));total=sum(counts.values())
        env.update(col=col,entity_cols=[col],prob_dict={col:{uid:n/total for uid,n in counts.items()}},colors=plt.cm.tab20.colors)
        source=''.join(notebook['cells'][21]['source']).replace('.png','.svg')
        # Redirect the original save call to the local SVG and retain all bar/style statements.
        m=types.ModuleType('_motif');m.draw=lambda:exec(compile(source,str(code/'unique_structures.ipynb')+':cell21','exec'),env)
        export_one(m,'draw','entity_',output)
    elif kind == 'group_bars':
        from collections import Counter
        import os
        notebook=json.loads((code/'unique_structures.ipynb').read_text(encoding='utf-8'))
        env={'pd':pd,'plt':plt,'Counter':Counter,'os':os,'rcParams':mpl.rcParams}
        exec(''.join(notebook['cells'][19]['source']),env)
        source=''.join(notebook['cells'][22]['source']).replace('.png','.svg')
        source=source.replace('data_dir = "_combined_hkust1_e1_to_50_pairtol_0p2"','data_dir = _local_data')
        source=source.replace('<= 30','<= 16').replace('range(1, 31)','range(1, 17)').replace('0.5, 20.5','0.5, 16.5')
        source=source.replace('title="Group ID"','title="Structure Index"')
        env['_local_data']=str(data)
        m=types.ModuleType('_groups');m.draw=lambda:exec(compile(source,str(code/'unique_structures.ipynb')+':cell22','exec'),env)
        export_one(m,'draw','group_summary',output)
    elif kind == 'repair_source':
        import csv
        m=types.ModuleType('_repair')
        m.__dict__.update(Path=Path,plt=plt,csv=csv)
        for name,allowed in [('analysis_utils.py',{'read_rows_csv'}),('plot_missing_bdc_repair_progress.py',{'first_row_with','make_plot'})]:
            tree=ast.parse((code/name).read_text(encoding='utf-8-sig'))
            unit=ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in allowed],type_ignores=[]))
            exec(compile(unit,str(code/name),'exec'),m.__dict__)
        rows=m.read_rows_csv(data/'repair_progress_vs_growth.csv')
        m.make_plot(rows,output.with_suffix('.png'),output)
    elif kind == 'mixed_source':
        from xml.sax.saxutils import escape
        import math
        filename=code/'plot_zr6_only_growth_trace.py'
        tree=ast.parse(filename.read_text(encoding='utf-8-sig'))
        # The original SVG plotter is independent of the simulation imports.
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'svg_polyline','linear_map','make_svg_plot'}]
        env={'Path':Path,'np':np,'math':math,'escape':escape}
        exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(filename),'exec'),env)
        payload=json.loads(next(data.glob('*.summary_trace.json')).read_text(encoding='utf-8'))
        env['make_svg_plot'](payload['replay_runs'],payload['mean_trace'],output,mean_alignment=payload.get('mean_alignment','time'))
    else:raise ValueError(kind)
    if not output.is_file():raise RuntimeError(f'Missing output {output}')
    print(str(output))
