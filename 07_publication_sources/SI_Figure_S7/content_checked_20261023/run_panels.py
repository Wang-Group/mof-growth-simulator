"""Run copied, content-checked panel code; outputs stay in this packet."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    group=Path(__file__).resolve().parent
    os.environ['MPLCONFIGDIR']=str(group/'_mpl_cache')
    os.environ['MPLBACKEND']='Agg'
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',help='One panel ID; omitted means all declared panels')
    parser.add_argument('--list',action='store_true',help='List panel IDs without plotting')
    args=parser.parse_args()
    config=json.loads((group/'panel_jobs.json').read_text(encoding='utf-8'))
    if args.list:
        print('\n'.join(config));return
    if args.panel and args.panel not in config:
        raise ValueError('Panel is not in this source packet: '+args.panel)
    source_root=os.path.normcase(os.path.abspath(r'D:\SYH\课题\MOL生长模拟\论文'))
    def audit(event,arguments):
        if event=='open':
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if not write:return
        elif event in {'os.remove','os.rmdir','os.mkdir','os.rename','os.replace','os.truncate','shutil.copyfile','shutil.copymode','shutil.copystat'}:
            path=arguments[-1] if event=='shutil.copyfile' else arguments[0]
        else:return
        if isinstance(path,(str,bytes,os.PathLike)):
            value=os.path.normcase(os.path.abspath(os.fsdecode(path))).removeprefix('\\\\?\\')
            if value==source_root or value.startswith(source_root+os.sep):
                raise PermissionError('Original paper tree is read-only in this runner: '+value)
    sys.addaudithook(audit)
    import panel_runtime
    # Data heatmaps use vector cells. This operates on numeric arrays only.
    import matplotlib.axes
    import numpy as np
    original_imshow=matplotlib.axes.Axes.imshow
    def numeric_imshow(ax,values,*args,**kwargs):
        a=np.asanyarray(values)
        if a.ndim!=2:return original_imshow(ax,values,*args,**kwargs)
        origin=kwargs.pop('origin','upper')
        extent=kwargs.pop('extent',None)
        aspect=kwargs.pop('aspect',None)
        if aspect is not None:ax.set_aspect(aspect)
        for name in ('interpolation','filternorm','filterrad','resample','interpolation_stage'):kwargs.pop(name,None)
        rows,cols=a.shape
        if extent is None:
            extent=(-.5,cols-.5,rows-.5,-.5) if origin=='upper' else (-.5,cols-.5,-.5,rows-.5)
        x0,x1,y0,y1=extent
        x=np.linspace(x0,x1,cols+1)
        y=np.linspace(y0,y1,rows+1)
        if origin=='upper':y=y[::-1]
        if kwargs.get('norm') is not None:
            kwargs.pop('vmin',None);kwargs.pop('vmax',None)
        artist=ax.pcolormesh(x,y,a,shading='flat',rasterized=False,**kwargs)
        ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
        return artist
    matplotlib.axes.Axes.imshow=numeric_imshow
    try:
        for panel,job in config.items():
            if args.panel and panel!=args.panel:continue
            panel_runtime.run(group,panel,job)
    finally:
        matplotlib.axes.Axes.imshow=original_imshow


if __name__=='__main__':
    main()
