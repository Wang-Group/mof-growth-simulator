"""Replot verified saved results and replace only the authorized vector panel."""
from pathlib import Path
from copy import deepcopy
from lxml import etree as E
import json,hashlib,io,base64
import numpy as np
import pandas as pd
from PIL import Image,ImageChops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import resvg_py
B=Path(__file__).resolve().parent;C=B.parent;I=C/'student_integration_20260922_v1'
O=B/'figures';O.mkdir(exist_ok=True);D=B/'source_data';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.linewidth':.8})
audit={'pxrd':[],'panels':{}}
def save(fig,name):
 fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
 for t in fig.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text().strip():
   box=t.get_window_extent(r)
   if box.width and box.height and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5):outside.append(t.get_text())
 audit['panels'][name]={'inches':list(fig.get_size_inches()),'outside_text':outside}
 for ext in ['svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=350)
 plt.close(fig)

src=C/'manuscript_revision_v34_results_provenance_20260916/source_data_steric'
s=pd.read_csv(src/'matched_anchor_selected_9rows.csv');t=pd.read_csv(src/'source_trace_summary.csv')
assert len(s)==9
for row in s.itertuples():
 for assay in ['nucleation','growth']:
  g=t[(t.identity==row.cap)&(t.assay==assay)&(t.rho_off==1)&(t.q_atomic_Zr_BDC==1)&(t.eta_S==row.etaS)]
  assert len(g)==32 and g.status.eq('target_reached').sum()==getattr(row,'target_'+assay)
  if assay=='growth' and getattr(row,'target_growth'):
   assert np.isclose(g[g.status.eq('target_reached')].model_time.median(),row.median_growth_passage_time)
s.to_csv(D/'steric_matched_nine_conditions.csv',index=False)
g=t[(t.rho_off==1)&(t.q_atomic_Zr_BDC==1)&t.identity.isin(['benzoate','4-OMe','4-tBu'])]
g.to_csv(D/'steric_matched_particle_records.csv',index=False)
caps=['benzoate','4-OMe','4-tBu'];cols=['#4477AA','#228833','#AA3377'];marks=['o','s','^']
def steric(ax,k):
 for i,(cap,col,marker) in enumerate(zip(caps,cols,marks)):
  a=s[s.cap.eq(cap)].sort_values('etaS');x=a.etaS.to_numpy()+(i-1)*.018
  if k<2:
   assay=['nucleation','growth'][k];y=a['target_'+assay].to_numpy()/32
   lo=a[assay+'_wilson95_low'].to_numpy();hi=a[assay+'_wilson95_high'].to_numpy()
   ax.errorbar(x,y,yerr=[y-lo,hi-y],color=col,marker=marker,ms=4,capsize=2,lw=1.1)
  else:ax.plot(a.etaS,a.median_growth_passage_time,color=col,marker=marker,ms=4,lw=1.1)
 ax.set_xticks([.1,.5,.9]);ax.set_xlim(.04,.96);ax.set_xlabel(r'$\eta_{\mathrm{S}}$')
 if k<2:ax.set_ylim(-.05,1.13);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Target-reaching fraction')
 else:
  ax.set_ylim(0,11);ax.set_ylabel('Median growth time\n(model-time units)')
  ax.text(.53,9.25,'4-tBu at 0.1:\nno successful growth',fontsize=7.5,color=cols[2],ha='center')
 ax.set_title(['Assembly to N = 30','Growth from 76 to 126','Successful growth time'][k],pad=12,fontsize=9.5)
for k in range(3):
 fig,ax=plt.subplots(figsize=(2.7,2.8));fig.subplots_adjust(left=.25,bottom=.20,right=.98,top=.80);steric(ax,k);save(fig,f'S61_{"ABC"[k]}')
fig,axs=plt.subplots(1,3,figsize=(7.05,3.20));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.72,wspace=.67)
for k,ax in enumerate(axs):
 steric(ax,k);pos=ax.get_position();fig.text(pos.x0-.055,.82,'ABC'[k],fontsize=13,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color=c,marker=m,label=l,lw=1) for c,m,l in zip(cols,marks,['Benzoate','4-Methoxybenzoate','4-tert-Butylbenzoate'])],ncol=3,loc='upper center',bbox_to_anchor=(.52,1.01),fontsize=9)
save(fig,'Figure_S61_steric')

records=[]
for rho in [10,30,50,100]:
 for f in sorted((I/'evidence/btb/pxrd').glob(f'btb_extend_rho{rho}_rep*_milestone_N02000.json')):
  m=json.loads(f.read_text());csv=f.with_suffix('.csv');assert hashlib.sha256(csv.read_bytes()).hexdigest()==m['pattern_sha256'];assert m['source']['observed_entities']==2000
  df=pd.read_csv(csv);records.append((rho,m,df));audit['pxrd'].append({'rho':rho,'replicate':m['source']['replicate'],'source_sha256':m['source']['sha256'],'pattern_sha256':m['pattern_sha256']})
  (D/f.name).write_bytes(f.read_bytes());(D/csv.name).write_bytes(csv.read_bytes())
assert len(records)==16
refs=pd.read_csv(C/'btb_extra_peak_diagnosis_20260907_v1/data/full_hk0_positions.csv')
refs.to_csv(D/'kgd_reference_positions.csv',index=False)
colors={10:'#4477AA',30:'#228833',50:'#AA3377',100:'#BB8800'}
def pxrd(ax,zoom=False):
 ax.axvspan(5.55,6.10,color='#E5C778',alpha=.26,lw=0,zorder=0)
 for k,rho in enumerate([10,30,50,100]):
  rr=[r for r in records if r[0]==rho];assert len(rr)==4
  offset=(3-k)*1.12;stack=[]
  for _,m,d in rr:
   angle=d.two_theta_deg.to_numpy();y=d.intensity_smoothed_normalized.to_numpy();stack.append(y)
   ax.plot(angle,y*.86+offset,color=colors[rho],lw=.65,alpha=.35)
   if m['shoulder_candidates']:
    ax.plot(angle,y*.86+offset,color=colors[rho],ls='--',lw=.95,alpha=.85)
    if zoom:
     for p in m['shoulder_candidates']:ax.plot(p['two_theta_deg'],p['height']*.86+offset,'o',ms=3,mfc='white',mec=colors[rho],mew=.9,zorder=6)
  ax.plot(angle,np.median(stack,axis=0)*.86+offset,color=colors[rho],lw=1.65)
  ax.text(6.47 if zoom else 12.45,offset+(.72 if zoom else .14),rf'{rho} ($n=4$)',ha='right',fontsize=10,color=colors[rho])
 if not zoom:
  for r in refs[refs.two_theta_deg<11].itertuples():
   ax.vlines(r.two_theta_deg,-.42,-.25,lw=1.15,color='black');ax.text(r.two_theta_deg,-.49,f'({r.h}{r.k}0)',ha='center',va='top',fontsize=8)
 ax.set(xlim=(4.65,6.55) if zoom else (3.5,12.7),ylim=(-.08 if zoom else -.87,4.48),yticks=[],xlabel=r'$2\theta$ (degrees, Cu K$\alpha$)',ylabel='Normalized intensity + offset')
 ax.set_title('N = 2,000',pad=8)
 if zoom:ax.set_xticks([4.8,5.2,5.6,6.0,6.4])
 else:ax.set_xticks([4,6,8,10,12])
for name,zoom in [('Figure6C_N2000',False),('S56B_N2000',True)]:
 fig,ax=plt.subplots(figsize=(6.9,3.25) if zoom else (3.35,2.55));fig.subplots_adjust(left=.11 if zoom else .17,right=.98,bottom=.20,top=.86);pxrd(ax,zoom);save(fig,name)

S='http://www.w3.org/2000/svg';ns={'s':S}
def svgread(p):return E.parse(str(p)).getroot()
def render(root,path,width=2400):
 data=E.tostring(root,xml_declaration=True,encoding='utf8');path.with_suffix('.svg').write_bytes(data)
 png=resvg_py.svg_to_bytes(svg_string=data.decode(),width=width,background='white',dpi=96);path.with_suffix('.png').write_bytes(png)
def isolate(root,group_id):
 rr=deepcopy(root);layer=rr.xpath('.//s:g[@id="layer1"]',namespaces=ns)[0]
 for e in list(layer):
  if e.get('id')!=group_id:layer.remove(e)
 return rr
def bounds(root):
 png=resvg_py.svg_to_bytes(svg_string=E.tostring(root).decode(),width=2100,dpi=96)
 im=Image.open(io.BytesIO(png)).convert('RGBA');box=im.getchannel('A').getbbox();return [v/10 for v in box]
# The author's PNG is the actual manuscript artwork. Its older editable SVG has
# extra off-canvas titles and renderer-dependent markers, so never regenerate
# unaffected panels from it. Keep the original raster at 1:1 and overlay vector C.
original=B/'Figure6_original.png';im=Image.open(original).convert('RGB');w,h=im.size
assert (w,h)==(2355,1239)
root=E.Element('{'+S+'}svg',nsmap={None:S,'xlink':'http://www.w3.org/1999/xlink'},viewBox=f'0 0 {w} {h}',width=str(w),height=str(h))
uri='data:image/png;base64,'+base64.b64encode(original.read_bytes()).decode()
def retained_image(parent):
 e=E.SubElement(parent,'{'+S+'}image',x='0',y='0',width=str(w),height=str(h));e.set('{http://www.w3.org/1999/xlink}href',uri)
retained_image(root)
E.SubElement(root,'{'+S+'}rect',x='1571',y='0',width=str(w-1571),height='588',fill='white')
panel=svgread(O/'Figure6C_N2000.svg');pw,ph=map(float,panel.get('viewBox').split()[2:]);sc=765/pw
group=E.SubElement(root,'{'+S+'}g',transform=f'translate(1590 0) scale({sc})')
for child in panel:group.append(deepcopy(child))
defs=E.SubElement(root,'{'+S+'}defs');clip=E.SubElement(defs,'{'+S+'}clipPath',id='original_C_letter');E.SubElement(clip,'{'+S+'}rect',x='1571',y='0',width='67',height='68')
E.SubElement(clip,'{'+S+'}rect',x='1535',y='195',width='60',height='115')
letter=E.SubElement(root,'{'+S+'}g',attrib={'clip-path':'url(#original_C_letter)'});retained_image(letter)
render(root,O/'Figure6_N2000',width=w)
newim=Image.open(O/'Figure6_N2000.png').convert('RGB')
# Use the original pixel canvas for the final raster to avoid any color-conversion
# round-off on embedded microscopy introduced by the SVG rasterizer.
final=im.copy();final.paste(newim.crop((1571,0,2355,588)),(1571,0));final.paste(im.crop((1535,195,1595,310)),(1535,195));final.save(O/'Figure6_N2000.png')
delta=np.asarray(im)!=np.asarray(final)
delta[:588,1571:]=False
assert not delta.any(),'Unexpected pixel change outside authorized C panel'
audit['figure6_other_pixels_identical']=True
audit['figure6_new_C_bounds_pixels']=[1571,0,2355,588]
audit['figure6_source_master_note']='Unchanged panels retained exactly from manuscript raster; replacement C remains vector in composite SVG.'
# SI S56: retain the original vector A panel, replace only the old PXRD B panel.
a=svgread(C/'manuscript_revision_v29_figure_consolidation_20260910/figures/panels/7C_retained.svg');b=svgread(O/'S56B_N2000.svg')
aw,ah=map(float,a.get('viewBox').split()[2:]);bw,bh=map(float,b.get('viewBox').split()[2:]);width=500.;asc=width/aw;bsc=width/bw;gap=23.;height=ah*asc+gap+bh*bsc
out=E.Element('{'+S+'}svg',nsmap={None:S},viewBox=f'0 0 {width} {height}',width=str(width),height=str(height))
for letter,rr,sc,y in [('A',a,asc,0),('B',b,bsc,ah*asc+gap)]:
 gg=E.SubElement(out,'{'+S+'}g',transform=f'translate(0 {y}) scale({sc})')
 # Namespace IDs to avoid collisions with independently exported matplotlib panels.
 ids={e.get('id'):letter+'_'+e.get('id') for e in rr.iter() if e.get('id')}
 for e in rr.iter():
  for attr,val in list(e.attrib.items()):
   if attr=='id':e.set(attr,ids[val]);continue
   for oldid,newid in ids.items():
    val=val.replace('url(#'+oldid+')','url(#'+newid+')')
    if val=='#'+oldid:val='#'+newid
   e.set(attr,val)
 for child in rr:gg.append(child)
 tt=E.SubElement(out,'{'+S+'}text',x='3',y=str(y+12),style='font-family:Arial;font-size:13px;font-weight:bold');tt.text=letter
render(out,O/'Figure_S56_N2000',width=1900)
(B/'FIGURE_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print('Figures complete; verified 16 PXRD records and nine steric conditions against raw trajectories.')
