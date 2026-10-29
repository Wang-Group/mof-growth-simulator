"""Read-only condition-matched synthesis for manuscript v34; no trajectories run."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

B=Path(__file__).resolve().parent
C=B.parent
OLD=C/'defect_322_analysis_20260914_v1'
NEW=C/'defect_N1000_completion_20260916_v1'
hashes=json.loads((NEW/'REMOTE_SHA256.json').read_text())
checked=[]
def read_new(rel):
    p=NEW/'remote'/rel
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==hashes[rel]['sha256'],rel
    checked.append(rel)
    return json.loads(data)
def stats(x):
    a=np.asarray(x,dtype=float)
    a=a[np.isfinite(a)]
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,
            'q25':float(np.quantile(a,.25)) if len(a) else None,
            'q75':float(np.quantile(a,.75)) if len(a) else None,
            'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None}

m=read_new('manifest.json')
rows=[]
for t in m['traces']:
    prefix='traces/'+t['id']+'/'
    r=read_new(prefix+'result.json')
    ss=read_new(prefix+'filling_analysis/terminal/sensitivity.json')
    chosen=[v for v in ss if v['anchor']==0 and v['alpha']==1.05 and v['boundary']==.35 and v['tolerance']==2]
    assert len(chosen)==1
    z=chosen[0]['result']
    s=z['interior_linker_positions']; present=z['occupied_interior_linkers']; dang=z['present_singly_attached_interior_BDC']
    if isinstance(dang,dict):raise TypeError(dang)
    row={k:t[k] for k in ['id','condition','replicate','delta','rho0','mode']}
    row.update(status=r['status'],N=r['final_N'],events=r['events'],events_added=r['continuation']['events_added'],
      tau=r['tau'],tau_added=r['tau']-t['source_tau'],wall=r['wall_seconds'],sites=s,present=present,
      missing=z['missing_interior_linkers'],dangling=dang,D=r['final']['D_count'],
      vacancy_pct=100*z['missing_interior_linkers']/s if s>=20 else np.nan,
      dangling_pct=100*dang/present if present and s>=20 else np.nan,
      coverage=z['registered_nodes']/z['node_count'])
    for key in ['new_D_states','D_states_left','remove_bond_accepted','close_internal_accepted','add_linker_accepted','open_boundary_fragment_losses']:
        row[key]=r['transition_counts'].get(key,0)
    alt=[v['result'] for v in ss if v['anchor']==0 and v['boundary']==.35 and
         v['alpha']==1.05 and v['tolerance'] in [.5,1,2]]
    assert len(alt)==3
    row['positive_all_three_tolerances']=all(v['missing_interior_linkers']>0 for v in alt)
    row['eligible_all_three_tolerances']=all(v['interior_linker_positions']>=20 for v in alt)
    row['positive_all_13_variants']=all(v['result']['missing_interior_linkers']>0 for v in ss)
    row['eligible_all_13_variants']=all(v['result']['interior_linker_positions']>=20 for v in ss)
    row['minimum_sites_across_13_variants']=min(v['result']['interior_linker_positions'] for v in ss)
    rows.append(row)
df=pd.DataFrame(rows)
df.to_csv(B/'followon_condition_matched_ledger.csv',index=False)
groups=[]
for keys,g in df.groupby(['condition','status']):
    eligible=g[g.sites>=20]
    groups.append(dict(condition=keys[0],status=keys[1],n=len(g),eligible_n=len(eligible),
      vacancy_positive=int((eligible.missing>0).sum()),dangling_positive=int((eligible.dangling>0).sum()),
      full_node_registration=int((g.coverage==1).sum()),positive_all_three_tolerances=int(g.positive_all_three_tolerances.sum()),
      eligible_all_three_tolerances=int(g.eligible_all_three_tolerances.sum()),
      positive_all_13_variants=int(g.positive_all_13_variants.sum()),
      eligible_all_13_variants=int(g.eligible_all_13_variants.sum()),
      minimum_sites_across_13_variants=int(g.minimum_sites_across_13_variants.min()),
      **{k:stats(g[k]) for k in ['N','events','events_added','tau','tau_added','sites','vacancy_pct','dangling_pct','D','new_D_states','D_states_left','remove_bond_accepted','close_internal_accepted','add_linker_accepted','coverage']}))
(B/'followon_groups.json').write_text(json.dumps(groups,indent=2))
flat=[]
for g in groups:
    z={k:v for k,v in g.items() if not isinstance(v,dict)}
    for k,v in g.items():
        if isinstance(v,dict):
            z.update({k+'_'+a:b for a,b in v.items()})
    flat.append(z)
pd.DataFrame(flat).to_csv(B/'followon_condition_summary.csv',index=False)
sourceindex={}
for t in m['traces']:
    key=t['condition']; prefix='traces/'+t['id']+'/'
    sourceindex.setdefault(key,[]).append({'id':t['id'],'source_original_result':t['source_result'],
     'source_original_result_sha256':t['source_result_sha256'],
     'followon_result':hashes[prefix+'result.json'],
     'sensitivity_result':hashes[prefix+'filling_analysis/terminal/sensitivity.json']})
(B/'condition_source_paths.json').write_text(json.dumps(sourceindex,indent=2))
old=pd.read_csv(OLD/'diagnostic_counts.csv')
reference=old[(old.tol==2)&(old.alpha==1.05)&(old.linker_sites>=20)]
og=[]
for keys,g in reference.groupby(['cohort','delta','rho','mode']):
    og.append(dict(cohort=keys[0],delta=int(keys[1]),rho0=float(keys[2]),mode=keys[3],n=len(g),
      vacancy_positive=int((g.missing_BDC>0).sum()),dangling_positive=int((g.dangling>0).sum()),
      statuses=g.status.value_counts().to_dict(),
      **{k:stats(g[k]) for k in ['N','events','tau','vacancy_pct','dangling_pct']}))
(B/'frozen322_groups_recomputed.json').write_text(json.dumps(og,indent=2))
orig=json.loads((OLD/'original_1152.json').read_text())
od=pd.DataFrame([{**r['trace'],**{k:r[k] for k in ['N','events','tau','status','valid']}} for r in orig])
assert len(od)==1152 and od.id.nunique()==1152 and od.valid.all()
originalgroups=[]
for c,g in od.groupby('condition'):
    originalgroups.append(dict(condition=c,n=len(g),statuses=g.status.value_counts().to_dict(),
      **{k:stats(g[k]) for k in ['N','events','tau']}))
(B/'original1152_growth_groups.json').write_text(json.dumps(originalgroups,indent=2))
oldref=old[(old.tol==2)&(old.alpha==1.05)]
old_follow=oldref[oldref.cohort=='size_followon']
new_follow=df[df.status=='size_target_reached']
overlap=old_follow.merge(new_follow,on='id',suffixes=('_old','_new'))
assert len(overlap)==113 and (overlap.N_old==overlap.N_new).all()
assert (overlap.missing_BDC==overlap.missing).all() and (overlap.linker_sites==overlap.sites).all()
original=oldref[oldref.cohort=='original_endpoint']
assert len(original)==209
catalog=dict(original_N_ge1000=209,new_size_target=142,unique_N_ge1000=351,
 old_size_followon=113,unchanged_overlapping_endpoints=113,
 newly_added_size_endpoints=29,new_condition_additions=new_follow[~new_follow.id.isin(old_follow.id)].groupby('condition').size().astype(int).to_dict(),
 eligible20_original=int((original.linker_sites>=20).sum()),eligible20_followon=int((new_follow.sites>=20).sum()),
 vacancy_positive_original=int((original.missing_BDC>0).sum()),vacancy_positive_followon=int((new_follow.missing>0).sum()),
 dangling_positive_original=int((original.dangling>0).sum()),dangling_positive_followon=int((new_follow.dangling>0).sum()),
 warning='Catalog only: original and size-target outcomes are not a common-endpoint defect yield.')
(B/'current_Nge1000_catalog.json').write_text(json.dumps(catalog,indent=2))
(B/'validation.json').write_text(json.dumps({'hash_verified_remote_files':len(checked),'followon_n':len(df),'status':df.status.value_counts().to_dict(),'source_files':{
str(p.relative_to(C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'diagnostic_counts.csv',OLD/'original_1152.json',NEW/'REMOTE_SHA256.json']}}
,indent=2))
print(json.dumps(catalog,indent=2))
