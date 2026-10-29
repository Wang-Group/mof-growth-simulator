"""Descriptive paired readout; all outcomes counted, no imputed censored endpoints."""
import ast,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def output(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def eligible(p):return {r['key']:r for r in csvrows(p) if r['interior']=='True' and r['ambiguous']=='False'}
def missing(r):return r['missing']=='True'
def frac(num,den):return num/den if den else None
def main():
    m=read(B/'manifest.json');records=[];outcomes=[]
    for t in m['traces']:
        d=B/'traces'/t['id'];rp=d/'result.json'
        result=read(rp) if rp.exists() else {}
        base=dict(id=t['id'],replicate=t['replicate'],delta=t['delta'],rho=t['rho'],pool=t['pool'])
        outcomes.append(dict(**base,status=result.get('status','not_completed'),valid=result.get('valid'),N=result.get('final_N'),tau=result.get('tau'),events=result.get('events')))
        initial=d/'filling_analysis/initial'
        if not (initial/'summary.json').exists():continue
        for p in (d/'filling_analysis').glob('*/summary.json'):
            tag=p.parent.name
            if tag not in ['initial']+[f'tau{x:g}' for x in m['times']]:continue
            s=read(p)
            for tol in (.5,1.,2.):
                a=next(r['result'] for r in s['analysis'] if r['tolerance']==tol)
                before=eligible(initial/f'linker_tol{tol:g}.csv');after=eligible(p.parent/f'linker_tol{tol:g}.csv')
                shared=before.keys()&after.keys();old={k for k in shared if missing(before[k])};new={k for k in shared if missing(after[k])}
                center=np.array(a['reference']['center_A']);obs=csvrows(p.parent/f'observed_node_tol{tol:g}.csv');radius=max(float(r['radius_from_center_A']) for r in obs)
                regional=defaultdict(list)
                for r in after.values():
                    dist=float(np.linalg.norm(np.array(ast.literal_eval(r['xyz_A']))-center))
                    regional['inner' if dist<radius/2 else 'outer'].append(missing(r))
                records.append(dict(**base,snapshot=tag,tau=s['tau'],N=s['N'],tolerance=tol,
                    eligible=len(after),vacancy_fraction=frac(sum(missing(r) for r in after.values()),len(after)),
                    singly_attached_fraction=frac(a.get('present_singly_attached_interior_BDC',0),a['occupied_interior_linkers']),
                    node_registration=frac(a['registered_nodes'],a['node_count']),shared_eligible=len(shared),
                    initial_vacancies_on_shared=len(old),retained=len(old&new),filled=len(old-new),newly_empty=len(new-old),
                    paired_vacancy_change_on_shared=frac(len(new)-len(old),len(shared)),
                    inner_eligible=len(regional['inner']),outer_eligible=len(regional['outer']),
                    inner_vacancy_fraction=frac(sum(regional['inner']),len(regional['inner'])),outer_vacancy_fraction=frac(sum(regional['outer']),len(regional['outer']))))
    out=B/'summary';out.mkdir(exist_ok=True);output(out/'all_outcomes.csv',outcomes);output(out/'particle_snapshots.csv',records)
    groups=defaultdict(list)
    for r in records:groups[(r['delta'],r['rho'],r['pool'],r['snapshot'],r['tolerance'])].append(r)
    stats=[]
    for key,rs in sorted(groups.items()):
        row=dict(zip(('delta','rho','pool','snapshot','tolerance'),key));row['observed_particles']=len(rs);row['planned_particles']=32
        for name in ('N','vacancy_fraction','singly_attached_fraction','paired_vacancy_change_on_shared','inner_vacancy_fraction','outer_vacancy_fraction'):
            # At least 20 interior sites, and at least 10 per radial region.
            vals=[r[name] for r in rs if r[name] is not None and (name=='N' or r['eligible']>=20) and (not name.startswith(('inner_','outer_')) or (r['inner_eligible']>=10 and r['outer_eligible']>=10)) and (name!='paired_vacancy_change_on_shared' or r['shared_eligible']>=20)]
            row[name+'_n']=len(vals)
            for q,label in [(25,'q25'),(50,'median'),(75,'q75')]:row[name+'_'+label]=float(np.percentile(vals,q)) if vals else None
        stats.append(row)
    output(out/'condition_snapshots.csv',stats)
    # Added penalty contrast at identical rho, pool and time, paired by source particle.
    contrasts=[];rng=np.random.default_rng(20260916)
    for rho in (1,3):
        for pool in ('fixed','finite250','finite100'):
            for tau in m['times']:
                snap=f'tau{tau:g}'
                for tol in (.5,1.,2.):
                    arms=[{r['replicate']:r for r in groups.get((delta,rho,pool,snap,tol),[])} for delta in (0,2)]
                    for quantity in ('vacancy_fraction','paired_vacancy_change_on_shared'):
                        ids=[i for i in arms[0].keys()&arms[1].keys() if all(a[i][quantity] is not None and a[i]['eligible']>=20 and (quantity!='paired_vacancy_change_on_shared' or a[i]['shared_eligible']>=20) for a in arms)]
                        vals=np.array([arms[1][i][quantity]-arms[0][i][quantity] for i in sorted(ids)])
                        ci=np.percentile(np.median(rng.choice(vals,(10000,len(vals))),axis=1),[2.5,97.5]).tolist() if len(vals)>=2 else [None,None]
                        contrasts.append(dict(rho=rho,pool=pool,tau=tau,tolerance=tol,quantity=quantity,contrast='Delta2 minus Delta0',paired_n=len(vals),planned_n=32,median=float(np.median(vals)) if len(vals) else None,bootstrap95_low=ci[0],bootstrap95_high=ci[1]))
    output(out/'paired_penalty_contrasts.csv',contrasts)
    (out/'README.md').write_text('All fractions are in units of 1, not percent. Counts and missing observations are retained. Paired intervals resample starting particles, not lattice sites. These are descriptive pointwise intervals, not multiplicity-adjusted significance tests. Equal-time comparisons exclude unobserved times rather than imputing censored results. Every branch shares its initial structure with the other conditions for that replicate. The cohort tests established-crystal response only.\n')
    print(json.dumps(dict(outcomes=len(outcomes),snapshot_rows=len(records),summary=str(out))))
if __name__=='__main__':main()
