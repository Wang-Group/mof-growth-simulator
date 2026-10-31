"""Static cap-packing/contact diagnostic; NOT a growth simulation or rate model.

No force-field energy enters rates. No engine is imported or modified. Run this
small diagnostic on 5090 after preparing the input folder locally.
"""
from pathlib import Path
import json,hashlib,csv,argparse
import numpy as np

def read_mol(p):
    lines=p.read_text().splitlines();n=int(lines[3][:3]);nb=int(lines[3][3:6])
    xyz=np.array([[float(v) for v in line.split()[:3]] for line in lines[4:4+n]])
    elements=[line.split()[3] for line in lines[4:4+n]]
    edges=[(int(line[:3])-1,int(line[3:6])-1) for line in lines[4+n:4+n+nb]]
    return elements,xyz,edges

def anchors(elements,edges):
    adj=[[] for _ in elements]
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    out=[]
    for c,e in enumerate(elements):
        oo=[a for a in adj[c] if elements[a]=='O']
        if e=='C' and len(oo)==2:out.append([c,*sorted(oo)])
    return out,adj

def proper_fit(moving,target):
    a=moving.mean(0);b=target.mean(0)
    U,_,Vt=np.linalg.svd((moving-a).T@(target-b))
    D=np.eye(3);D[-1,-1]=np.sign(np.linalg.det(U@Vt))
    R=U@D@Vt
    assert np.linalg.det(R)>0.999999
    return R,b-a@R,float(np.sqrt(np.mean(np.sum(((moving-a)@R-(target-b))**2,axis=1))))

def clashes(a,b,cutoff=2.0):
    if len(a)==0 or len(b)==0:return 0
    return int(np.sum(np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)<cutoff**2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    H=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lib=json.loads((H/'molecule_library/molecule_library_index.json').read_text())
    elems,xyz,edges=read_mol(H/'framework_inputs/Zr6_AA_reference.mol');sites,adj=anchors(elems,edges)
    assert len(sites)==12
    # Separate complete organic cap fragments from the inorganic Zr6O8 core.
    capgroups=[]
    for anchor in sites:
        found=set();stack=[anchor[0]]
        while stack:
            i=stack.pop()
            if i in found or elems[i]=='Zr':continue
            found.add(i);stack.extend(j for j in adj[i] if elems[j]!='Zr')
        capgroups.append(found)
    assert all(not(a&b) for i,a in enumerate(capgroups) for b in capgroups[i+1:])
    coreidx=sorted(set(range(len(elems)))-set.union(*capgroups))
    assert [elems[i] for i in coreidx].count('Zr')==6
    core=xyz[coreidx]
    be,bx,bedges=read_mol(H/'framework_inputs/BDC_explicit_H.mol');ba,_=anchors(be,bedges)
    assert len(ba)==2 and be.count('H')==4
    records=[];detailed=[]
    for identity in lib['identities']:
        candidates=[];fiterrors=[]
        ind=identity['indices'];mi=[ind['Ccarboxylate'],ind['O1'],ind['O2']]
        for target in sites:
            choices=[]
            for conf in identity['conformers']:
                ce,cx,_=read_mol(H/'molecule_library'/conf['mol_file'])
                for flip in [False,True]:
                    ai=[mi[0],mi[2],mi[1]] if flip else mi
                    rot,shift,err=proper_fit(cx[ai],xyz[target]);placed=cx@rot+shift
                    choices.append((placed,ce,conf['conformer'],flip,err))
            candidates.append(choices)
        # Packing search is an initialization diagnostic, not a thermodynamic ensemble.
        # Multiple insertion orders expose order dependence; incompatible caps remain flagged.
        best=None;rng=np.random.default_rng(20260911)
        for trial in range(32):
            order=np.arange(12) if trial==0 else rng.permutation(12)
            chosen={}
            for site in order:
                environment=np.vstack([core,*[c[0] for c in chosen.values()]])
                scored=[(clashes(c[0],environment),i) for i,c in enumerate(candidates[site])]
                _,k=min(scored);chosen[int(site)]=candidates[site][k]
            cc=sum(clashes(chosen[i][0],chosen[j][0]) for i in range(12) for j in range(i+1,12))
            ck=sum(clashes(chosen[i][0],core) for i in range(12))
            score=(cc+ck,trial)
            if best is None or score<best[0]:best=(score,chosen,cc,ck)
        score,chosen,cc,ck=best
        ee=[elems[i] for i in coreidx];xx=[core];groups=[-1]*len(core)
        for i in range(12):ee+=chosen[i][1];xx.append(chosen[i][0]);groups += [i]*len(chosen[i][0])
        xx=np.vstack(xx)
        name=identity['identity'].replace('-','_')
        (out/f'{name}_candidate_capped_node.xyz').write_text(str(len(xx))+'\nCANDIDATE ONLY; rigid cap packing; inspect clash report\n'+''.join(f'{e} {x:.6f} {y:.6f} {z:.6f}\n' for e,(x,y,z) in zip(ee,xx)))
        (out/f'{name}_atom_groups.json').write_text(json.dumps({'groups':groups,'minus_one':'inorganic core','sites':sites,'coordinates':'candidate, not optimized bound structure'},indent=2))
        contact_trials=[]
        for site,target in enumerate(sites):
            other=np.vstack([core,*[chosen[i][0] for i in range(12) if i!=site]])
            for ai0 in ba:
                for flip in [False,True]:
                    ai=[ai0[0],ai0[2],ai0[1]] if flip else ai0
                    rot,shift,err=proper_fit(bx[ai],xyz[target]);incoming=bx@rot+shift
                    ncl=clashes(incoming,other)
                    contact_trials.append({'site':site,'bdc_anchor':ai0,'flip':flip,'contact_count':ncl,'anchor_rmsd_A':err})
        rr={'identity':identity['identity'],'cap_cap_contact_pairs':cc,'cap_core_contact_pairs':ck,
            'selected_packing_trial':score[1],'all_12_caps_feasible_at_2A':cc+ck==0,
            'replacement_orientations_tested':len(contact_trials),
            'replacement_orientations_without_contacts':sum(r['contact_count']==0 for r in contact_trials),
            'max_cap_anchor_rmsd_A':max(chosen[i][4] for i in range(12)),
            'status':'STATIC_GEOMETRY_ONLY_NOT_GROWTH_OR_RATE'}
        records.append(rr);detailed.append({'summary':rr,'replacement_trials':contact_trials,
                                          'selected_caps':[{'site':i,'conformer':chosen[i][2],'flip':chosen[i][3]} for i in range(12)]})
    with (out/'geometry_preflight.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'geometry_preflight.json').write_text(json.dumps(detailed,indent=2))
    (out/'README.md').write_text('# Static cap-geometry diagnostic\n\nNot a KMC result or crystal-size prediction. Twelve caps are a stress test, not an assumed measured coverage. Molecules are fitted rigidly; cap-core coordination is not optimized. Zero allowed sampled orientations does not establish zero physical growth rate. A valid reversible cap/solvent model and reviewed cap occupancy are required before growth.\n')
    manifest=[{'path':p.relative_to(out).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.is_file()]
    (out/'REMOTE_SHA256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
