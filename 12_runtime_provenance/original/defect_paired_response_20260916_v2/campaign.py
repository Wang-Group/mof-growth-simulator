"""Paired parameter interventions on frozen molecular states; 5090 only."""
import argparse, collections, gzip, hashlib, importlib.util, json, os
from pathlib import Path
import random, signal, socket, subprocess, sys, time, traceback

B=Path(__file__).resolve().parent
ROOT=B.parent
PARENT=ROOT/'defect_50000steps_production_20260913_v1'
SOURCE=ROOT/'defect_N1000_followon_20260914_v1'
STOP=False
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,allow_nan=False)); os.replace(tmp,p)
def load():
    m=read(B/'manifest.json')
    for name,digest in m['files'].items(): assert sha(B/name)==digest,name
    assert sha(PARENT/'manifest.json')==m['parent_sha256']
    return m
def setup():
    sys.path.insert(0,str(PARENT))
    import campaign_io
    base,pm=campaign_io.load_campaign(PARENT/'manifest.json')
    engine,root,acetate,provenance=campaign_io.setup(base,pm)
    return engine,root,provenance,campaign_io
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare():
    assert not (B/'manifest.json').exists()
    sm=read(SOURCE/'manifest.json'); seeds=[]
    for t in sorted(sm['traces'],key=lambda t:t['replicate']):
        if t['condition']!='d00_rho3p0_fixed': continue
        d=SOURCE/'traces'/t['id']; r=read(d/'result.json'); cp=d/'snapshots/terminal/checkpoint.pkl'
        assert r['final_N']==1000 and r['status']=='size_target_reached'
        seeds.append(dict(replicate=t['replicate'],id=t['id'],checkpoint=str(cp),checkpoint_sha256=sha(cp),result=str(d/'result.json'),result_sha256=sha(d/'result.json'),source_tau=r['tau'],source_events=r['events']))
    assert len(seeds)==32 and {s['replicate'] for s in seeds}==set(range(32))
    traces=[]
    # Interleave conditions so queue order does not prioritize one treatment.
    for s in seeds:
        for delta in (0,2):
            for rho in (1,3):
                for pool,free in [('fixed',[100,600]),('finite250',[250,1500]),('finite100',[100,600])]:
                    traces.append(dict(id=f"d{delta}_rho{rho}_{pool}_r{s['replicate']:02d}",replicate=s['replicate'],delta=delta,rho=rho,pool=pool,mode='fixed' if pool=='fixed' else 'finite',free=free,seed=2026091600+s['replicate'],source=s))
    m=dict(schema='paired-defect-response-2',parent=str(PARENT),parent_sha256=sha(PARENT/'manifest.json'),source_manifest_sha256=sha(SOURCE/'manifest.json'),files={n:sha(B/n) for n in ['campaign.py','PLAN.md','summarize.py','extended_checks.py']},traces=traces,times=[.25,.5,1.,2.,5.,10.,20.],primary_times=[.5,1.,2.],pilot_status=str(ROOT/'defect_paired_response_20260916_v1/status.json'),wall_seconds=7200,max_events=200000,safety_N=2500,workers=64,analysis_python=read(PARENT/'manifest.json')['analysis']['python'])
    save(B/'manifest.json',m); print(json.dumps(dict(traces=len(traces),manifest_sha256=sha(B/'manifest.json'))))
def signature(sim):
    import numpy as np
    from vacancy_counter import assembly_coordinates
    h=hashlib.sha256()
    for x in assembly_coordinates(sim.model.assembly): h.update(np.asarray(x).tobytes())
    h.update(json.dumps(sim.model.intermediate_record(),sort_keys=True).encode())
    return h.hexdigest()
def initialize(m,t):
    import numpy as np
    engine,root,oldprov,io=setup()
    from trajectory import Simulation
    from pool_clock import PoolClock
    from solvent_channel import SiteCompetition
    s=t['source'];assert sha(s['checkpoint'])==s['checkpoint_sha256'] and sha(s['result'])==s['result_sha256']
    sim=Simulation.restore(Path(s['checkpoint']),engine,oldprov)
    assert len(sim.model.assembly.entities)==1000 and sim.model.delta==0 and sim.model.ledger.rho0==3
    initial_signature=signature(sim); model=sim.model
    counts=dict(collections.Counter(e.entity_type for e in model.assembly.entities.items))
    model.ledger=PoolClock(dict(zip(('Zr','Ligand'),t['free'])),counts,t['mode'],'retain_oligomers',t['rho'],1.,.5)
    model.delta=float(t['delta']);model.bath=SiteCompetition.from_rho_fraction(t['rho'],.5)
    model.transition_counts=collections.Counter();model.last_transition=None;model._close_candidates=None
    # New, recorded experiment clock/random stream, NOT a deterministic continuation.
    random.seed(t['seed']);np.random.seed(t['seed']);model.np_rng=np.random.default_rng(t['seed']^0xACA7E)
    sim.events_rng=random.Random(t['seed']^0xFD7);sim.event_count=0;sim.tau=0.;sim.time_compensation=0.;sim.pending_wait=None
    sim.maximum_N=1000;sim.first_passage={};sim.observations=[]
    from vacancy_counter import VacancyCohort
    sim.cohorts={d:VacancyCohort() for d in (1,2)}
    model.check_state();assert signature(sim)==initial_signature
    prov=dict(parent=oldprov,experiment_manifest_sha256=sha(B/'manifest.json'),source_checkpoint_sha256=s['checkpoint_sha256'],trace_id=t['id'],initial_structure_signature=initial_signature)
    return sim,prov,io
def worker(index,test=False):
    import resource
    m=load();t=m['traces'][index];out=B/('tests' if test else 'traces')/t['id'];out.mkdir(parents=True,exist_ok=False)
    sim,prov,io=initialize(m,t);wrap=module('frozen_snapshot',PARENT/'run_production_trace.py')
    times=[.001] if test else m['times'];wall=60 if test else m['wall_seconds'];event_cap=1000 if test else m['max_events']
    started=time.monotonic();snapshots=[];status='running';error=None
    def snap(tag):
        snapshots.append(wrap.snapshot(sim,out,tag,prov))
    def progress(): save(out/'progress.json',dict(N=len(sim.model.assembly.entities),events=sim.event_count,tau=sim.tau,elapsed_seconds=time.monotonic()-started,pid=os.getpid(),snapshots=snapshots))
    def stop(sig,frame):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save(out/'input.json',t);next_time=0;last=started
    try:
        with gzip.open(out/'entity_counts.jsonl.gz','wt') as curve:
            snap('initial');progress()
            while True:
                n=len(sim.model.assembly.entities)
                if next_time==len(times):status='time_target_reached';break
                if STOP:status='resource_stop';break
                if time.monotonic()-started>=wall:status='wall_time_cap';break
                if sim.event_count>=event_cap:status='event_cap';break
                if n>=m['safety_N']:status='size_safety_cap';break
                event=sim.advance(times[next_time])
                if event=='model_time_cap':
                    assert abs(sim.tau-times[next_time])<1e-12
                    snap(f"tau{times[next_time]:g}");next_time+=1
                elif event!='event':status=event;break
                for target in (1250,1500):
                    if len(sim.model.assembly.entities)>=target and str(target) not in sim.first_passage:
                        sim.first_passage[str(target)]=dict(N=len(sim.model.assembly.entities),events=sim.event_count,tau=sim.tau);snap(f'N{target}')
                if sim.event_count%250==0 or event=='model_time_cap':curve.write(json.dumps(io.lightweight(sim))+'\n');curve.flush()
                if time.monotonic()-last>=30:progress();last=time.monotonic()
    except Exception:error=traceback.format_exc();status='error'
    try:snap('terminal')
    except Exception:error=(error or '')+traceback.format_exc();status='error'
    save(out/'result.json',dict(status=status,valid=status!='error',error=error,censored=status!='time_target_reached',final_N=len(sim.model.assembly.entities),events=sim.event_count,tau=sim.tau,wall_seconds=time.monotonic()-started,max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,first_passage=sim.first_passage,snapshots=snapshots,transition_counts=dict(sim.model.transition_counts),provenance=prov,final=io.lightweight(sim)))
    assert sha(t['source']['checkpoint'])==t['source']['checkpoint_sha256'];progress()
    if error:raise RuntimeError(error)
    print(json.dumps(dict(id=t['id'],status=status,N=len(sim.model.assembly.entities),tau=sim.tau)))
def analyze_trace(out):
    import pickle,csv,importlib
    m=load()
    sys.path.insert(0,str(PARENT/'frozen/analysis'))
    from central_filling import CentralLattice,FilledEnvelope,analyze
    engine,root,oldprov,io=setup()
    from vacancy_counter import assembly_coordinates
    for n in ('numeric','multiarray','umath','_multiarray_umath'):sys.modules.setdefault('numpy._core.'+n,importlib.import_module('numpy.core.'+n))
    with (root/'framework_inputs/UiO66_N76_reviewed.pkl').open('rb') as f:ref=pickle.load(f)
    with (out/'snapshots/initial/checkpoint.pkl').open('rb') as f:initial=pickle.load(f)
    initial_a=initial['model_fields']['assembly'];lattice=CentralLattice.from_assemblies(ref,initial_a)
    dest=out/'filling_analysis';dest.mkdir(exist_ok=False);records=[]
    for p in sorted((out/'snapshots').glob('*/checkpoint.pkl')):
        digest=sha(p)
        with p.open('rb') as f:x=pickle.load(f)
        assert x['provenance']['experiment_manifest_sha256']==sha(B/'manifest.json')
        assert x['provenance']['source_checkpoint_sha256']==initial['provenance']['source_checkpoint_sha256']
        nodes,links,degrees=assembly_coordinates(x['model_fields']['assembly']);lattice.enumerate(nodes);env=FilledEnvelope(nodes,lattice.distance,alpha=1.05)
        d=dest/p.parent.name;d.mkdir();rr=[]
        for tol in (.5,1.,2.):
            r=analyze(lattice,nodes,links,degrees,tolerance=tol,boundary=.35,detail=True,envelope=env)
            for kind in ('node','linker','observed_node','observed_linker'):
                rows=r.pop(kind+'_rows')
                if rows:
                    with (d/f'{kind}_tol{tol:g}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            rr.append(dict(tolerance=tol,result=r))
        save(d/'summary.json',dict(tau=x['simulation_fields']['tau'],N=len(x['model_fields']['assembly'].entities),analysis=rr,initial_anchor=lattice.metadata,checkpoint_sha256=digest))
        assert sha(p)==digest;records.append(p.parent.name)
    save(dest/'summary.json',dict(status='completed',snapshots=records,fixed_initial_anchor=True));print(json.dumps(dict(analyzed=str(out),snapshots=len(records))))
def tests():
    m=load();env=environment();(B/'test_logs').mkdir(exist_ok=False)
    from concurrent.futures import ThreadPoolExecutor
    def one(i):
        with (B/'test_logs'/f'{i}.log').open('xb') as f:
            p=subprocess.run([sys.executable,str(B/'campaign.py'),'worker','--index',str(i),'--test'],stdout=f,stderr=subprocess.STDOUT,env=env)
        return i,p.returncode
    with ThreadPoolExecutor(max_workers=4) as pool:outcomes=dict(pool.map(one,range(12)))
    assert all(v==0 for v in outcomes.values()),outcomes
    signatures=[]
    for i in range(12):
        r=read(B/'tests'/m['traces'][i]['id']/'result.json');assert r['status']=='time_target_reached' and abs(r['tau']-.001)<1e-12
        signatures.append(r['provenance']['initial_structure_signature'])
    assert len(set(signatures))==1
    # Exact event replay across checkpoint save/restore; same molecular code.
    sim,prov,_=initialize(m,m['traces'][0]);engine=sim.model.engine
    for _ in range(20):sim.advance()
    sim.checkpoint(B/'test_logs/replay.pkl',prov)
    for _ in range(20):sim.advance()
    expected=(signature(sim),sim.tau,sim.event_count)
    from trajectory import Simulation
    restored=Simulation.restore(B/'test_logs/replay.pkl',engine,prov)
    for _ in range(20):restored.advance()
    assert (signature(restored),restored.tau,restored.event_count)==expected
    target=B/'tests'/m['traces'][0]['id']
    subprocess.run([m['analysis_python'],str(B/'campaign.py'),'analyze','--trace',str(target)],check=True,env=env)
    save(B/'TEST_RESULTS.json',dict(passed=True,manifest_sha256=sha(B/'manifest.json'),twelve_conditions=outcomes,identical_initial_structure=True,exact_clock_boundary=True,checkpoint_replay=True,measurement_pipeline=True))
    print('All branch tests passed')
def environment():return dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0')
def memory():return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2
def supervise():
    import fcntl,shutil
    m=load();assert read(B/'TEST_RESULTS.json')['manifest_sha256']==sha(B/'manifest.json') and read(B/'TEST_RESULTS.json')['passed']
    lock=(B/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (B/'launch.json').exists();save(B/'launch.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),manifest_sha256=sha(B/'manifest.json')))
    os.nice(10);(B/'logs').mkdir(exist_ok=False);env=environment();active={};analyses={};aq=[];done={};adone={};next_i=0;started=time.monotonic()
    while next_i<len(m['traces']) or active or analyses or aq:
        for tid,(p,f,st) in list(active.items()):
            rc=p.poll()
            if rc is None:
                if time.monotonic()-st>m['wall_seconds']+300:p.terminate()
                if time.monotonic()-st>m['wall_seconds']+600:p.kill()
                continue
            f.close();rp=B/'traces'/tid/'result.json'
            done[tid]=read(rp) if rp.exists() else dict(status='process_failure',valid=False,exit_code=rc)
            if rc!=0:done[tid]['valid']=False
            if rp.exists() and done[tid]['valid']:aq.append(tid)
            del active[tid]
        for tid,(p,f) in list(analyses.items()):
            rc=p.poll()
            if rc is not None:f.close();adone[tid]=rc;del analyses[tid]
        hold=(B/'HOLD_NEW_LAUNCHES').exists() or sum(not r['valid'] for r in done.values())>=3 or shutil.disk_usage(B).free<100*1024**3
        target=8 if time.monotonic()-started<120 else m['workers']
        peer=read(Path(m['pilot_status']))
        target=min(target,max(0,64-peer.get('running',0)))
        peak=max([r.get('max_rss_KiB',0)/1024**2 for r in done.values()]+[0.])
        # Reserve headroom for startup/growth, not just observed current RSS.
        slots=max(0,int((memory()-24)/max(2.,peak*2)))
        target=min(target,len(active)+slots)
        while not hold and next_i<len(m['traces']) and len(active)<target and memory()>26:
            t=m['traces'][next_i];f=(B/'logs'/f"{t['id']}.log").open('xb');p=subprocess.Popen([sys.executable,'-u',str(B/'campaign.py'),'worker','--index',str(next_i)],stdout=f,stderr=subprocess.STDOUT,env=env)
            active[t['id']]=(p,f,time.monotonic());next_i+=1
        while aq and len(analyses)<max(0,2-peer.get('analysis_running',0)) and memory()>24:
            tid=aq.pop(0);f=(B/'logs'/f'{tid}.analysis.log').open('xb');p=subprocess.Popen([m['analysis_python'],'-u',str(B/'campaign.py'),'analyze','--trace',str(B/'traces'/tid)],stdout=f,stderr=subprocess.STDOUT,env=env);analyses[tid]=(p,f)
        compact={tid:{k:r.get(k) for k in ('status','valid','final_N','tau','events','wall_seconds')} for tid,r in done.items()}
        save(B/'outcomes.json',compact);save(B/'analysis_outcomes.json',adone)
        save(B/'status.json',dict(state='holding' if hold else 'running',pid=os.getpid(),planned=len(m['traces']),submitted=next_i,running=len(active),completed=len(done),queued=len(m['traces'])-next_i,analysis_running=len(analyses),analysis_completed=len(adone),analysis_failed=sum(x!=0 for x in adone.values()),status_counts=dict(collections.Counter(r['status'] for r in done.values())),available_GiB=memory()))
        if hold and not active and not analyses and not aq:break
        time.sleep(5)
    status=read(B/'status.json');status['state']='completed' if next_i==len(m['traces']) else 'holding';save(B/'status.json',status)
def launch():
    m=load();assert read(B/'TEST_RESULTS.json')['passed'];assert not (B/'launch.json').exists()
    with (B/'supervisor.log').open('xb') as f:p=subprocess.Popen([sys.executable,'-u',str(B/'campaign.py'),'supervise'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment(),start_new_session=True,close_fds=True,cwd=B)
    print(json.dumps(dict(supervisor_pid=p.pid,manifest_sha256=sha(B/'manifest.json'))))
if __name__=='__main__':
    assert socket.gethostname()=='xmhy-G3261','Run molecular calculations only on 5090'
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','test','worker','analyze','launch','supervise']);ap.add_argument('--index',type=int);ap.add_argument('--test',action='store_true');ap.add_argument('--trace',type=Path);a=ap.parse_args()
    if a.action=='worker':worker(a.index,a.test)
    elif a.action=='analyze':analyze_trace(a.trace)
    else:globals()[{'test':'tests'}.get(a.action,a.action)]()
