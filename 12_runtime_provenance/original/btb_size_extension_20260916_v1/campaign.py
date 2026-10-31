"""Prepare, validate and run a finite 16-trajectory BTB size extension on 5090."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'btb_high_rho_N1000_20260907_v1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def prepare():
    from extend_btb import atomic_json, sha256
    with (SOURCE/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    rows = []
    for row in original:
        if int(row['replicate']) >= 4:
            continue
        row['source_run_id'] = row['run_id']
        row['run_id'] = f"btb_extend_rho{int(float(row['off_weight']))}_rep{int(row['replicate']):02d}"
        row.update(target_entities=10000, event_cap=1000000000, wall_time_cap_s=7200,
                   mol2_milestones='250;500;1000;2000;5000;10000')
        rows.append(row)
    path = HERE/'manifest.csv'
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    frozen = {}
    for path in sorted((SOURCE/'snapshot').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            frozen[str(path.relative_to(SOURCE/'snapshot'))] = sha256(path)
    atomic_json(HERE/'source_snapshot_sha256.json', frozen)
    atomic_json(HERE/'design.json', dict(planned=len(rows), workers=8, target=10000,
        rho_off=[10,30,50,100], replicate_selection='first four indices 0-3 at every condition, no outcome selection',
        wall_seconds_per_trajectory=7200, maximum_worker_hours=32,
        seed_entities=30, node_fraction=1/7, s_close=1, external_activity=1,
        constant_pool=True, molecular_growth='unrestricted 3D; all 12 node directions',
        relationship_to_previous='deterministic replay and extension, not independent new replicates',
        checkpoint='Full assembly and three RNG states; 120 s checks and every size milestone',
        limitations=['Finite cap does not guarantee N=10000.',
            'Higher N does not impose a flat layer or remove genuine secondary scattering.',
            'Retain capped structures; compare matched sizes before comparing endpoints.']))


def rows():
    with (HERE/'manifest.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def validate():
    from extend_btb import run, atomic_json, sha256
    evidence = []
    # Matched full legacy N=1000 structures at rho 10 and 30.
    for rho in (10,30):
        row = next(r.copy() for r in rows() if float(r['off_weight'])==rho and int(r['replicate'])==0)
        row.update(target_entities=1000, wall_time_cap_s=900)
        got = run(row, HERE/'validation'/f'legacy_match_rho{rho}')
        old = json.loads((SOURCE/'results'/f"{row['source_run_id']}.json").read_text())
        checks = dict(target_reached=got['termination']['status']=='target_reached',
            events=got['termination']['executed_events']==old['termination']['executed_events'],
            clock=got['clock']['model_time']==old['clock']['model_time'],
            counters=got['event_counts']==old['event_counts'],
            molecular_sha256=got['final_mol2_sha256']==old['provenance']['mol2_output']['sha256'])
        evidence.append(dict(rho=rho, checks=checks, seconds=got['wall_seconds']))
        atomic_json(HERE/'validation/progress.json', evidence)
        if not all(checks.values()):
            raise AssertionError(evidence[-1])
    # Split checkpoint replay must reproduce unsplit assembly, clock, counters.
    row = rows()[0].copy()
    row.update(target_entities=300, wall_time_cap_s=900, mol2_milestones='80;200;300')
    whole = run(row, HERE/'validation/whole')
    split_row = row.copy(); split_row['target_entities']=200
    run(split_row, HERE/'validation/split')
    tail = run(row, HERE/'validation/resumed', HERE/'validation/split/checkpoint.pkl')
    checks = dict(clock=whole['clock']==tail['clock'],
        events=whole['termination']['executed_events']==tail['termination']['executed_events'],
        counters=whole['event_counts']==tail['event_counts'],
        molecular_sha256=whole['final_mol2_sha256']==tail['final_mol2_sha256'])
    evidence.append(dict(test='checkpoint_continuation', checks=checks))
    if not all(checks.values()):
        raise AssertionError(evidence[-1])
    atomic_json(HERE/'validation/PASS.json', dict(passed=True, tests=evidence,
        runner_sha256=sha256(HERE/'extend_btb.py'), unix=time.time()))


def run_one(row):
    from extend_btb import run, atomic_json
    # Serialization traverses cyclic molecular graphs; this changes no events.
    import sys
    sys.setrecursionlimit(20000)
    out = HERE/'results'/row['run_id']
    try:
        result = run(row, out)
        return dict(run_id=row['run_id'], status=result['termination']['status'],
                    N=result['final_state']['entities'], events=result['termination']['executed_events'])
    except Exception:
        error = dict(run_id=row['run_id'], status='error', traceback=traceback.format_exc())
        atomic_json(out/'ERROR.json', error)
        return error


def launch():
    from extend_btb import atomic_json, sha256
    test = json.loads((HERE/'validation/PASS.json').read_text())
    assert test['passed'] and test['runner_sha256']==sha256(HERE/'extend_btb.py')
    expected = json.loads((HERE/'source_snapshot_sha256.json').read_text())
    for relative, checksum in expected.items():
        assert sha256(SOURCE/'snapshot'/relative)==checksum, relative
    lock = HERE/'production.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    completed = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, row) for row in rows()]
        atomic_json(HERE/'launch.json', dict(pid=os.getpid(), started_unix=time.time(), workers=8,
            planned=len(futures), python=os.sys.executable))
        for future in as_completed(futures):
            completed.append(future.result())
            atomic_json(HERE/'completed.json', completed)
    atomic_json(HERE/'COMPLETE.json', dict(completed=completed, finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action', choices=['prepare','validate','launch'])
    action=p.parse_args().action
    {'prepare':prepare, 'validate':validate, 'launch':launch}[action]()
