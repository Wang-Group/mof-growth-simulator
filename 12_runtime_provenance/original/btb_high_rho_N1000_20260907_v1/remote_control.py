"""Snapshot, verify, launch once, and inspect the approved pilot on 5090."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stamp():
    return datetime.now(timezone.utc).isoformat()

def write_once(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def read_design():
    design = json.loads((HERE / 'design.json').read_text())
    assert HERE == Path(design['remote_output']).resolve(), 'Wrong launch directory'
    assert sha(HERE / 'manifest.csv') == design['manifest_sha256']
    return design

def verify_snapshot(design):
    for relative, expected in design['expected_source_sha256'].items():
        assert sha(HERE / 'snapshot' / relative) == expected, relative

def preflight(design):
    assert not (HERE / 'launch.json').exists(), 'Already launched'
    snapshot = HERE / 'snapshot'
    snapshot.mkdir(exist_ok=False)
    for relative, expected in design['expected_source_sha256'].items():
        source = Path(design['remote_root']) / relative
        assert sha(source) == expected, f'Upstream source changed: {relative}'
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verify_snapshot(design)
    sys.path.insert(0, str(snapshot / 'code/proposed'))
    os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = str(snapshot)
    from common_kmc_selector import PhysicalCoordinates, OpportunityCounts, compute_propensities
    from seeded_molecular_runner import run_seeded_trace
    import numpy as np
    smoke = design['baseline_smoke']
    record = run_seeded_trace(
        system='btb_mol', seed_pickle=snapshot / design['seed_relative_path'],
        initial_seed_entities=30,
        coordinates=PhysicalCoordinates(1, 1/7, 10, 1),
        random_seed=smoke['random_seed'], target_entities=1000,
        event_cap=smoke['stop_event'], wall_time_cap_s=120,
        mol2_output_path=HERE / 'preflight/baseline_rho10_N80.mol2',
        mol2_milestones=(80,),
    )
    assert record['final_state']['entities'] == 80
    assert record['first_passage_event']['80'] == smoke['stop_event']
    assert math.isclose(record['first_passage_model_time']['80'],
                        smoke['expected_first_passage_time_80'], rel_tol=1e-12, abs_tol=1e-12)
    assert record['seed']['initial_entity_count'] == 30
    assert record['topology_mode'] == 'unrestricted'
    write_once(HERE / 'preflight/baseline_replay.json', record)
    # Verify only the removal coefficient changes across the approved conditions.
    props = {}
    for rho in design['rho_off']:
        p = compute_propensities(OpportunityCounts(3, 12, 1, 5), PhysicalCoordinates(1,1/7,rho,1))
        assert p.remove_bond == 5 * rho
        assert p.add_node == 3/7 and math.isclose(p.add_linker, 12*6/7)
        assert math.isclose(p.close_internal, math.exp(30.9))
        props[str(rho)] = vars(p)
    checks = dict(timestamp_utc=stamp(), passed=True, baseline_replay_match=True,
                  python=sys.version, executable=sys.executable, numpy=np.__version__,
                  hostname=platform.node(), cpu_count=os.cpu_count(),
                  source_hashes=design['expected_source_sha256'],
                  manifest_sha256=sha(HERE / 'manifest.csv'),
                  controller_sha256=sha(Path(__file__)), propensities_test=props,
                  geometry_adapters=record['provenance'].get('monomer_path_redirects',
                                                           record['provenance'].get('path_redirects')))
    write_once(HERE / 'preflight/checks.json', checks)
    print(json.dumps(checks, indent=2))

def supervise(design):
    env = os.environ.copy()
    env['MOL_GROWTH_REVIEW_BUNDLE'] = str(HERE / 'snapshot')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[k] = '1'
    command = [sys.executable, '-u', str(HERE / 'snapshot/code/proposed/run_seeded_manifest.py'),
               '--manifest', str(HERE / 'manifest.csv'), '--output-dir', str(HERE / 'results'),
               '--workers', str(design['workers'])]
    result = subprocess.run(command, env=env, cwd=HERE)
    write_once(HERE / 'completion.json', dict(timestamp_utc=stamp(), returncode=result.returncode))
    raise SystemExit(result.returncode)

def launch(design):
    verify_snapshot(design)
    checks = json.loads((HERE / 'preflight/checks.json').read_text())
    assert checks['passed'] and checks['controller_sha256'] == sha(Path(__file__))
    lock = os.open(HERE / 'launch.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(lock)
    assert not (HERE / 'results').exists(), 'Results directory already exists'
    with (HERE / 'campaign.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__)), 'supervise'],
                                   cwd=HERE, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    info = dict(timestamp_utc=stamp(), supervisor_pid=process.pid, workers=design['workers'],
                command=[sys.executable, str(Path(__file__)), 'supervise'],
                manifest_sha256=sha(HERE / 'manifest.csv'), remote_output=str(HERE))
    write_once(HERE / 'launch.json', info)
    print(json.dumps(info, indent=2))

def status(design):
    with (HERE / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    summary, complete = [], Counter()
    artifacts = []
    for row in rows:
        path = HERE / 'results' / (row['run_id'] + '.json')
        error = path.with_suffix('.error.json')
        record = dict(run_id=row['run_id'], rho_off=float(row['off_weight']), replicate=int(row['replicate']))
        if path.exists():
            j = json.loads(path.read_text())
            record.update(status=j['termination']['status'], N=j['final_state']['entities'],
                          Nmax=j['termination']['maximum_entities'], events=j['termination']['executed_events'],
                          wall_seconds=j['wall_seconds'],
                          first_passage_1000=j['first_passage_model_time'].get('1000'),
                          seed_retained_fraction=j['seed']['final_retained_fraction'])
            for milestone in j['provenance'].get('mol2_milestones', []):
                record[f"N{milestone['requested_entities']}_saved"] = True
            complete[record['status']] += 1
        elif error.exists():
            record.update(status='error', error=json.loads(error.read_text()).get('error'))
            complete['error'] += 1
        else:
            record['status'] = 'pending_or_running'
            structure_dir = HERE / 'results/structures'
            record['saved_milestones'] = [n for n in design['milestones']
                if (structure_dir / f"{row['run_id']}_N{n:04d}.mol2").exists()]
        summary.append(record)
    completion = HERE / 'completion.json'
    print(json.dumps(dict(timestamp_utc=stamp(), counts=dict(complete), total=32,
                          completion=json.loads(completion.read_text()) if completion.exists() else None,
                          rows=summary), indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight','launch','supervise','status'])
    args = parser.parse_args()
    globals()[args.mode](read_design())
