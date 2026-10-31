"""Finite 5090 campaign runner. Fresh output required; no automatic reruns."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

RUNTIME = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(bundle, root):
    global RUNTIME
    os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = bundle
    sys.path.insert(0, str(Path(bundle) / 'code/proposed'))
    sys.path.insert(0, root)
    import common_molecular_runner as runner
    import common_kmc_selector as selector
    engine, path, _ = runner._load_engine('uio66')
    assert sha(path) == '4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9'
    actual = runner.ENGINE_SPECS['uio66']['monomer_files']
    assert sha(actual['Zr6_AA.mol']) == 'ae2b840ae1f9e8ff9699623d008be96c50bf3a1ded2d863be8afeaed6c860b2e'
    assert sha(actual['BDC.mol']) == 'b67069e53051b2aefbb273b6a08d9339191486cf8ab1b261420686ef12847864'
    RUNTIME = engine, selector


def worker(row, root, output):
    from mixed_solvent_kmc import run_trace
    try:
        return run_trace(row, root, *RUNTIME, Path(output) / row['run_id'])
    except Exception:
        directory = Path(output) / row['run_id']
        directory.mkdir(exist_ok=True)
        error = {'design': row, 'status': 'implementation_error', 'valid_physical_outcome': False,
                 'traceback': traceback.format_exc()}
        (directory / 'ERROR.json').write_text(json.dumps(error, indent=2))
        return error


def code_hashes(root):
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ('solvent_channel.py', 'mixed_solvent_kmc.py',
                                      'run_mixed_campaign.py')] + [root / name for name in (
        'explicit_cap_kmc.py', 'validate_replaceable_caps.py', 'cap_geometry_preflight.py')]
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--bundle-root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--stage', choices=['pilot', 'grid'], required=True)
    ap.add_argument('--pilot-output', type=Path)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    assert socket.gethostname() == 'xmhy-G3261', 'Growth jobs belong on 5090'
    assert 1 <= args.workers <= 96
    root = args.root.resolve()
    assert sha(root / 'framework_inputs/UiO66_N76_reviewed.pkl') == '9b08834d55e0e1674d6e379eb991594947d000adeea6831d430e1f69b98e48c7'
    with args.manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    hashes = code_hashes(root)
    if args.stage == 'grid':
        assert args.pilot_output is not None
        summary = json.loads((args.pilot_output / 'SUMMARY.json').read_text())
        pilot_run = json.loads((args.pilot_output / 'RUN_MANIFEST.json').read_text())
        assert summary['completed'] == summary['designed'] == 288
        assert not any(k.startswith('invalid') or k == 'implementation_error'
                       for k, v in summary['status_counts'].items() if v)
        assert hashes == pilot_run['code_sha256'], 'Pilot-tested code changed; repeat pilot before grid'
        assert len(rows) == 27648
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bundle = args.bundle_root.resolve()
    inputs = list((root / 'molecule_library').glob('*.mol')) + [
        root / 'molecule_library/molecule_library_index.json',
        root / 'framework_inputs/UiO66_N76_reviewed.pkl',
        root / 'framework_inputs/BDC_explicit_H.mol']
    reviewed = [bundle / p for p in [
        'code/candidate_corrected/UiO66_Assembly_Large_Correction_20250811_candidate.py',
        'code/proposed/common_molecular_runner.py', 'code/proposed/common_kmc_selector.py',
        'models/workstation_source_20260831/mol/Zr6_AA.mol',
        'models/approved_derived/BDC_explicit_H.mol']]
    plan = {'hostname': socket.gethostname(), 'cpu_count': os.cpu_count(), 'workers': args.workers,
            'created_unix': time.time(), 'stage': args.stage, 'records': len(rows),
            'source_manifest_sha256': sha(args.manifest), 'code_sha256': hashes,
            'input_sha256': {str(p.relative_to(root)): sha(p) for p in inputs},
            'reviewed_source_sha256': {str(p.relative_to(bundle)): sha(p) for p in reviewed},
            'pilot_gate': str(args.pilot_output) if args.pilot_output else None,
            'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'invalid_is_not_physical_censoring': True, 'small_data_file': 'TRACE_SUMMARY.csv'}
    (output / 'RUN_MANIFEST.json').write_text(json.dumps(plan, indent=2))
    statuses, groups = Counter(), defaultdict(Counter)
    fields = ['run_id', 'identity', 'assay', 'rho_off', 'q_atomic_Zr_BDC', 'eta_S', 'replicate',
              'status', 'valid_physical_outcome', 'final_N', 'maximum_N', 'model_time',
              'executed_events', 'wall_seconds', 'initial_solvent_caps', 'initial_total_caps',
              'final_solvent_caps', 'final_total_caps', 'time_average_solvent_fraction', 'failure_reason']
    completed, stopped = 0, False
    row_iterator = iter(rows)
    with (output / 'TRACE_SUMMARY.csv').open('w', newline='') as small:
        writer = csv.DictWriter(small, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(bundle), str(root))) as pool:
            pending = {}
            def fill():
                if stopped:
                    return
                while len(pending) < 2 * args.workers:
                    row = next(row_iterator, None)
                    if row is None:
                        return
                    pending[pool.submit(worker, row, str(root), str(output))] = row
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    r = future.result()
                    completed += 1
                    statuses[r['status']] += 1
                    key = '|'.join(str(row[k]) for k in ('identity', 'assay', 'rho_off', 'q_atomic_Zr_BDC', 'eta_S'))
                    groups[key][r['status']] += 1
                    writer.writerow({k: r[k] if k in r else row.get(k) for k in fields})
                    small.flush()
                    invalid = sum(v for k, v in statuses.items() if k.startswith('invalid') or k == 'implementation_error')
                    # Stop scheduling new work if a systematic representation problem appears.
                    # Already-running records finish normally and all unscheduled records are marked.
                    if invalid >= 10:
                        stopped = True
                    progress = {'designed': len(rows), 'completed': completed, 'status_counts': dict(statuses),
                                'stopped_new_submission': stopped, 'updated_unix': time.time()}
                    (output / 'PROGRESS.json').write_text(json.dumps(progress, indent=2))
                    print(json.dumps({'completed': completed, 'run_id': row['run_id'], 'status': r['status'],
                                      'N': r.get('final_N'), 'wall_s': r.get('wall_seconds')}), flush=True)
                fill()
    summary = {'designed': len(rows), 'completed': completed, 'not_started': len(rows) - completed,
               'status_counts': dict(statuses), 'by_condition': {k: dict(v) for k, v in groups.items()},
               'stopped_new_submission': stopped, 'finished_unix': time.time()}
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    files = [{'path': p.relative_to(output).as_posix(), 'sha256': sha(p)}
             for p in sorted(output.rglob('*')) if p.is_file()]
    (output / 'REMOTE_SHA256.json').write_text(json.dumps(files, indent=2))
    print(json.dumps({'done': summary['completed'], 'status_counts': dict(statuses)}), flush=True)


if __name__ == '__main__':
    main()
