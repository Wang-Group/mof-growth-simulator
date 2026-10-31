"""Finite queue, guarded concurrency, preserved failures; no automatic reruns."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from campaign_io import load_campaign, write_json, sha


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    return 0.


def rss_gib(pid):
    try:
        values = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                values[line.split(':')[0]] = int(line.split()[1])/1024**2
        return max(values.values(), default=0.)
    except FileNotFoundError:
        return 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()
    base, m = load_campaign(args.manifest)
    lock = (base/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (base/'launch.json').exists():
        raise RuntimeError('Existing launch; explicit recovery is required, never duplicate trajectories')
    tests = json.loads((base/'wrapper_checks/summary.json').read_text())
    assert tests['passed'] and tests['manifest_sha256'] == sha(args.manifest)
    write_json(base/'launch.json', {'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
               'manifest_sha256': sha(args.manifest), 'planned_traces': len(m['traces']),
               'initial_workers': m['workers']['initial'], 'maximum_workers': m['workers']['maximum']})
    os.nice(10)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='0')
    (base/'logs').mkdir(exist_ok=True)
    running, analyses, analysis_queue = {}, {}, []
    outcomes, analyzed = {}, {}
    next_index, started, last_save, observed_peak = 0, time.monotonic(), 0., 0.
    while next_index < len(m['traces']) or running or analysis_queue or analyses:
        for tid, item in list(running.items()):
            proc, handle, launch_time = item
            observed_peak = max(observed_peak, rss_gib(proc.pid))
            code = proc.poll()
            if code is None:
                # Finite external watchdog only for an unresponsive event/export.
                if time.monotonic()-launch_time > m['wall_seconds']+600:
                    proc.terminate()
                if time.monotonic()-launch_time > m['wall_seconds']+900:
                    proc.kill()
                continue
            handle.close()
            result_path = base/'traces'/tid/'result.json'
            if result_path.exists():
                r = json.loads(result_path.read_text())
                outcome = {k: r[k] for k in ('status', 'valid', 'censored', 'target_reached',
                           'final_N', 'maximum_N', 'events', 'tau', 'wall_seconds', 'max_rss_KiB')}
                observed_peak = max(observed_peak, r['max_rss_KiB']/1024**2)
                analysis_queue.append(tid)
            else:
                outcome = {'status': 'process_failure', 'valid': False, 'censored': False,
                           'target_reached': False, 'exit_code': code,
                           'note': 'Not a physical zero-defect or no-growth outcome; logs retained'}
            outcome['exit_code'] = code
            outcomes[tid] = outcome
            del running[tid]
        for tid, (proc, handle) in list(analyses.items()):
            code = proc.poll()
            if code is not None:
                handle.close()
                analyzed[tid] = code
                del analyses[tid]
        elapsed = time.monotonic()-started
        target = m['workers']['initial'] if elapsed < m['workers']['ramp_after_seconds'] else m['workers']['maximum']
        memory_per_worker = max(m['workers']['minimum_worker_memory_GiB'],
                                observed_peak*m['workers']['observed_worker_peak_safety_factor'])
        reserve = m['workers']['memory_reserve_GiB']
        free_slots = max(0, int((available_gib()-reserve)/memory_per_worker))
        errors = sum(not x.get('valid', False) for x in outcomes.values())
        # Three invalid runs pause NEW launches for review, without touching active runs.
        hold = errors >= 3 or (base/'HOLD_NEW_LAUNCHES').exists() or shutil.disk_usage(base).free < 100*1024**3
        count = 0 if hold else min(target-len(running), free_slots, len(m['traces'])-next_index)
        for _ in range(max(0, count)):
            t = m['traces'][next_index]
            log = (base/'logs'/(t['id']+'.log')).open('xb')
            cmd = [sys.executable, '-u', str(base/'run_production_trace.py'), '--manifest',
                   str(args.manifest), '--index', str(next_index)]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=base)
            running[t['id']] = (proc, log, time.monotonic())
            next_index += 1
        while analysis_queue and len(analyses) < m['analysis']['workers'] and available_gib() > reserve:
            tid = analysis_queue.pop(0)
            log = (base/'logs'/(tid+'.analysis.log')).open('xb')
            proc = subprocess.Popen([m['analysis']['python'], '-u', str(base/'analyze_production_trace.py'),
                       '--manifest', str(args.manifest), '--trace', str(base/'traces'/tid)],
                       stdout=log, stderr=subprocess.STDOUT, env=env, cwd=base)
            analyses[tid] = (proc, log)
        if time.monotonic()-last_save >= 15:
            progress = {}
            for tid, (proc, _, _) in running.items():
                p = base/'traces'/tid/'progress.json'
                progress[tid] = json.loads(p.read_text()) if p.exists() else {'pid': proc.pid, 'status': 'starting'}
            write_json(base/'status.json', {'utc': datetime.now(timezone.utc).isoformat(),
                'state': 'holding_new_launches' if hold else 'running', 'pid': os.getpid(),
                'submitted': next_index, 'planned': len(m['traces']), 'running': len(running),
                'completed': len(outcomes), 'queued': len(m['traces'])-next_index,
                'status_counts': dict(Counter(x['status'] for x in outcomes.values())),
                'available_GiB': available_gib(), 'observed_peak_worker_GiB': observed_peak,
                'current_worker_target': target, 'launch_hold': hold,
                'analysis_running': len(analyses), 'analysis_completed': len(analyzed),
                'analysis_failed': sum(x != 0 for x in analyzed.values()),
                'active_progress': progress})
            write_json(base/'outcomes.json', outcomes)
            write_json(base/'analysis_outcomes.json', analyzed)
            last_save = time.monotonic()
        if hold and not running and not analyses and not analysis_queue:
            break
        time.sleep(5)
    write_json(base/'outcomes.json', outcomes)
    write_json(base/'analysis_outcomes.json', analyzed)
    final_record = {'submitted': next_index, 'planned': len(m['traces']),
        'completed': len(outcomes), 'status_counts': dict(Counter(x['status'] for x in outcomes.values())),
        'all_submitted': next_index == len(m['traces']), 'utc': datetime.now(timezone.utc).isoformat()}
    write_json(base/'queue_end.json', final_record)
    write_json(base/'status.json', dict(final_record,
        state='completed' if next_index == len(m['traces']) else 'holding_new_launches',
        running=0, queued=len(m['traces'])-next_index, analysis_running=0,
        analysis_completed=len(analyzed), analysis_failed=sum(x != 0 for x in analyzed.values())))


if __name__ == '__main__':
    main()
