"""Bounded production wrapper; delegates EVERY event to frozen Simulation.advance."""
import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback
from campaign_io import load_campaign, setup, lightweight, write_json, sha

STOP = None


def handle_signal(number, frame):
    global STOP
    STOP = 'resource_stop' if number == signal.SIGTERM else 'interrupted'


def snapshot(sim, out, tag, provenance):
    from validate_replaceable_caps import export_xyz
    dest = out / 'snapshots' / tag
    dest.mkdir(parents=True, exist_ok=False)
    sim.model.check_state()
    sim.checkpoint(dest / 'checkpoint.pkl', provenance)
    export_xyz(dest / 'framework_only.xyz', sim.model.bare_blocks(),
               'Physical Zr6/BDC coordinates; acetate and solvent proxies excluded')
    export_xyz(dest / 'GEOMETRY_ONLY.xyz', sim.model.blocks(),
               'Solvent C,O,O surrogate included; NOT physical all-atom PXRD input')
    write_json(dest / 'state.json', {**lightweight(sim),
                'intermediates': sim.model.intermediate_record(), 'provenance': provenance})
    write_json(dest / 'SHA256.json', {p.name: sha(p) for p in dest.iterdir() if p.is_file()})
    return str(dest.relative_to(out))


def run(sim, out, provenance, spec, trace):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'input.json', trace)
    started = time.monotonic()
    last_progress = last_checkpoint = started
    snapshots, status, error = [], 'running', None
    if not hasattr(sim, 'first_passage_steps'):
        sim.first_passage_steps = {}
    curve = gzip.open(out / 'entity_counts.jsonl.gz', 'wt')

    def record():
        row = lightweight(sim)
        curve.write(json.dumps(row, allow_nan=False) + '\n')
        curve.flush()
        return row

    def progress(state='running'):
        write_json(out / 'progress.json', {**lightweight(sim), 'status': state,
            'elapsed_seconds': time.monotonic()-started,
            'max_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'pid': os.getpid(), 'snapshots': snapshots})

    try:
        record()
        progress()
        while True:
            n = len(sim.model.assembly.entities)
            for value in spec['milestones_N']:
                if n >= value and str(value) not in sim.first_passage:
                    sim.first_passage[str(value)] = {'N': n, 'events': sim.event_count, 'tau': sim.tau}
                    snapshots.append(snapshot(sim, out, f'N{value:05d}', provenance))
            for value in sorted(set(spec['milestones_steps']) | {spec['target_events']}):
                if sim.event_count >= value and str(value) not in sim.first_passage_steps:
                    sim.first_passage_steps[str(value)] = {'N': n, 'events': sim.event_count, 'tau': sim.tau}
                    snapshots.append(snapshot(sim, out, f'steps{value:06d}', provenance))
            # Entity number is an observable, not a stopping condition.
            if sim.event_count >= spec['target_events']:
                status = 'step_target_reached'
                break
            if STOP:
                status = STOP
                break
            if time.monotonic()-started >= spec['wall_seconds']:
                status = 'wall_time_cap'
                break
            outcome = sim.advance()
            if outcome != 'event':
                status = outcome
                break
            if sim.event_count % spec['curve_event_interval'] == 0:
                record()
            now = time.monotonic()
            if now-last_progress >= spec['progress_wall_interval_seconds']:
                progress()
                last_progress = now
            if now-last_checkpoint >= spec['checkpoint_wall_interval_seconds']:
                tag = f'checkpoint_e{sim.event_count:09d}'
                dest = out / (tag + '.pkl')
                sim.checkpoint(dest, provenance)
                last_checkpoint = now
    except Exception:
        status, error = 'error', traceback.format_exc()
    finally:
        final = None
        export_error = None
        try:
            final = record()
            snapshots.append(snapshot(sim, out, 'terminal', provenance))
        except Exception:
            export_error = traceback.format_exc()
            status = 'error'
        curve.close()
        report = {'status': status, 'error': error, 'export_error': export_error,
            'valid': status != 'error',
            'censored': status in {'event_cap', 'wall_time_cap', 'resource_stop', 'interrupted'},
            'target_reached': status == 'step_target_reached', 'target_kind': 'event_steps',
            'target_events': spec['target_events'], 'target_N': None,
            'final_N': len(sim.model.assembly.entities), 'maximum_N': sim.maximum_N,
            'events': sim.event_count, 'tau': sim.tau, 'trace': trace,
            'wall_seconds': time.monotonic()-started,
            'max_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'first_passage': sim.first_passage, 'first_passage_steps': sim.first_passage_steps,
            'snapshots': snapshots,
            'transition_counts': dict(sim.model.transition_counts),
            'final': final, 'provenance': provenance}
        write_json(out / 'result.json', report)
        try:
            progress(status)
        except Exception:
            pass
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--index', type=int, required=True)
    args = ap.parse_args()
    base, spec = load_campaign(args.manifest)
    trace = spec['traces'][args.index]
    engine, root, acetate, provenance = setup(base, spec)
    from runtime import make_model
    from trajectory import Simulation
    model = make_model(engine, root, acetate, seed=trace['seed'], delta=trace['delta'],
                       mode=trace['mode'], fragment_policy=trace['fragment_policy'],
                       rho=trace['rho0'], s_close=spec['s_close'],
                       free=(spec['initial_free']['Zr6'], spec['initial_free']['BDC']))
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    result = run(Simulation(model, trace['seed'] ^ 0xFD7), base/'traces'/trace['id'],
                 provenance, spec, trace)
    print(json.dumps({k: result[k] for k in ('status', 'final_N', 'events', 'tau', 'wall_seconds')}), flush=True)
    if not result['valid']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
