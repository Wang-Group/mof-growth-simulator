"""Capped end-to-end tests including disk restart and physical closure thinning."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

from runtime import configure, make_model, sha
from trajectory import Simulation, run_bounded


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'bundle', 'preflight', 'acetate-source', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    engine, provenance = configure(args.root, args.bundle, args.preflight)
    from check_molecular_integration import fingerprint
    from defect_model import GraphTransaction
    provenance['sources'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    report = {'status': 'running', 'tests': [], 'provenance': provenance, 'not_production': True}
    started = time.perf_counter()

    def fresh(**kw):
        return make_model(engine, args.root, args.acetate_source, **kw)

    def passed(name, **details):
        item = {'name': name, 'passed': True, **details}
        report['tests'].append(item)
        print(json.dumps(item), flush=True)

    try:
        # Preview matches the full transactional closure, without RNG consumption.
        m = fresh(seed=20260920, rho=.1)
        rng = random.Random(20260920)
        tested, blocked = 0, 0
        for _ in range(1200):
            for pair in list(m.assembly.ready_to_connect_carboxylate_pairs.items):
                allowed = pair in m.closure_candidates()
                signature = fingerprint(m)
                tx = GraphTransaction(m)
                before_blocks = m.block_map()
                before_rng = json.dumps(m.np_rng.bit_generator.state)
                m.in_transition = True
                m.assembly.link_internal_carboxylate(pair)
                m._refresh_physical_state()
                valid, _ = m._geometry_valid(before_blocks)
                assert allowed == valid
                assert json.dumps(m.np_rng.bit_generator.state) == before_rng
                tx.restore()
                m.in_transition = False
                assert fingerprint(m) == signature
                m.check_state()
                tested += 1
                blocked += not allowed
            event, item = m.choose(m.propensities(), rng)
            m.attempt(event, item, rng)
            if tested >= 30 and blocked:
                break
        assert tested >= 30 and blocked > 0
        passed('deterministic_closure_thinning_equals_actual_geometry', checked=tested, blocked=blocked)

        # Exact disk restart, including both global RNGs and pending event state.
        s = Simulation(fresh(seed=20260930), 447)
        for _ in range(200):
            assert s.advance() == 'event'
        s.observe()
        cp = args.output/'restart_checkpoint.pkl'
        s.checkpoint(cp, provenance)
        for _ in range(150):
            s.advance()
        expected = (fingerprint(s.model), s.tau, s.events_rng.getstate(), s.event_count)
        restored = Simulation.restore(cp, engine, provenance)
        for _ in range(150):
            restored.advance()
        actual = (fingerprint(restored.model), restored.tau, restored.events_rng.getstate(), restored.event_count)
        assert actual == expected
        passed('disk_checkpoint_exact_150_event_continuation', events=restored.event_count, fingerprint=actual[0])
        try:
            Simulation.restore(cp, engine, {'wrong_source': True})
        except ValueError:
            passed('checkpoint_provenance_mismatch_rejected')
        else:
            raise AssertionError('Accepted mismatched checkpoint')

        # Time-boundary truncation must preserve the remaining exponential wait.
        limited = Simulation(fresh(seed=20260931), 448)
        outcome = limited.advance(1e-20)
        assert outcome == 'model_time_cap' and limited.event_count == 0
        cp = args.output/'pending_checkpoint.pkl'
        limited.checkpoint(cp, provenance)
        limited.advance()
        expected = (fingerprint(limited.model), limited.tau, limited.events_rng.getstate())
        recovered = Simulation.restore(cp, engine, provenance)
        recovered.advance()
        assert (fingerprint(recovered.model), recovered.tau, recovered.events_rng.getstate()) == expected
        passed('model_time_cap_and_pending_wait_restart')

        # Eight short, fully timed molecular traces. Preserve every capped result.
        for delta in (0., 6.):
            for mode in ('fixed', 'finite'):
                for policy in ('retain_oligomers', 'redissolve'):
                    name = f'trace_delta{delta:g}_{mode}_{policy}'
                    s = Simulation(fresh(seed=20260932, rho=.1, delta=delta,
                                         mode=mode, fragment_policy=policy), 449)
                    result = run_bounded(s, args.output/name, provenance, target=150,
                                         max_events=5000, wall_seconds=30)
                    assert result['status'] in ('target_reached', 'event_cap', 'wall_time_cap')
                    rows = s.observations
                    assert all(a['tau'] <= b['tau'] and a['events'] <= b['events']
                               for a, b in zip(rows, rows[1:]))
                    assert all(r['pool']['coefficients']['remove_bond'] == .1 for r in rows)
                    if mode == 'fixed':
                        assert all(r['pool']['free'] == {'Zr': 1000, 'Ligand': 6000} for r in rows)
                    else:
                        assert rows[-1]['pool']['free'] != rows[0]['pool']['free']
                    assert (args.output/name/'checkpoint.pkl').stat().st_size > 0
                    passed(name, status=result['status'], final_N=result['final_N'], events=result['events'],
                           tau=result['tau'], wall_seconds=result['wall_seconds'],
                           interior_linker_positions=result['final']['vacancies']['1']['interior_linker_positions'])

        # Deliberate unattainable target: tiny finite feedstock never becomes negative.
        s = Simulation(fresh(seed=20260933, rho=.1, free=(4, 12), fragment_policy='redissolve'), 450)
        result = run_bounded(s, args.output/'tiny_pool_censor', provenance, target=150,
                             max_events=1500, wall_seconds=20)
        assert result['censored'] and result['maximum_N'] <= 92
        s.model.ledger.check()
        passed('finite_pool_capacity_and_censoring', status=result['status'], maximum_N=result['maximum_N'],
               free=s.model.ledger.free)

        # The real CLI must reproduce across separate Python processes.
        signatures = []
        for replicate in range(2):
            out = args.output/f'cli_replay_{replicate}'
            command = [sys.executable, str(Path(__file__).parent/'run_defect_trace.py')]
            for name in ('root', 'bundle', 'preflight', 'acetate_source'):
                command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
            command += ['--output', str(out), '--mode', 'finite', '--fragment-policy', 'retain_oligomers',
                        '--delta', '2', '--rho', '.1', '--seed', '20260934', '--target', '90',
                        '--max-events', '1000', '--wall-seconds', '30']
            completed = subprocess.run(command, text=True, capture_output=True, timeout=60)
            assert completed.returncode == 0, completed.stderr
            result = json.loads((out/'result.json').read_text())
            signatures.append((result['final_N'], result['events'], result['tau'],
                               sha(out/'final_framework_only.xyz'), result['transition_counts']))
        assert signatures[0] == signatures[1]
        passed('independent_process_cli_replay', final_N=signatures[0][0], events=signatures[0][1])
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        (args.output/'trajectory_checks.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
