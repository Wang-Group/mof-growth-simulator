"""Fast orchestration acceptance tests, not production replicates."""
import gzip
import json
from pathlib import Path
import sys
from campaign_io import load_campaign, setup, lightweight, sha, write_json
from run_production_trace import run, snapshot


def main():
    base, spec = load_campaign(Path(__file__).parent/'manifest.json')
    out = base/'wrapper_checks'
    out.mkdir(exist_ok=False)
    engine, root, acetate, provenance = setup(base, spec)
    from runtime import make_model
    from trajectory import Simulation
    from check_molecular_integration import fingerprint
    seed = 2026091329
    def fresh():
        return Simulation(make_model(engine, root, acetate, seed=seed, rho=.1), seed ^ 0xFD7)
    raw = fresh()
    for _ in range(400):
        raw.advance()
    expected = (fingerprint(raw.model), raw.tau, raw.events_rng.getstate(), raw.event_count)
    observed = fresh()
    for i in range(400):
        observed.advance()
        if i % 13 == 0:
            lightweight(observed)
        if i == 199:
            snapshot(observed, out, 'replay_e200', provenance)
    actual = (fingerprint(observed.model), observed.tau, observed.events_rng.getstate(), observed.event_count)
    assert expected == actual, 'Observation/export perturbed molecular dynamics'
    restored = Simulation.restore(out/'snapshots/replay_e200/checkpoint.pkl', engine, provenance)
    for _ in range(200):
        restored.advance()
    assert (fingerprint(restored.model), restored.tau, restored.events_rng.getstate(), restored.event_count) == expected
    tests = ['400_event_observation_invariance', '200_event_disk_restart_exact']
    for name, target, wall, expected_status in [('target',400,60,'step_target_reached'),
            ('short_target',2,60,'step_target_reached'), ('wall_cap',50000,1e-9,'wall_time_cap')]:
        # A deliberately tiny legacy target_N must have no effect whatsoever.
        bounded = dict(spec, target_N=77, target_events=target, wall_seconds=wall,
                       milestones_N=[90,100] if name=='target' else [250,5000],
                       milestones_steps=[100,200,400] if name=='target' else [])
        production_sim = fresh()
        result = run(production_sim, out/name, provenance, bounded, {'id': name, 'purpose': 'wrapper_test'})
        assert result['status'] == expected_status and result['valid'], result
        assert result['censored'] == (name == 'wall_cap')
        assert result['events'] == (0 if name=='wall_cap' else target)
        if name=='target':
            assert result['final_N'] > 100 and result['target_N'] is None
            tests.append('no_entity_number_termination')
            assert (fingerprint(production_sim.model), production_sim.tau,
                    production_sim.events_rng.getstate(), production_sim.event_count) == expected
            tests.append('step_milestone_wrapper_matches_400_raw_events')
        with gzip.open(out/name/'entity_counts.jsonl.gz','rt') as f:
            rows = [json.loads(line) for line in f]
        assert rows[0]['N'] == 76 and rows[-1]['N'] == result['final_N']
        assert all(b['tau'] >= a['tau'] for a,b in zip(rows,rows[1:]))
        tests.append(name+'_exports_curves_status')
    assert (out/'target/snapshots/N00090/checkpoint.pkl').exists()
    assert (out/'target/snapshots/N00100/checkpoint.pkl').exists()
    for value in (100,200,400):
        assert (out/f'target/snapshots/steps{value:06d}/checkpoint.pkl').exists()
    tests.append('size_and_exact_step_snapshots')
    write_json(out/'summary.json', {'passed': True, 'tests': tests,
                'manifest_sha256': sha(base/'manifest.json'), 'not_production_statistics': True})
    print(json.dumps({'passed': True, 'tests': tests}))


if __name__ == '__main__':
    main()
