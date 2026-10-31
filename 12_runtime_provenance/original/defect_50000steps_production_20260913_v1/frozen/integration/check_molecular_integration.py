"""Bounded 5090 molecular integration tests; no production launcher."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from runtime import configure, make_model, sha
from vacancy_counter import FCCRegistration, assembly_coordinates


def fingerprint(model):
    h = hashlib.sha256()
    rows = []
    for block in model.blocks():
        rows.append((block['entity'], block['site'], block['role'],
                     ','.join(block['elements']), np.asarray(block['xyz']).tobytes().hex()))
    h.update(json.dumps(sorted(rows)).encode())
    pairs = {}
    for name in ('linked_carboxylate_pairs', 'ready_to_connect_carboxylate_pairs'):
        pairs[name] = sorted(sorted((model.identifiers[s.belonging_entity],
                                    s.belonging_entity.carboxylates.index(s)) for s in pair)
                             for pair in getattr(model.assembly, name).items)
    h.update(json.dumps(pairs, sort_keys=True).encode())
    h.update(json.dumps(model.ledger.record(), sort_keys=True).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--preflight', type=Path, required=True)
    ap.add_argument('--acetate-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    engine, provenance = configure(args.root, args.bundle, args.preflight)
    from preflight_crowded_geometry import nonseparating_edge
    from validate_replaceable_caps import export_xyz
    from defect_model import ordered_pair
    report = {'status': 'running', 'tests': [], 'provenance': provenance,
              'not_production': True, 'source_hashes': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    started = time.perf_counter()

    def passed(name, **details):
        report['tests'].append({'name': name, 'passed': True, **details})
        print(json.dumps(report['tests'][-1]), flush=True)

    def fresh(**kw):
        return make_model(engine, args.root, args.acetate_source, **kw)

    try:
        model = fresh()
        nodes, linkers, degrees = assembly_coordinates(model.assembly)
        reg = FCCRegistration.infer(nodes)
        result = reg.count(nodes, linkers, degrees)
        assert result['unregistered_node_count'] == 0
        assert result['unregistered_linker_count'] == 0
        passed('seed_registration', **result)
        initial = fingerprint(model)
        pair = next(p for p in model.assembly.linked_carboxylate_pairs.items
                    if all(len(s.belonging_entity.connected_entities) >= 2 for s in p)
                    and nonseparating_edge(model.assembly, p))
        rng = random.Random(100)
        for attempt in range(128):
            accepted = model.attempt('remove_bond', pair, rng)
            if accepted:
                break
            assert fingerprint(model) == initial
        else:
            raise AssertionError('No F-D candidate was reached in bounded attempts')
        assert len(model.intermediates) == 1
        export_xyz(args.output/'D_GEOMETRY_ONLY.xyz', model.blocks(), 'Explicit capped D; solvent proxy, not PXRD')
        d = next(iter(model.intermediates.values()))
        assert model.attempt('close_internal', d['pair'], rng)
        assert fingerprint(model) == initial
        passed('F_D_F_round_trip_and_rejection_rollback', attempts=attempt+1)

        # Real bond deletion and repair: preserve the actual receiver sites, not
        # a posthoc relabelled probability on the original graph.
        for attempt in range(128):
            if model.attempt('remove_bond', pair, rng):
                break
        else:
            raise AssertionError('Failed to re-enter D')
        linker = next(iter(model.intermediates))
        retained = next(p for p in model.assembly.linked_carboxylate_pairs.items
                        if any(s.belonging_entity is linker for s in p))
        repair_site = ordered_pair(retained)[0]
        before_pool = model.ledger.free['Ligand']
        for attempt in range(128):
            if model.attempt('remove_bond', retained, rng):
                break
        else:
            raise AssertionError('No D-V transition')
        assert len(model.assembly.entities) == 75 and not model.intermediates
        assert model.ledger.free['Ligand'] == before_pool+1
        export_xyz(args.output/'V_GEOMETRY_ONLY.xyz', model.blocks(), 'Actual BDC vacancy; caps remain')
        for attempt in range(128):
            if model.attempt('add_linker', repair_site, rng):
                break
        else:
            raise AssertionError('No V-D repair')
        assert len(model.assembly.entities) == 76 and len(model.intermediates) == 1
        assert model.ledger.free['Ligand'] == before_pool
        passed('D_V_D_actual_linker_loss_repair_and_pool_transfer', attempts=attempt+1)

        # Energy rejection must leave both graph and precursor pool unchanged.
        energy_model = fresh(delta=10.)
        energy_pair = energy_model.assembly.linked_carboxylate_pairs.items[0]
        before = fingerprint(energy_model)
        observed_energy_rejection = False
        for _ in range(128):
            accepted = energy_model.attempt('remove_bond', energy_pair, rng)
            if energy_model.last_transition.get('rejection') == 'intermediate_energy':
                observed_energy_rejection = True
                assert fingerprint(energy_model) == before
                break
            if accepted:
                raise AssertionError('Unexpected rare event in fixed deterministic energy test')
        assert observed_energy_rejection
        passed('energy_rejection_rolls_back_graph_caps_and_inventory')

        # Discrete microscopic rate-ratio checks use molecularly generated D
        # geometries and explicitly retained cap/angle proposal measures.
        ratios = []
        for delta in (0., 2., 10.):
            m = fresh(delta=delta)
            p = next(p for p in m.assembly.linked_carboxylate_pairs.items
                     if all(len(s.belonging_entity.connected_entities) >= 2 for s in p)
                     and nonseparating_edge(m.assembly, p))
            for _ in range(128):
                if m.attempt('remove_bond', p, rng, apply_energy=False):
                    break
            d = next(iter(m.intermediates.values()))
            prior = d['angle_prior'] * m.cap[d['metal_site']]['prior']
            forward = m.ledger.rho0 * prior * math_exp(-delta)
            reverse = m.ledger.coefficients()['close_internal']
            equilibrium_ratio = m.ledger.rho0 / reverse * prior * math_exp(-delta)
            assert abs((forward/reverse)/equilibrium_ratio-1) < 1e-12
            ratios.append({'delta': delta, 'cap_kind': m.cap[d['metal_site']]['kind'],
                           'prior': prior, 'D_over_F_local_measure': equilibrium_ratio})
        passed('molecular_microstate_local_pair_ratios', cases=ratios)

        # Bounded dynamic checks under both declared fragment limits and baths.
        signatures = []
        for mode in ('fixed', 'finite'):
            for policy in ('retain_oligomers', 'redissolve'):
                m = fresh(mode=mode, fragment_policy=policy, seed=20260914)
                events = random.Random(102)
                maximum = 76
                for step in range(400):
                    prop = m.propensities()
                    event, item = m.choose(prop, events)
                    m.attempt(event, item, events)
                    maximum = max(maximum, len(m.assembly.entities))
                m.check_state()
                signature = fingerprint(m)
                signatures.append((mode, policy, signature))
                passed('bounded_dynamic_'+mode+'_'+policy, events=400, final_N=len(m.assembly.entities),
                       maximum_N=maximum, counters=dict(m.transition_counts), pool=m.ledger.record(), fingerprint=signature)
                if mode == 'finite' and policy == 'retain_oligomers':
                    export_xyz(args.output/'dynamic_GEOMETRY_ONLY.xyz', m.blocks(), 'Bounded integration, not production')
        replay = fresh(mode='finite', fragment_policy='retain_oligomers', seed=20260914)
        events = random.Random(102)
        for _ in range(400):
            event, item = replay.choose(replay.propensities(), events)
            replay.attempt(event, item, events)
        expected = next(v for mode, policy, v in signatures if mode == 'finite' and policy == 'retain_oligomers')
        assert fingerprint(replay) == expected
        passed('independent_restart_exact_replay', fingerprint=expected)
        # Exercise a joint entry deliberately; a short random trace need not
        # encounter every transition type. This is a structural stress test.
        joint = fresh(seed=20260920, rho=.1)
        events = random.Random(20260920)
        for step in range(3000):
            event, item = joint.choose(joint.propensities(), events)
            joint.attempt(event, item, events)
            if joint.transition_counts['maximum_joint_D_entry'] >= 2:
                break
        assert joint.transition_counts['maximum_joint_D_entry'] >= 2
        joint.check_state()
        export_xyz(args.output/'joint_D_GEOMETRY_ONLY.xyz', joint.blocks(), 'Joint D geometry; not physical PXRD input')
        passed('incoming_node_joint_intermediates', events=step+1,
               maximum_joint_entry=joint.transition_counts['maximum_joint_D_entry'],
               intermediate_states=joint.intermediate_record())
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        (args.output/'integration_checks.json').write_text(json.dumps(report, indent=2))


def math_exp(value):
    import math
    return math.exp(value)


if __name__ == '__main__':
    main()
