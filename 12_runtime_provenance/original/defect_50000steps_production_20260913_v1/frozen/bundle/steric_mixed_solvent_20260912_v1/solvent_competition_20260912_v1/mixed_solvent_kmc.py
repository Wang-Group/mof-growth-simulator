"""Mixed solvent/modulator physical-geometry layer for the reviewed UiO KMC.

Caps are explicitly exchanged on the common opening clock. Site initialization
and retermination use product-prior sampling conditional on hard-core feasibility,
not a first-feasible packing or a solvent fallback. Framework bond removal is
never vetoed. A finite sampler failure is an invalid record, not zero growth.
"""
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from explicit_cap_kmc import (CapLibrary, ExplicitCapAssembly, PackingFailure,
                              stack_xyz, touches, tree_for)
from validate_replaceable_caps import export_xyz
from solvent_channel import SiteCompetition, on_metal_site


class WeightedCapSampler:
    """Exact conditional product-prior draws, with a finite rejection cap.

    Filter each domain against fixed atoms; factor by the remaining pair-contact
    graph; reject whole component proposals. This conditioning does not change
    relative prior weights of feasible configurations. It is not MCMC or greedy
    sequential packing. Trial exhaustion is reported, never replaced by all S.
    """
    def __init__(self, slots, fixed_xyz, max_trials=100000):
        self.keys = list(slots)
        self.max_trials = max_trials
        if max_trials < 1:
            raise ValueError('max_trials must be positive')
        fixed_tree = tree_for(fixed_xyz)
        self.domains, self.probabilities = [], []
        for key in self.keys:
            candidates = slots[key]
            if any(not math.isfinite(c['prior']) or c['prior'] < 0 for c in candidates):
                raise ValueError('Invalid candidate prior')
            domain = [c for c in candidates if c['prior'] > 0
                      and not touches(c['xyz'], fixed_tree)]
            if not domain:
                raise PackingFailure('no_allowed_termination', {'site_index': len(self.domains)})
            weights = np.array([c['prior'] for c in domain], dtype=float)
            self.domains.append(domain)
            self.probabilities.append(weights / weights.sum())
        self.edges = {}
        adjacency = {i: set() for i in range(len(self.keys))}
        bounds = []
        for domain in self.domains:
            xyz = np.vstack([c['xyz'] for c in domain])
            bounds.append((xyz.min(0), xyz.max(0)))
        for i, di in enumerate(self.domains):
            for j in range(i):
                gap = np.maximum(np.maximum(bounds[i][0] - bounds[j][1],
                                            bounds[j][0] - bounds[i][1]), 0)
                if np.dot(gap, gap) >= 4.0:
                    continue
                dj = self.domains[j]
                incompatible = np.array([[touches(a['xyz'], tree_for(b['xyz']))
                                          for b in dj] for a in di], dtype=bool)
                if incompatible.any():
                    self.edges[(i, j)] = incompatible
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        self.components = []
        remaining = set(adjacency)
        while remaining:
            seed = min(remaining)
            component, queue = set(), [seed]
            while queue:
                k = queue.pop()
                if k in component:
                    continue
                component.add(k)
                queue.extend(adjacency[k] - component)
            self.components.append(sorted(component))
            remaining -= component
        self.diagnostics = {'component_sizes': [len(c) for c in self.components],
                            'sample_calls': 0, 'proposals_tested': 0}

    def sample(self, rng):
        answer = {}
        for component in self.components:
            edges = [(i, j, m) for (i, j), m in self.edges.items()
                     if i in component and j in component]
            trials = 0
            while trials < self.max_trials:
                batch = 1 if not edges else min(128, self.max_trials - trials)
                draws = {k: rng.choice(len(self.domains[k]), size=batch,
                                       p=self.probabilities[k]) for k in component}
                valid = np.ones(batch, dtype=bool)
                for i, j, matrix in edges:
                    valid &= ~matrix[draws[i], draws[j]]
                positions = np.flatnonzero(valid)
                if len(positions):
                    first = int(positions[0])
                    trials += first + 1
                    for k in component:
                        answer[self.keys[k]] = self.domains[k][int(draws[k][first])]
                    break
                trials += batch
            else:
                raise PackingFailure('mixed_sampling_limit', {'component_size': len(component),
                                                               'proposals': trials})
            self.diagnostics['proposals_tested'] += trials
        self.diagnostics['sample_calls'] += 1
        return answer


class MixedCapAssembly(ExplicitCapAssembly):
    def __init__(self, assembly, engine, library, bath, np_rng):
        self.assembly, self.engine, self.library = assembly, engine, library
        self.bath, self.np_rng = bath, np_rng
        self.core, self.cap, self.candidates, self.identifiers = {}, {}, {}, {}
        self.next_identifier = 0
        self.last_rejection = None
        self.sampling = Counter()
        for entity in assembly.entities.items:
            self.register(entity)
        self.cap = self.sample_terminations(assembly.MC_free_carboxylates.items,
                                            stack_xyz(self.bare_blocks()), 'initial')
        self.assert_bookkeeping()

    def register(self, entity):
        if entity in self.identifiers:
            return
        super().register(entity)
        if entity.entity_type == 'Zr':
            eta = self.bath.solvent_fraction
            for site in entity.carboxylates:
                modulators = self.candidates[site]
                self.candidates[site] = [dict(on_metal_site(site), kind='S', prior=eta)] + [
                    dict(c, kind='M', prior=(1 - eta) / len(modulators), geometry_only=False)
                    for c in modulators]

    def sample_terminations(self, sites, fixed, context):
        sampler = WeightedCapSampler({s: self.candidates[s] for s in sites}, fixed)
        selected = sampler.sample(self.np_rng)
        self.sampling[context + '_calls'] += 1
        self.sampling[context + '_component_proposals'] += sampler.diagnostics['proposals_tested']
        return selected

    def blocks(self, excluded_caps=()):
        excluded = set(excluded_caps)
        result = self.bare_blocks()
        for site, cap in self.cap.items():
            if site not in excluded:
                entity = site.belonging_entity
                result.append(dict(cap, role='solvent_proxy' if cap['kind'] == 'S' else 'modulator_cap',
                                   entity=self.identifiers[entity], site=entity.carboxylates.index(site)))
        return result

    def framework_blocks(self):
        """The only default scattering export: framework, with all caps excluded."""
        return self.bare_blocks()

    def solvent_count(self):
        return sum(c['kind'] == 'S' for c in self.cap.values())

    def exchange(self, site, target_kind):
        if site not in self.cap or target_kind not in {'S', 'M'}:
            raise ValueError('Exchange requires a terminated free metal site')
        if self.cap[site]['kind'] == target_kind:
            raise ValueError('Exchange target must differ from source kind')
        options = [c for c in self.candidates[site] if c['kind'] == target_kind]
        # Full M prior, NOT renormalized over only the feasible conformers.
        proposed = options[int(self.np_rng.integers(len(options)))]
        if touches(proposed['xyz'], tree_for(stack_xyz(self.blocks(excluded_caps=[site])))):
            return False
        self.cap[site] = proposed
        self.assert_bookkeeping()
        return True

    def add(self, selected_site):
        a = self.assembly
        assert selected_site in a.free_carboxylates
        new = self.engine.Zr6_AA() if selected_site.belonging_entity.entity_type != 'Zr' else self.engine.BDC()
        incoming_site = new.align_carboxylates(selected_site, visualize_steps=False)
        self.register(new)
        # Draw a bath node's FULL pre-encounter termination configuration against
        # its own core only. Do not optimize caps against the receiving crystal.
        incoming_caps = (self.sample_terminations(new.carboxylates, self.core[new]['xyz'], 'incoming_bath')
                         if new.entity_type == 'Zr' else {})
        occupied_new, occupied_old = {incoming_site}, {selected_site}
        additional = []
        for old in a.free_carboxylates.items:
            if old in occupied_old or old.belonging_entity.entity_type == new.entity_type:
                continue
            for ns in new.carboxylates:
                if ns not in occupied_new and old.carboxylates_superimposed(ns):
                    additional.append((old, ns))
                    occupied_new.add(ns)
                    occupied_old.add(old)
                    break
        removed = [s for s in occupied_old if s.belonging_entity.entity_type == 'Zr']
        environment = self.blocks(excluded_caps=removed)
        new_caps = {s: c for s, c in incoming_caps.items() if s not in occupied_new}
        environment_tree = tree_for(stack_xyz(environment))
        bare = self.bare_block(new)
        if touches(bare['xyz'], environment_tree):
            self.last_rejection = 'incoming_framework_or_core_contact'
            self.forget([new])
            return False
        if any(touches(c['xyz'], environment_tree) for c in new_caps.values()):
            self.last_rejection = 'incoming_termination_contact'
            self.forget([new])
            return False
        a.add_entity(selected_site, new, incoming_site, additional, [p[1] for p in additional])
        for s in removed:
            self.cap.pop(s)
        self.cap.update(new_caps)
        self.last_rejection = None
        self.assert_bookkeeping()
        return True

    def open(self, pair):
        a = self.assembly
        before = set(a.entities.items)
        a.remove_linkage(pair)  # Unconditional, uniform selected bond removal.
        remaining = set(a.entities.items)
        self.forget(before - remaining)
        vacant = [s for s in a.MC_free_carboxylates.items if s not in self.cap]
        if vacant:
            try:
                self.cap.update(self.sample_terminations(vacant, stack_xyz(self.blocks()), 'retermination'))
            except PackingFailure as exc:
                raise PackingFailure('retermination_' + exc.reason,
                                     dict(exc.details, newly_vacant_sites=len(vacant))) from exc
        self.assert_bookkeeping()
        return 'internal_opening' if before == remaining else 'fragment_dissociation'


def all_propensities(model, coordinates, selector):
    p = selector.compute_propensities(selector.counts_from_assembly(model.assembly), coordinates)
    rates = model.bath.cap_exchange_rates()
    n_s = model.solvent_count()
    return {'add_node': p.add_node, 'add_linker': p.add_linker,
            'close_internal': p.close_internal, 'remove_bond': p.remove_bond,
            'solvent_to_modulator': n_s * rates['solvent_to_modulator'],
            'modulator_to_solvent': (len(model.cap) - n_s) * rates['modulator_to_solvent']}


def choose_event(model, propensities, rng):
    a = model.assembly
    draw = rng.random() * sum(propensities.values())
    active = [(k, v) for k, v in propensities.items() if v > 0]
    if not active:
        raise RuntimeError('No enabled event')
    event = active[-1][0]
    for name, rate in active:
        if draw < rate:
            event = name
            break
        draw -= rate
    pools = {'add_node': a.Linker_free_carboxylates.items,
             'add_linker': a.MC_free_carboxylates.items,
             'close_internal': a.ready_to_connect_carboxylate_pairs.items,
             'remove_bond': a.linked_carboxylate_pairs.items}
    if event in pools:
        return event, rng.choice(pools[event])
    source = 'S' if event == 'solvent_to_modulator' else 'M'
    return event, rng.choice([s for s, c in model.cap.items() if c['kind'] == source])


def run_trace(row, root, engine, selector, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start_wall = time.perf_counter()
    seed = int(row['random_seed'])
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    event_rng = random.Random(seed ^ 0x5EEDBEEF)
    cap_rng = np.random.default_rng(seed ^ 0xCA9CA9)
    target = int(row['N_target'])
    engine.MAX_DEPTH = target + 1
    if row['assay'] == 'growth':
        with (Path(root) / 'framework_inputs/UiO66_N76_reviewed.pkl').open('rb') as f:
            assembly = pickle.load(f)
    else:
        assembly = engine.Assembly(engine.Zr6_AA(), 1.0, math.exp(30.9), 2.0)
    initial = len(assembly.entities)
    assert initial == int(row['N_start'])
    coordinates = selector.PhysicalCoordinates(1., float(row['f_Zr']), float(row['rho_off']), float(row['s_close']))
    bath = SiteCompetition.from_rho_fraction(float(row['rho_off']), float(row['eta_S']))
    result = {'schema': 'mixed-solvent-kmc-v1', 'design': dict(row),
              'coordinates': dict(selector.chemistry_record(coordinates)),
              'bath_opening_outcomes': bath.opening_outcomes,
              'uncrowded_eta_S': bath.solvent_fraction,
              'cap_exchange_per_site_rates': bath.cap_exchange_rates(),
              'uniform_framework_bond_removal': True,
              'opened_ready_pairs_cap_free': True,
              'cap_sampling': 'exact product-prior conditional on fixed geometry; component rejection',
              'status': None, 'valid_physical_outcome': False}
    try:
        model = MixedCapAssembly(assembly, engine, CapLibrary(root, row['identity']), bath, cap_rng)
    except PackingFailure as exc:
        result.update(status='invalid_initial_geometry', failure_reason=exc.reason,
                      failure_details=exc.details, executed_events=0, model_time=0.,
                      final_N=initial, wall_seconds=time.perf_counter() - start_wall)
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        return result
    export = int(row['replicate']) == 0
    if export:
        export_xyz(output / 'initial_GEOMETRY_ONLY.xyz', model.blocks(), 'Contains solvent proxies: NOT physical PXRD input')
    maximum, clock, executed = initial, 0., 0
    counts, passage, curve = Counter(), {}, []
    occupancy_integral = 0.
    next_sample = 0.
    time_limit = float(row['model_time_limit'])
    sample_dt = time_limit / 200
    status = 'event_cap'
    initial_s = model.solvent_count()
    initial_caps = len(model.cap)
    initial_prop = all_propensities(model, coordinates, selector)

    def snapshot():
        return {'time': clock, 'event': executed, 'N': len(assembly.entities),
                'bonds': len(assembly.linked_carboxylate_pairs),
                'ready_pairs': len(assembly.ready_to_connect_carboxylate_pairs),
                'solvent_caps': model.solvent_count(), 'modulator_caps': len(model.cap) - model.solvent_count()}

    curve.append(snapshot())
    for event_index in range(1, int(row['event_cap']) + 1):
        if time.perf_counter() - start_wall >= float(row['wall_time_cap_s']):
            status = 'wall_time_cap'
            break
        prop = all_propensities(model, coordinates, selector)
        total = sum(prop.values())
        dt = selector.gillespie_waiting_time(total, event_rng)
        dt_used = min(dt, max(0., time_limit - clock))
        occupancy_integral += dt_used * (model.solvent_count() / len(model.cap) if model.cap else 0.)
        if clock + dt > time_limit:
            clock = time_limit
            status = 'observation_window'
            break
        clock += dt
        event, item = choose_event(model, prop, event_rng)
        counts[event + '_attempt'] += 1
        try:
            if event in {'add_node', 'add_linker'}:
                accepted = model.add(item)
                counts[event + ('_accepted' if accepted else '_rejected')] += 1
                if not accepted:
                    counts[model.last_rejection] += 1
            elif event == 'close_internal':
                model.close(item)
                counts['close_internal_accepted'] += 1
            elif event == 'remove_bond':
                counts[model.open(item)] += 1
            else:
                accepted = model.exchange(item, 'M' if event == 'solvent_to_modulator' else 'S')
                counts[event + ('_accepted' if accepted else '_rejected')] += 1
        except PackingFailure as exc:
            status = 'invalid_dynamic_geometry'
            result.update(failure_reason=exc.reason, failure_details=exc.details)
            executed = event_index
            break
        executed = event_index
        n = len(assembly.entities)
        if n > maximum:
            for level in range(maximum + 1, n + 1):
                passage[str(level)] = clock
            maximum = n
        if clock >= next_sample or n >= target:
            curve.append(snapshot())
            next_sample = clock + sample_dt
        if n >= target:
            status = 'target_reached'
            break
    curve.append(snapshot())
    valid = not status.startswith('invalid_')
    final_prop = all_propensities(model, coordinates, selector)
    def off_to_add(prop):
        denominator = prop['add_node'] + prop['add_linker']
        return prop['remove_bond'] / denominator if denominator else None
    result.update(status=status, valid_physical_outcome=valid,
                  input_N=initial, target_N=target, final_N=len(assembly.entities), maximum_N=maximum,
                  executed_events=executed, model_time=clock, event_counts=dict(counts),
                  first_passage_model_time=passage, sampled_growth_curve=curve,
                  initial_solvent_caps=initial_s, initial_total_caps=initial_caps,
                  final_solvent_caps=model.solvent_count(), final_total_caps=len(model.cap),
                  time_average_solvent_fraction=occupancy_integral / clock if clock else None,
                  initial_total_removal_to_addition=off_to_add(initial_prop),
                  final_total_removal_to_addition=off_to_add(final_prop),
                  fixed_rho_off=float(row['rho_off']), sampling_diagnostics=dict(model.sampling),
                  wall_seconds=time.perf_counter() - start_wall)
    if valid:
        model.assert_bookkeeping()
    if export or not valid:
        prefix = 'final' if valid else 'INVALID'
        export_xyz(output / (prefix + '_GEOMETRY_ONLY.xyz'), model.blocks(),
                   'Contains solvent proxies; NOT physical PXRD input; status=' + status)
        if valid:
            export_xyz(output / 'final_FRAMEWORK_ONLY.xyz', model.framework_blocks(),
                       'Framework-only scattering input; excludes ALL solvent/modulator caps')
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    return result
