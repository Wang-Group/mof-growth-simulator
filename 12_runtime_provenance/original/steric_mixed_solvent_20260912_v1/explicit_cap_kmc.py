"""Reviewed UiO graph KMC with explicit, replaceable modulator caps.

Author-approved 2026-09-12: linked AND opened/ready pairs are cap-free.
Only genuinely free metal sites carry caps. The underlying legacy coordinate
arrays are geometric site records; physical_atoms/blocks are the ONLY physical
collision/export representation. No acetate atom enters those physical arrays.

No reverse event is sterically vetoed. A failed cap restoration ends a trace as
a MODEL REPRESENTATION FAILURE, not as zero growth or physical censoring.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import copy
import json
import math
import pickle
import random
import sys
import time

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    class cKDTree:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)

from cap_geometry_preflight import proper_fit, read_mol
from validate_replaceable_caps import entity_core_indices, export_xyz, sha


class PackingFailure(RuntimeError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def stack_xyz(blocks):
    return np.vstack([b['xyz'] for b in blocks]) if blocks else np.empty((0, 3))


def touches(xyz, tree, cutoff=2.0):
    if not len(xyz) or tree is None:
        return False
    if hasattr(tree, 'query'):
        return bool(np.any(tree.query(xyz, k=1, distance_upper_bound=cutoff)[0] < cutoff))
    # 5090's reviewed runner provides a NumPy-compatible KDTree when SciPy is
    # unavailable. Use the identical strict distance criterion, in finite blocks.
    for start in range(0, len(tree.data), 512):
        delta = xyz[:, None, :] - tree.data[None, start:start + 512, :]
        if np.any(np.sum(delta * delta, axis=2) < cutoff ** 2):
            return True
    return False


def tree_for(xyz):
    return cKDTree(xyz) if len(xyz) else None


def solve_cap_packing(slots, fixed_xyz, max_search=100000):
    """Finite-conformer CSP, not an energy minimizer or Boltzmann sampler.

    Full backtracking over feasible candidate indices; no new conformers or
    fractional cap coverage. Existing caps are fixed. This searches a collision-
    free initialization/restoration without deleting caps to obtain a fit.
    """
    tree = tree_for(fixed_xyz)
    domains = {site: [i for i, c in enumerate(candidates) if not touches(c['xyz'], tree)]
               for site, candidates in slots.items()}
    empty = [site for site in slots if not domains[site]]
    if empty:
        raise PackingFailure('no_candidate_fits_fixed_atoms', {'empty_domains': len(empty)})
    keys = list(slots)
    compatibility = {}
    searches = 0

    def compatible(a, ai, b, bi):
        key = (a, ai, b, bi)
        if key not in compatibility:
            x, y = slots[a][ai]['xyz'], slots[b][bi]['xyz']
            compatibility[key] = not touches(x, tree_for(y))
        return compatibility[key]

    def visit(remaining, selected):
        nonlocal searches
        searches += 1
        if searches > max_search:
            raise PackingFailure('packing_search_limit', {'search_nodes': searches})
        if not remaining:
            return selected
        site = min(remaining, key=lambda s: (len(remaining[s]), keys.index(s)))
        for candidate in remaining[site]:
            reduced = {}
            valid = True
            for other, choices in remaining.items():
                if other == site:
                    continue
                allowed = [k for k in choices if compatible(site, candidate, other, k)]
                if not allowed:
                    valid = False
                    break
                reduced[other] = allowed
            if valid:
                answer = visit(reduced, {**selected, site: candidate})
                if answer is not None:
                    return answer
        return None

    solution = visit(domains, {})
    if solution is None:
        raise PackingFailure('no_compatible_cap_packing', {'search_nodes': searches})
    return {s: slots[s][k] for s, k in solution.items()}


class CapLibrary:
    def __init__(self, root, identity, zero_size_control=False):
        self.root = Path(root)
        identities = json.loads((self.root / 'molecule_library/molecule_library_index.json').read_text())['identities']
        self.identity = next(i for i in identities if i['identity'] == identity)
        self.zero_size_control = zero_size_control
        d = self.identity['indices']
        self.anchor = [d['Ccarboxylate'], d['O1'], d['O2']]
        self.conformers = []
        for record in self.identity['conformers']:
            e, x, _ = read_mol(self.root / 'molecule_library' / record['mol_file'])
            self.conformers.append((record['conformer'], e, x))

    def candidates(self, site):
        target = site.belonging_entity.coordinates[site.atom_indexes]
        result = []
        for number, elem, xyz in self.conformers:
            for flip in (False, True):
                idx = ([self.anchor[0], self.anchor[2], self.anchor[1]] if flip else self.anchor)
                rot, shift, err = proper_fit(xyz[idx], target)
                result.append({'elements': [] if self.zero_size_control else elem,
                               'xyz': np.empty((0, 3)) if self.zero_size_control else xyz @ rot + shift,
                               'conformer': number, 'oxygen_swapped': flip,
                               'anchor_rmsd_A': err})
        return result


class ExplicitCapAssembly:
    def __init__(self, assembly, engine, library):
        self.assembly, self.engine, self.library = assembly, engine, library
        self.core = {}
        self.cap = {}
        self.candidates = {}
        self.identifiers = {}
        self.next_identifier = 0
        self.last_rejection = None
        for entity in assembly.entities.items:
            self.register(entity)
        slots = {s: self.candidates[s] for s in assembly.MC_free_carboxylates.items}
        self.cap = solve_cap_packing(slots, stack_xyz(self.bare_blocks()))
        self.assert_bookkeeping()

    def register(self, entity):
        if entity in self.identifiers:
            return
        self.identifiers[entity] = self.next_identifier
        self.next_identifier += 1
        if entity.entity_type == 'Zr':
            indices = entity_core_indices(entity)
            self.core[entity] = {'elements': [entity.elements[i] for i in indices],
                                 'xyz': entity.coordinates[indices].copy()}
            for site in entity.carboxylates:
                self.candidates[site] = self.library.candidates(site)
        else:
            assert list(entity.elements).count('H') == 4

    def bare_block(self, entity):
        data = (self.core[entity] if entity.entity_type == 'Zr' else
                {'elements': list(entity.elements), 'xyz': entity.coordinates})
        return {**data, 'role': 'inorganic_core' if entity.entity_type == 'Zr' else 'framework_BDC',
                'entity': self.identifiers[entity], 'site': -1}

    def bare_blocks(self):
        return [self.bare_block(e) for e in self.assembly.entities.items]

    def blocks(self, excluded_caps=()):
        excluded = set(excluded_caps)
        blocks = self.bare_blocks()
        for site, cap in self.cap.items():
            if site in excluded:
                continue
            e = site.belonging_entity
            blocks.append({**cap, 'role': 'modulator_cap', 'entity': self.identifiers[e],
                           'site': e.carboxylates.index(site)})
        return blocks

    def physical_atoms(self):
        return stack_xyz(self.blocks())

    def assert_bookkeeping(self):
        a = self.assembly
        assert set(self.cap) == set(a.MC_free_carboxylates.items)
        linked = {s for p in a.linked_carboxylate_pairs.items for s in p}
        ready = {s for p in a.ready_to_connect_carboxylate_pairs.items for s in p}
        free = set(a.free_carboxylates.items)
        assert not linked & ready and not linked & free and not ready & free
        assert len(linked) == 2 * len(a.linked_carboxylate_pairs)
        assert len(ready) == 2 * len(a.ready_to_connect_carboxylate_pairs)
        all_sites = {s for e in a.entities.items for s in e.carboxylates}
        assert all_sites == linked | ready | free
        assert all(s not in self.cap for s in linked | ready)
        assert all(len(e.carboxylates) == 12 for e in a.entities.items if e.entity_type == 'Zr')

    def forget(self, entities):
        for entity in entities:
            self.core.pop(entity, None)
            for site in entity.carboxylates:
                self.cap.pop(site, None)
                self.candidates.pop(site, None)
            self.identifiers.pop(entity, None)

    def add(self, selected_site):
        a = self.assembly
        assert selected_site in a.free_carboxylates
        new = self.engine.Zr6_AA() if selected_site.belonging_entity.entity_type != 'Zr' else self.engine.BDC()
        incoming_site = new.align_carboxylates(selected_site, visualize_steps=False)
        self.register(new)
        occupied_new, occupied_old = {incoming_site}, {selected_site}
        additional = []
        # Only genuinely free sites can become new ready pairs. No entire
        # receiving entity is exempted from collision checks.
        for old in a.free_carboxylates.items:
            if old in occupied_old or old.belonging_entity.entity_type == new.entity_type:
                continue
            for ns in new.carboxylates:
                if ns not in occupied_new and old.carboxylates_superimposed(ns):
                    additional.append((old, ns))
                    occupied_new.add(ns)
                    occupied_old.add(old)
                    break
        removed_caps = [s for s in occupied_old if s.belonging_entity.entity_type == 'Zr']
        environment = self.blocks(excluded_caps=removed_caps)
        new_bare = self.bare_block(new)
        # Include inorganic core, remaining target caps and all framework atoms.
        if touches(new_bare['xyz'], tree_for(stack_xyz(environment))):
            self.last_rejection = 'incoming_framework_or_core_contact'
            self.forget([new])
            return False
        slots = ({s: self.candidates[s] for s in new.carboxylates if s not in occupied_new}
                 if new.entity_type == 'Zr' else {})
        try:
            new_caps = solve_cap_packing(slots, stack_xyz(environment + [new_bare]))
        except PackingFailure as exc:
            if exc.reason == 'packing_search_limit':
                raise
            self.last_rejection = 'incoming_cap_' + exc.reason
            self.forget([new])
            return False
        a.add_entity(selected_site, new, incoming_site, additional, [p[1] for p in additional])
        for site in removed_caps:
            self.cap.pop(site)
        self.cap.update(new_caps)
        self.last_rejection = None
        self.assert_bookkeeping()
        return True

    def close(self, pair):
        before = set(self.cap)
        self.assembly.link_internal_carboxylate(pair)
        assert before == set(self.cap)
        self.assert_bookkeeping()

    def open(self, pair):
        a = self.assembly
        before = set(a.entities.items)
        # Every selected bond opens. No coordination-count or cap-clash veto.
        a.remove_linkage(pair)
        remaining = set(a.entities.items)
        self.forget(before - remaining)
        slots = {s: self.candidates[s] for s in a.MC_free_carboxylates.items if s not in self.cap}
        if slots:
            try:
                self.cap.update(solve_cap_packing(slots, stack_xyz(self.blocks())))
            except PackingFailure as exc:
                # DO NOT continue with missing caps, or reinterpret this as a
                # physical inability to remove this bond.
                raise PackingFailure('restoration_' + exc.reason,
                                     {**exc.details, 'newly_vacant_sites': len(slots),
                                      'entities_deleted': len(before - remaining)}) from exc
        self.assert_bookkeeping()
        return 'internal_opening' if before == remaining else 'fragment_dissociation'


def run_trace(row, root, engine, selector, output, *, wall_cap=None, event_cap=None):
    """One finite pilot trajectory, with invalid representations kept separate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_value = int(row['random_seed'])
    random.seed(seed_value)
    np.random.seed(seed_value % (2 ** 32 - 1))
    rng = random.Random(seed_value ^ 0x5EEDBEEF)
    target = int(row['N_target'])
    engine.MAX_DEPTH = target + 1
    if row['assay'] == 'growth':
        with (Path(root) / 'framework_inputs/UiO66_N76_reviewed.pkl').open('rb') as f:
            assembly = pickle.load(f)
    else:
        assembly = engine.Assembly(engine.Zr6_AA(), 1.0, math.exp(30.9), 2.0)
    initial = len(assembly.entities)
    assert initial == int(row['N_start'])
    initial_entities = set(assembly.entities.items)
    coords = selector.PhysicalCoordinates(1.0, float(row['f_Zr']), float(row['rho_off']), float(row['s_close']))
    initial_prop = selector.compute_propensities(selector.counts_from_assembly(assembly), coords)
    initial_add = initial_prop.add_node + initial_prop.add_linker
    result = {'schema': 'explicit-cap-pilot-v1', 'design': dict(row),
              'coordinates': dict(selector.chemistry_record(coords)),
              'author_approved_opened_state': 'adjacent_ready_pairs_cap_free',
              'chemical_constants_identical_across_identities': True,
              'no_bond_accessibility_rules': True,
              'input_N': initial, 'target_N': target,
              'cap_conformer_model': 'finite isolated-MMFF library; deterministic feasible packing; no energy in rates',
              'status': None, 'valid_physical_outcome': False}
    try:
        model = ExplicitCapAssembly(assembly, engine, CapLibrary(root, row['identity']))
    except PackingFailure as exc:
        result.update(status='invalid_initial_cap_packing', failure_reason=exc.reason, failure_details=exc.details,
                      executed_events=0, model_time=0.0)
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        return result
    export_xyz(output / 'initial_physical_atoms.xyz', model.blocks(), 'Explicit caps; no legacy acetate atoms')
    cap_limit = int(row['event_cap']) if event_cap is None else event_cap
    wall_limit = float(row['wall_time_cap_s']) if wall_cap is None else wall_cap
    started, clock = time.perf_counter(), 0.0
    counts = Counter()
    curve = [{'event': 0, 'time': 0.0, 'N': initial, 'bonds': len(assembly.linked_carboxylate_pairs),
              'ready_pairs': len(assembly.ready_to_connect_carboxylate_pairs), 'caps': len(model.cap)}]
    passage = {}
    maximum = initial
    executed = 0
    status = 'event_cap'
    interval = max(1, cap_limit // 500)
    for event in range(1, cap_limit + 1):
        if time.perf_counter() - started >= wall_limit:
            status = 'wall_time_cap'
            break
        chosen = selector.select_event(assembly, coords, rng)
        clock += selector.gillespie_waiting_time(chosen.propensities.total, rng)
        counts[chosen.event + '_attempt'] += 1
        try:
            if chosen.event in {'add_node', 'add_linker'}:
                accepted = model.add(chosen.selected_site)
                counts[chosen.event + ('_accepted' if accepted else '_rejected')] += 1
                if not accepted:
                    counts[model.last_rejection] += 1
            elif chosen.event == 'close_internal':
                model.close(chosen.selected_pair)
                counts['close_internal_accepted'] += 1
            else:
                counts[model.open(chosen.selected_pair)] += 1
        except PackingFailure as exc:
            status = ('invalid_cap_restoration' if exc.reason.startswith('restoration_')
                      else 'invalid_geometry_search')
            result.update(failure_reason=exc.reason, failure_details=exc.details)
            executed = event
            break
        executed = event
        n = len(assembly.entities)
        if n > maximum:
            for level in range(maximum + 1, n + 1):
                passage[str(level)] = clock
            maximum = n
        if event <= 100 or event % interval == 0 or n >= target:
            curve.append({'event': event, 'time': clock, 'N': n,
                          'bonds': len(assembly.linked_carboxylate_pairs),
                          'ready_pairs': len(assembly.ready_to_connect_carboxylate_pairs),
                          'caps': len(model.cap)})
        if n >= target:
            status = 'target_reached'
            break
    valid = not status.startswith('invalid_')
    result.update(status=status, valid_physical_outcome=valid, executed_events=executed,
                  model_time=clock, wall_seconds=time.perf_counter() - started,
                  event_cap=cap_limit, wall_time_cap_s=wall_limit,
                  final_N=len(assembly.entities), maximum_N=maximum,
                  final_bonds=len(assembly.linked_carboxylate_pairs),
                  final_caps=len(model.cap), event_counts=dict(counts),
                  seed_entities_retained=len(initial_entities & set(assembly.entities.items)),
                  first_passage_model_time=passage, sampled_growth_curve=curve,
                  coefficient_fixed_during_trace=True,
                  initial_total_removal_to_external_addition=(initial_prop.remove_bond / initial_add
                                                             if initial_add else None))
    # Store actual opportunity ratios, distinct from the fixed rho_off input.
    fc = selector.counts_from_assembly(assembly)
    prop = selector.compute_propensities(fc, coords)
    denominator = prop.add_node + prop.add_linker
    result['final_total_removal_to_external_addition'] = prop.remove_bond / denominator if denominator else None
    if valid:
        model.assert_bookkeeping()
        export_xyz(output / 'final_physical_atoms.xyz', model.blocks(), 'Valid explicit-cap final structure')
    else:
        export_xyz(output / 'INVALID_restoration_state.xyz', model.blocks(), 'INVALID: cap restoration incomplete; do not use as physical result')
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    return result
