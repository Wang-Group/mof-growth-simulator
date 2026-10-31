"""Transactional molecular F-D-V candidate with explicit physical intermediates.

Import after runtime.configure(). Canonical graph coordinates and physical D
coordinates are deliberately separate. No degree-dependent rate coefficients.
Prior proposals, geometry rejection and energy rejection are recorded separately.
"""
from collections import Counter
import copy
import math

import numpy as np

from mixed_solvent_kmc import MixedCapAssembly
from explicit_cap_kmc import stack_xyz, tree_for, touches, PackingFailure
from preflight_crowded_geometry import rotate_about_axis

ANGLES = (-40., -30., -20., -10., 10., 20., 30., 40.)


def ordered_pair(pair):
    return sorted(pair, key=lambda s: s.belonging_entity.entity_type != 'Zr')


def clone_fields(obj):
    result = {}
    for key, value in obj.__dict__.items():
        if hasattr(value, 'item_to_index'):
            cloned = copy.copy(value)
            cloned.items, cloned.item_to_index = list(value.items), dict(value.item_to_index)
            result[key] = cloned
        elif isinstance(value, list):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = {k: list(v) if isinstance(v, list) else v for k, v in value.items()}
        else:
            result[key] = value
    return result


class GraphTransaction:
    """Copy graph containers, not all atomic arrays; support exact rejection undo."""
    def __init__(self, model):
        self.model = model
        self.assembly = clone_fields(model.assembly)
        self.entities = [(e, clone_fields(e)) for e in model.assembly.entities.items]
        self.sites = [(s, dict(s.__dict__)) for e, _ in self.entities for s in e.carboxylates]
        self.maps = {name: dict(getattr(model, name)) for name in
                     ('core', 'cap', 'candidates', 'identifiers', 'intermediates')}
        self.next_identifier = model.next_identifier
        self.ledger_fields = copy.deepcopy(model.ledger.__dict__)

    def restore(self):
        self.model.assembly.__dict__.clear()
        self.model.assembly.__dict__.update(self.assembly)
        for obj, state in self.entities + self.sites:
            obj.__dict__.clear()
            obj.__dict__.update(state)
        for name, value in self.maps.items():
            setattr(self.model, name, value)
        self.model.next_identifier = self.next_identifier
        self.model.ledger.__dict__.clear()
        self.model.ledger.__dict__.update(self.ledger_fields)


class DefectAssembly(MixedCapAssembly):
    def __init__(self, assembly, engine, library, bath, np_rng, ledger, delta):
        if not math.isfinite(delta) or delta < 0:
            raise ValueError('delta must be finite and nonnegative')
        self.delta, self.ledger = float(delta), ledger
        self.intermediates = {}
        self.in_transition = False
        self.transition_counts = Counter()
        self.last_transition = None
        self._close_candidates = None
        super().__init__(assembly, engine, library, bath, np_rng)
        if len(assembly.ready_to_connect_carboxylate_pairs):
            raise ValueError('Initial seed must have no undefined cap-free ready pairs')
        self.check_state()

    def bare_block(self, entity):
        block = super().bare_block(entity)
        if entity in self.intermediates:
            block = dict(block, xyz=self.intermediates[entity]['xyz'], role='displaced_BDC')
        return block

    def assert_bookkeeping(self):
        if self.in_transition:
            return
        a = self.assembly
        linked = {s for p in a.linked_carboxylate_pairs.items for s in p}
        ready = {s for p in a.ready_to_connect_carboxylate_pairs.items for s in p}
        free = set(a.free_carboxylates.items)
        assert not linked & ready and not linked & free and not ready & free
        assert len(linked) == 2*len(a.linked_carboxylate_pairs)
        assert len(ready) == 2*len(a.ready_to_connect_carboxylate_pairs)
        assert linked | ready | free == {s for e in a.entities.items for s in e.carboxylates}
        metal_ready = {ordered_pair(p)[0] for p in a.ready_to_connect_carboxylate_pairs.items}
        assert set(self.cap) == set(a.MC_free_carboxylates.items) | metal_ready
        assert not set(self.cap) & linked
        assert len(self.intermediates) == len(a.ready_to_connect_carboxylate_pairs)
        for linker, d in self.intermediates.items():
            assert d['pair'] in a.ready_to_connect_carboxylate_pairs
            assert len(linker.connected_entities) == 1
            assert d['metal_site'] in self.cap
        assert all(len(e.carboxylates) == 12 for e in a.entities.items if e.entity_type == 'Zr')

    def check_state(self):
        self.assert_bookkeeping()
        self.ledger.check(Counter(e.entity_type for e in self.assembly.entities.items))
        for pool_name in ('entities', 'free_carboxylates', 'MC_free_carboxylates',
                          'Linker_free_carboxylates', 'linked_carboxylate_pairs',
                          'ready_to_connect_carboxylate_pairs'):
            pool = getattr(self.assembly, pool_name)
            assert len(pool.items) == len(pool.item_to_index)
            assert all(pool.item_to_index[v] == i for i, v in enumerate(pool.items))
        for p in self.assembly.linked_carboxylate_pairs.items:
            x, y = [s.belonging_entity for s in p]
            assert y in x.connected_entities and x in y.connected_entities
        assert all(Counter(e.elements).get('H') == 4 for e in self.assembly.entities.items
                   if e.entity_type == 'Ligand')

    def block_map(self):
        result = {}
        for e in self.assembly.entities.items:
            result[('entity', e)] = self.bare_block(e)
        for s, cap in self.cap.items():
            e = s.belonging_entity
            result[('cap', s)] = dict(cap, role='solvent_proxy' if cap['kind'] == 'S' else 'modulator_cap',
                                     entity=self.identifiers[e], site=e.carboxylates.index(s))
        return result

    def draw_cap(self, site):
        choices = self.candidates[site]
        weights = np.array([c['prior'] for c in choices])
        assert abs(weights.sum() - 1) < 1e-12
        return choices[int(self.np_rng.choice(len(choices), p=weights))]

    def _graph_add(self, selected):
        a = self.assembly
        assert selected in a.free_carboxylates
        new = self.engine.Zr6_AA() if selected.belonging_entity.entity_type != 'Zr' else self.engine.BDC()
        incoming = new.align_carboxylates(selected, visualize_steps=False)
        self.register(new)
        # Reference is the incoming node with its selected reacting site opened.
        # Draw each remaining cap ONCE from its prior; reject the whole joint
        # geometry later. Conditioning a fully capped node first and discarding
        # the selected cap introduces an uncancelled packing-normalization factor.
        incoming_caps = ({s: self.draw_cap(s) for s in new.carboxylates if s is not incoming}
                         if new.entity_type == 'Zr' else {})
        used_old, used_new, additional = {selected}, {incoming}, []
        for old in a.free_carboxylates.items:
            if old in used_old or old.belonging_entity.entity_type == new.entity_type:
                continue
            for ns in new.carboxylates:
                if ns not in used_new and old.carboxylates_superimposed(ns):
                    additional.append((old, ns))
                    used_old.add(old)
                    used_new.add(ns)
                    break
        a.add_entity(selected, new, incoming, additional, [p[1] for p in additional])
        # Only the primary linked pair is cap-free. Additional ready pairs retain caps.
        if selected.belonging_entity.entity_type == 'Zr':
            self.cap.pop(selected, None)
        incoming_caps.pop(incoming, None)
        self.cap.update(incoming_caps)
        return new

    def _refresh_physical_state(self):
        a = self.assembly
        ready = list(a.ready_to_connect_carboxylate_pairs.items)
        target = {}
        for pair in ready:
            ms, ls = ordered_pair(pair)
            linker = ls.belonging_entity
            if len(linker.connected_entities) != 1 or linker in target:
                raise PackingFailure('unsupported_intermediate_topology', {'degree': len(linker.connected_entities)})
            previous = self.intermediates.get(linker)
            if previous is not None and previous['pair'] == pair:
                target[linker] = previous
                continue
            attached = next(s for s in linker.carboxylates if s is not ls)
            anchor_xyz = linker.coordinates[attached.atom_indexes]
            angle = float(self.np_rng.choice(ANGLES))
            xyz = rotate_about_axis(linker.coordinates, anchor_xyz[1], anchor_xyz[2]-anchor_xyz[1], angle)
            error = float(np.linalg.norm(xyz[attached.atom_indexes] - anchor_xyz, axis=1).max())
            target[linker] = {'pair': pair, 'metal_site': ms, 'linker_site': ls,
                              'angle': angle, 'xyz': xyz, 'attached_error_A': error,
                              'angle_prior': 1/len(ANGLES)}
        required = list(a.MC_free_carboxylates.items) + [ordered_pair(p)[0] for p in ready]
        required_set = set(required)
        self.cap = {s: c for s, c in self.cap.items() if s in required_set}
        for site in required:
            if site not in self.cap:
                self.cap[site] = self.draw_cap(site)
        self.intermediates = target

    def _geometry_valid(self, before):
        if any(d['attached_error_A'] > .5 for d in self.intermediates.values()):
            return False, 'retained_attachment_displacement'
        after = self.block_map()
        changed = [v for k, v in after.items() if k not in before
                   or v['elements'] != before[k]['elements']
                   or not np.array_equal(v['xyz'], before[k]['xyz'])]
        changed_keys = {k for k, v in after.items() if k not in before
                        or v['elements'] != before[k]['elements']
                        or not np.array_equal(v['xyz'], before[k]['xyz'])}
        fixed = [v for k, v in after.items() if k not in changed_keys]
        tree = tree_for(stack_xyz(fixed))
        for i, block in enumerate(changed):
            if touches(block['xyz'], tree):
                return False, 'physical_interfragment_contact'
            for other in changed[:i]:
                if touches(block['xyz'], tree_for(other['xyz'])):
                    return False, 'joint_candidate_contact'
        return True, None

    def attempt(self, event, item, rng, *, apply_energy=True):
        """One prior proposal; reject atomically, never resample until acceptance.

        apply_energy=False is for declared structural unit tests only. Runners
        always use the default. There is no per-node accessibility multiplier.
        """
        transaction = GraphTransaction(self)
        before_blocks = self.block_map()
        before_entities = list(self.assembly.entities.items)
        before_types = {e: e.entity_type for e in before_entities}
        old_nD = len(self.intermediates)
        before_thermo = self.thermo_record()
        old_caps = dict(self.cap)
        old_D = dict(self.intermediates)
        new = None
        self.in_transition = True
        try:
            if event in {'add_node', 'add_linker'}:
                species = 'Zr' if event == 'add_node' else 'Ligand'
                if self.ledger.free[species] == 0:
                    raise ValueError('An exhausted addition channel was selected')
                new = self._graph_add(item)
                assert new.entity_type == species
            elif event == 'remove_bond':
                self.assembly.remove_linkage(item)
                remaining = set(self.assembly.entities.items)
                self.forget(e for e in before_entities if e not in remaining)
            elif event == 'close_internal':
                self.assembly.link_internal_carboxylate(item)
            elif event in {'solvent_to_modulator', 'modulator_to_solvent'}:
                target_kind = 'M' if event == 'solvent_to_modulator' else 'S'
                choices = [c for c in self.candidates[item] if c['kind'] == target_kind]
                self.cap[item] = choices[int(self.np_rng.integers(len(choices)))]
            else:
                raise ValueError(event)
            self._refresh_physical_state()
            new_nD = len(self.intermediates)
            energy_factor = math.exp(-max(0., self.delta*(new_nD-old_nD)))
            valid, reason = self._geometry_valid(before_blocks)
            self.last_transition = {'event': event, 'nD_before': old_nD, 'nD_proposed': new_nD,
                                    'energy_factor': energy_factor, 'geometry_valid': valid}
            if valid and apply_energy and rng.random() >= energy_factor:
                valid, reason = False, 'intermediate_energy'
            if not valid:
                transaction.restore()
                self.last_transition.update(accepted=False, rejection=reason)
                self.transition_counts[event + '_rejected_' + reason] += 1
                return False
            remaining = set(self.assembly.entities.items)
            removed = [before_types[e] for e in before_entities if e not in remaining]
            if new is not None:
                self.ledger.incorporate(new.entity_type)
            if removed:
                self.ledger.release(removed)
            self.last_transition.update(accepted=True, removed_species=removed,
                                        added_species=new.entity_type if new else None)
            if apply_energy and len(removed) <= 1:
                check = self.check_transition_balance(event, before_thermo, old_caps, old_D,
                                                      new.entity_type if new else None, removed)
                self.last_transition['balance_check'] = check
                self.transition_counts['balanced_elementary_transitions_checked'] += 1
            elif len(removed) > 1:
                self.transition_counts['open_boundary_fragment_losses'] += 1
            self.transition_counts[event + '_accepted'] += 1
            self.transition_counts['new_D_states'] += max(0, new_nD-old_nD)
            self.transition_counts['D_states_left'] += max(0, old_nD-new_nD)
            self.transition_counts['maximum_joint_D_entry'] = max(
                self.transition_counts['maximum_joint_D_entry'], new_nD-old_nD)
            self._close_candidates = None
            return True
        except Exception:
            transaction.restore()
            raise
        finally:
            self.in_transition = False
            self.check_state()

    def closure_candidates(self):
        """Exactly thin deterministic zero-acceptance closure attempts.

        A closure restores one canonical BDC and removes its receiving cap.
        Nothing else moves or is sampled. Test that precise target geometry,
        rather than spending ~exp(30.9) null attempts on a blocked closure.
        This does not change any nonzero transition rate or add an energy rule.
        """
        if self._close_candidates is None:
            candidates = []
            for linker, d in self.intermediates.items():
                environment = [block for k, block in self.block_map().items()
                               if k not in {('entity', linker), ('cap', d['metal_site'])}]
                if not touches(linker.coordinates, tree_for(stack_xyz(environment))):
                    candidates.append(d['pair'])
            self._close_candidates = candidates
        return self._close_candidates

    def propensities(self):
        a = self.assembly
        c = self.ledger.coefficients()
        n_s = self.solvent_count()
        sizes = {'add_node': len(a.Linker_free_carboxylates), 'add_linker': len(a.MC_free_carboxylates),
                 'remove_bond': len(a.linked_carboxylate_pairs),
                 'close_internal': len(self.closure_candidates()),
                 'solvent_to_modulator': n_s, 'modulator_to_solvent': len(self.cap)-n_s}
        return {name: n*c[name] for name, n in sizes.items()}

    def choose(self, prop, rng):
        draw = rng.random()*sum(prop.values())
        active = [(k, v) for k, v in prop.items() if v > 0]
        if not active:
            raise ValueError('No positive proposal propensity')
        event = active[-1][0]
        for name, value in active:
            if draw < value:
                event = name
                break
            draw -= value
        a = self.assembly
        pools = {'add_node': a.Linker_free_carboxylates.items, 'add_linker': a.MC_free_carboxylates.items,
                 'remove_bond': a.linked_carboxylate_pairs.items,
                 'close_internal': self.closure_candidates()}
        if event in pools:
            return event, rng.choice(pools[event])
        source = 'S' if event == 'solvent_to_modulator' else 'M'
        return event, rng.choice([s for s, cap in self.cap.items() if cap['kind'] == source])

    def intermediate_record(self):
        return [{'linker_id': self.identifiers[e], 'receiving_node_id': self.identifiers[d['metal_site'].belonging_entity],
                 'angle_degrees': d['angle'], 'cap_kind': self.cap[d['metal_site']]['kind'],
                 'attached_error_A': d['attached_error_A'], 'angle_prior': d['angle_prior']}
                for e, d in self.intermediates.items()]

    def thermo_record(self):
        a = self.assembly
        n, b = len(a.entities), len(a.linked_carboxylate_pairs)
        return {'tracked': dict(self.ledger.tracked), 'free': dict(self.ledger.free),
                'bonds': b, 'cycle_rank': b-n+1, 'nD': len(self.intermediates),
                'cap_log_measure': sum(math.log(c['prior']) for c in self.cap.values()),
                'D_log_measure': sum(math.log(d['angle_prior']) for d in self.intermediates.values())}

    def check_transition_balance(self, event, before, old_caps, old_D, added, removed):
        """Check actual elementary graph changes against one declared state measure.

        Weight ~ reservoir factorial/activity * rho^-b * E^(b-N+1)
                 * product(cap priors) * product(hinge priors) * exp(-delta*nD).
        Only monomer transfers and internal/cap events have a modeled reverse.
        Oligomer loss is explicitly outside this closed local check.
        Molecularly equivalent attachment labels use the common entity activity
        convention; this is not a continuum orientational-partition calculation.
        """
        after = self.thermo_record()
        c = self.ledger.coefficients()
        log_measure = -(after['bonds']-before['bonds'])*math.log(self.ledger.rho0)
        log_measure += (after['cycle_rank']-before['cycle_rank'])*30.9*self.ledger.s_close
        log_measure += after['cap_log_measure']-before['cap_log_measure']
        log_measure += after['D_log_measure']-before['D_log_measure']
        du = self.delta*(after['nD']-before['nD'])
        log_measure -= du
        total0 = sum(self.ledger.initial_free.values())
        for species in ('Zr', 'Ligand'):
            dn = after['tracked'][species]-before['tracked'][species]
            if self.ledger.mode == 'fixed':
                log_measure += dn*math.log(self.ledger.initial_free[species]/total0)
            else:
                log_measure += (math.lgamma(before['free'][species]+1)
                                - math.lgamma(after['free'][species]+1)-dn*math.log(total0))
        qf = sum(math.log(cap['prior']) for site, cap in self.cap.items() if site not in old_caps)
        qr = sum(math.log(cap['prior']) for site, cap in old_caps.items() if site not in self.cap)
        qf += sum(math.log(d['angle_prior']) for e, d in self.intermediates.items()
                  if e not in old_D or d['pair'] != old_D[e]['pair'])
        qr += sum(math.log(d['angle_prior']) for e, d in old_D.items()
                  if e not in self.intermediates or d['pair'] != self.intermediates[e]['pair'])
        if added:
            channel = 'add_node' if added == 'Zr' else 'add_linker'
            forward = before['free'][added]/total0
            reverse = self.ledger.rho0
        elif removed:
            species = removed[0]
            forward = self.ledger.rho0
            reverse = after['free'][species]/total0
        elif event == 'remove_bond':
            forward, reverse = self.ledger.rho0, c['close_internal']
        elif event == 'close_internal':
            forward, reverse = c['close_internal'], self.ledger.rho0
        else:
            forward = c[event]
            reverse_event = 'modulator_to_solvent' if event == 'solvent_to_modulator' else 'solvent_to_modulator'
            reverse = c[reverse_event]
            # Two acetate oxygen assignments, one solvent assignment.
            qf += -math.log(2) if event == 'solvent_to_modulator' else 0.
            qr += -math.log(2) if event == 'modulator_to_solvent' else 0.
        log_rate_ratio = math.log(forward)-math.log(reverse)+qf-qr-du
        residual = log_rate_ratio-log_measure
        if abs(residual) > 1e-8:
            raise AssertionError(('elementary_detailed_balance_mismatch', event, residual,
                                  log_rate_ratio, log_measure))
        return {'log_rate_ratio': log_rate_ratio, 'log_state_measure_ratio': log_measure,
                'residual': residual, 'scope': 'declared discrete-state reference; not oligomer boundary'}
