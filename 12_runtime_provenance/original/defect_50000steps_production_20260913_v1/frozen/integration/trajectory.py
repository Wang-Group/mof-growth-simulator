"""Timed molecular runner and trusted-local checkpoint support.

No automatic production launcher. Every run has finite event and wall caps.
Pickles are only for trusted project-generated checkpoints, never external input.
"""
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np

from vacancy_counter import FCCRegistration, VacancyCohort, assembly_coordinates


class Simulation:
    def __init__(self, model, seed):
        self.model = model
        self.events_rng = random.Random(seed)
        self.event_count = 0
        self.tau = 0.
        self.time_compensation = 0.
        self.pending_wait = None
        self.maximum_N = len(model.assembly.entities)
        self.reg = FCCRegistration.infer(assembly_coordinates(model.assembly)[0])
        self.cohorts = {depth: VacancyCohort() for depth in (1, 2)}
        self.first_passage = {}
        self.observations = []

    def add_time(self, dt):
        adjusted = dt-self.time_compensation
        new = self.tau+adjusted
        self.time_compensation = (new-self.tau)-adjusted
        self.tau = new

    def advance(self, model_time_limit=None):
        prop = self.model.propensities()
        total = math.fsum(prop.values())
        if total <= 0:
            return 'no_active_channel'
        if self.pending_wait is None:
            self.pending_wait = -math.log1p(-self.events_rng.random())/total
        dt = self.pending_wait
        if model_time_limit is not None and self.tau+dt > model_time_limit:
            remaining = max(0., model_time_limit-self.tau)
            self.pending_wait = dt-remaining
            self.add_time(remaining)
            return 'model_time_cap'
        self.pending_wait = None
        self.add_time(dt)
        event, item = self.model.choose(prop, self.events_rng)
        self.model.attempt(event, item, self.events_rng)
        self.event_count += 1
        self.maximum_N = max(self.maximum_N, len(self.model.assembly.entities))
        return 'event'

    def observe(self):
        m = self.model
        nodes, linkers, degrees = assembly_coordinates(m.assembly)
        prop = m.propensities()
        external = prop['add_node']+prop['add_linker']
        measurements = {}
        for depth in (1, 2):
            measurement = self.reg.count(nodes, linkers, degrees, shell_depth=depth)
            measurement['vacancy_cohort'] = self.cohorts[depth].update(measurement)
            measurements[str(depth)] = measurement
        row = {'events': self.event_count, 'tau': self.tau, 'N': len(m.assembly.entities),
               'maximum_N': self.maximum_N, 'pool': m.ledger.record(),
               'bond_count': len(m.assembly.linked_carboxylate_pairs),
               'D_count': len(m.intermediates), 'cap_count': len(m.cap),
               'solvent_cap_count': m.solvent_count(),
               'geometry_allowed_closures': len(m.closure_candidates()),
               'blocked_closures': len(m.intermediates)-len(m.closure_candidates()),
               'proposal_propensities': prop,
               'total_removal_external_addition_proposal_ratio': prop['remove_bond']/external if external else None,
               'vacancies': measurements}
        self.observations.append(row)
        return row

    def checkpoint(self, path, provenance):
        # One pickle stream preserves graph/site/cap dictionary-key identity.
        fields = dict(self.model.__dict__)
        fields.pop('engine')
        state = dict(self.__dict__)
        state.pop('model')
        record = {'schema': 'acetate-defect-checkpoint-1', 'provenance': provenance,
                  'model_fields': fields, 'simulation_fields': state,
                  'python_global_rng': random.getstate(), 'numpy_global_rng': np.random.get_state()}
        with Path(path).open('xb') as handle:
            pickle.dump(record, handle, protocol=5)

    @classmethod
    def restore(cls, path, engine, expected_provenance):
        from defect_model import DefectAssembly
        with Path(path).open('rb') as handle:
            record = pickle.load(handle)
        if record['schema'] != 'acetate-defect-checkpoint-1' or record['provenance'] != expected_provenance:
            raise ValueError('Checkpoint schema or source provenance differs')
        model = DefectAssembly.__new__(DefectAssembly)
        model.__dict__.update(record['model_fields'])
        model.engine = engine
        model.check_state()
        restored = cls.__new__(cls)
        restored.__dict__.update(record['simulation_fields'])
        restored.model = model
        random.setstate(record['python_global_rng'])
        np.random.set_state(record['numpy_global_rng'])
        return restored


def run_bounded(sim, output, provenance, *, target, max_events, wall_seconds,
                model_time_limit=None, sample_every=100, milestones=(100, 150, 250, 500, 1000, 2500, 5000)):
    from validate_replaceable_caps import export_xyz
    if (target < 1 or max_events < 1 or wall_seconds <= 0 or not math.isfinite(wall_seconds)
            or sample_every < 1 or (model_time_limit is not None and
                                   (not math.isfinite(model_time_limit) or model_time_limit <= sim.tau))):
        raise ValueError('Explicit positive finite run bounds required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status, error = 'running', None
    initial_events = sim.event_count
    sim.observe()
    try:
        while status == 'running':
            n = len(sim.model.assembly.entities)
            for value in sorted(set(milestones) | {target}):
                if n >= value and str(value) not in sim.first_passage:
                    sim.first_passage[str(value)] = {'tau': sim.tau, 'events': sim.event_count, 'N': n}
            if n >= target:
                status = 'target_reached'
            elif sim.event_count-initial_events >= max_events:
                status = 'event_cap'
            elif time.perf_counter()-started >= wall_seconds:
                status = 'wall_time_cap'
            else:
                outcome = sim.advance(model_time_limit)
                if outcome != 'event':
                    status = outcome
                if sim.event_count % sample_every == 0 or n < target <= len(sim.model.assembly.entities):
                    sim.observe()
    except Exception as exc:
        status, error = 'error', repr(exc)
        raise
    finally:
        sim.model.check_state()
        final = sim.observe()
        export_xyz(output/'final_GEOMETRY_ONLY.xyz', sim.model.blocks(),
                   'Explicit solvent surrogate included; NOT a physical all-atom PXRD input')
        export_xyz(output/'final_framework_only.xyz', sim.model.bare_blocks(),
                   'Zr6 and BDC physical coordinates; caps and solvent surrogate excluded')
        sim.checkpoint(output/'checkpoint.pkl', provenance)
        report = {'status': status, 'error': error, 'censored': status != 'target_reached',
                  'purpose': 'bounded integration validation; not production defect statistics',
                  'target_N': target, 'final_N': final['N'], 'maximum_N': sim.maximum_N,
                  'events': sim.event_count, 'tau': sim.tau, 'wall_seconds': time.perf_counter()-started,
                  'caps': {'max_events_this_run': max_events, 'wall_seconds': wall_seconds,
                           'model_time_limit': model_time_limit},
                  'parameters': {'delta': sim.model.delta, 'rho0': sim.model.ledger.rho0,
                                 's_close': sim.model.ledger.s_close,
                                 'eta_solvent_prior': sim.model.ledger.eta_solvent},
                  'first_passage': sim.first_passage, 'provenance': provenance,
                  'transition_counts': dict(sim.model.transition_counts), 'final': final}
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        (output/'observations.json').write_text(json.dumps(sim.observations, indent=2, allow_nan=False))
    return report
