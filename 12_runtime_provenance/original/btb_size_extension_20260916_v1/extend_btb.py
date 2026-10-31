"""Checkpointable BTB size extension; unchanged frozen molecular engine.

Only bookkeeping and uniform bond sampling are accelerated. The latter uses
floor(u * number_of_bonds), exactly the old cumulative-unit-weight selection,
including its one RNG draw (not random.choice, which consumes different draws).
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get('MOL_GROWTH_REVIEW_BUNDLE',
    str(HERE.parent / 'btb_high_rho_N1000_20260907_v1' / 'snapshot')))
os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = str(BUNDLE)
sys.path.insert(0, str(BUNDLE / 'code' / 'proposed'))
from common_molecular_runner import _load_engine, assembly_state, connected_component_sizes, sha256
from common_kmc_selector import (PhysicalCoordinates, SelectedEvent, counts_from_assembly,
    compute_propensities, gillespie_waiting_time, chemistry_record, select_event)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def fast_select(assembly, coordinates, rng):
    prop = compute_propensities(counts_from_assembly(assembly), coordinates)
    if prop.total <= 0:
        raise RuntimeError('no enabled event')
    draw = rng.random() * prop.total
    if draw < prop.add_node:
        return SelectedEvent('add_node', rng.choice(assembly.Linker_free_carboxylates.items), None, prop)
    draw -= prop.add_node
    if draw < prop.add_linker:
        return SelectedEvent('add_linker', rng.choice(assembly.MC_free_carboxylates.items), None, prop)
    draw -= prop.add_linker
    if draw < prop.close_internal:
        return SelectedEvent('close_internal', None, rng.choice(assembly.ready_to_connect_carboxylate_pairs.items), prop)
    pool = assembly.linked_carboxylate_pairs.items
    return SelectedEvent('remove_bond', None, pool[min(int(rng.random() * len(pool)), len(pool)-1)], prop)


def coordinates(row):
    return PhysicalCoordinates(float(row['external_activity']), float(row['node_fraction']),
                               float(row['off_weight']), float(row['SC']))


def step(assembly, coord, rng, counters, selector):
    before = len(assembly.entities)
    selected = selector(assembly, coord, rng)
    dt = gillespie_waiting_time(selected.propensities.total, rng)
    if selected.event in ('add_node', 'add_linker'):
        counters[selected.event + '_attempt'] += 1
        assembly.grow_one_step(selected.selected_site)
        suffix = '_accepted' if len(assembly.entities) > before else '_collision_rejected'
        counters[selected.event + suffix] += 1
    elif selected.event == 'close_internal':
        assembly.link_internal_carboxylate(selected.selected_pair)
        counters['close_internal'] += 1
    else:
        assembly.remove_linkage(selected.selected_pair)
        counters['remove_bond'] += 1
    return dt


def run(row, destination, resume=None, reference=False):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'result.json'
    if result_path.exists():
        raise FileExistsError(f'Preserving existing result: {result_path}')
    engine, engine_path, redirects = _load_engine('btb_mol')
    target = int(row['target_entities'])
    engine.MAX_DEPTH = target + 1
    coord = coordinates(row)
    if resume:
        with Path(resume).open('rb') as f:
            state = pickle.load(f)
        if state['coordinates'] != asdict(coord) or state['random_seed'] != int(row['random_seed']):
            raise ValueError('Resume coordinates/random seed differ')
        if state['engine_sha256'] != sha256(engine_path):
            raise ValueError('Resume engine differs')
        assembly, rng = state['assembly'], random.Random()
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        rng.setstate(state['selector_rng'])
        counters = Counter(state['event_counts'])
        event, model_time, maximum = state['event'], state['model_time'], state['maximum_entities']
        first_passage = state['first_passage']
    else:
        seed = Path(row['seed_pickle'])
        if sha256(seed) != row['seed_pickle_sha256']:
            raise ValueError('Seed checksum mismatch')
        with seed.open('rb') as f:
            assembly = pickle.load(f)
        if connected_component_sizes(assembly) != [30]:
            raise ValueError('Expected reviewed connected N=30 seed')
        random_seed = int(row['random_seed'])
        random.seed(random_seed)
        np.random.seed(random_seed % (2**32-1))
        rng = random.Random(random_seed ^ 0x5EEDBEEF)
        counters, event, model_time, maximum, first_passage = Counter(), 0, 0.0, 30, {}
    for entity in assembly.entities.items:
        if hasattr(entity, 'kdtree'):
            entity.kdtree = None
    started = time.perf_counter()
    start_event = event
    milestones = [int(n) for n in row['mol2_milestones'].split(';') if int(n) > maximum and int(n) <= target]
    saved = []
    selector = select_event if reference else fast_select
    last_save = started

    def checkpoint():
        payload = dict(assembly=assembly, coordinates=asdict(coord), random_seed=int(row['random_seed']),
            engine_sha256=sha256(engine_path), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            selector_rng=rng.getstate(), event_counts=dict(counters), event=event,
            model_time=model_time, maximum_entities=maximum, first_passage=first_passage)
        temporary = out / 'checkpoint.pkl.tmp'
        with temporary.open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(out / 'checkpoint.pkl')
        atomic_json(out / 'progress.json', dict(event=event, entities=len(assembly.entities),
            maximum_entities=maximum, model_time=model_time, wall_seconds=time.perf_counter()-started,
            updated_unix=time.time(), status='running'))

    termination = 'target_reached' if len(assembly.entities) >= target else 'event_cap'
    while event < int(row['event_cap']) and len(assembly.entities) < target:
        if time.perf_counter() - started >= float(row['wall_time_cap_s']):
            termination = 'wall_time_cap'
            break
        try:
            dt = step(assembly, coord, rng, counters, selector)
        except RuntimeError as exc:
            if 'no enabled event' not in str(exc):
                raise
            termination = 'no_enabled_event'
            break
        model_time += dt
        event += 1
        size = len(assembly.entities)
        if size > maximum:
            for n in range(maximum+1, size+1):
                first_passage[str(n)] = dict(event=event, model_time=model_time)
            maximum = size
        while milestones and size >= milestones[0]:
            n = milestones.pop(0)
            mol2 = out / f'N{n:05d}.mol2'
            assembly.get_mol2_file(str(mol2))
            saved.append(dict(requested_entities=n, observed_entities=size, event=event,
                model_time=model_time, path=str(mol2), sha256=sha256(mol2)))
            atomic_json(out / 'milestones.json', saved)
            checkpoint()
            last_save = time.perf_counter()
        if event % 1000 == 0 and time.perf_counter() - last_save >= 120:
            checkpoint()
            last_save = time.perf_counter()
        if size >= target:
            termination = 'target_reached'
            break
    checkpoint()
    final_mol2 = out / 'final.mol2'
    assembly.get_mol2_file(str(final_mol2))
    result = dict(schema_version='btb-checkpointable-size-extension-v1', run_id=row['run_id'],
        source_run_id=row['source_run_id'], random_seed=int(row['random_seed']),
        coordinates=chemistry_record(coord), final_state=assembly_state(assembly),
        termination=dict(status=termination, target_entities=target, executed_events=event,
            events_this_segment=event-start_event, maximum_entities=maximum,
            wall_time_cap_s=float(row['wall_time_cap_s']), event_cap=int(row['event_cap'])),
        clock=dict(model_time=model_time), event_counts=dict(counters), first_passage=first_passage,
        milestones=saved, final_mol2_sha256=sha256(final_mol2), wall_seconds=time.perf_counter()-started,
        provenance=dict(engine_path=str(engine_path), engine_sha256=sha256(engine_path),
            runner_sha256=sha256(Path(__file__)), bundle=str(BUNDLE), import_redirects=redirects,
            resume_path=str(resume) if resume else None,
            checkpoint_sha256=sha256(out/'checkpoint.pkl'), selector='reference' if reference else 'equivalent_uniform_fast',
            topology='unrestricted; all 12 node directions available', pool='constant activities'))
    atomic_json(result_path, result)
    atomic_json(out/'progress.json', dict(status=termination, entities=len(assembly.entities), event=event,
        model_time=model_time, updated_unix=time.time()))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--reference', action='store_true')
    args = p.parse_args()
    with args.manifest.open(encoding='utf-8-sig', newline='') as f:
        row = list(csv.DictReader(f))[args.index]
    run(row, args.output, args.resume, args.reference)
