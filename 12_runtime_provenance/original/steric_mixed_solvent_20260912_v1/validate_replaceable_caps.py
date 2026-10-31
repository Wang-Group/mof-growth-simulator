"""Explicit cap substitution and link/unlink validation, NOT a growth-rate run.

Runs against the reviewed engine and frozen seed on 5090. No legacy source is
modified. Full acetate fragments are removed, inorganic coordinates and all
framework coordinates are retained, and only unlinked metal sites receive the
selected monocarboxylate. An unresolved ready-state overlap is a failed model
validation, never a physical steric rejection or a censored growth trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from cap_geometry_preflight import read_mol, anchors, proper_fit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_indices(elements, edges):
    sites, adj = anchors(elements, edges)
    groups = []
    for site in sites:
        found, stack = set(), [site[0]]
        while stack:
            i = stack.pop()
            if i in found or elements[i] == 'Zr':
                continue
            found.add(i)
            stack.extend(j for j in adj[i] if elements[j] != 'Zr')
        groups.append(found)
    assert len(groups) == 12
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    core = sorted(set(range(len(elements))) - set.union(*groups))
    assert Counter(elements[i] for i in core) == {'Zr': 6, 'O': 8}
    return core, groups


def contact_summary(left, right, cutoff=2.0):
    if not len(left) or not len(right):
        return {'pairs_below_cutoff': 0, 'minimum_distance_A': None}
    d2 = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return {'pairs_below_cutoff': int(np.count_nonzero(d2 < cutoff ** 2)),
            'minimum_distance_A': float(np.sqrt(d2.min()))}


def entity_core_indices(entity):
    """Seed import can reorder atoms; use its own twelve COO-site records."""
    assert len(entity.carboxylates) == 12
    cap_oxygen = {int(i) for c in entity.carboxylates for i in c.atom_indexes[1:]}
    core = [i for i, e in enumerate(entity.elements)
            if e == 'Zr' or (e == 'O' and i not in cap_oxygen)]
    assert Counter(entity.elements[i] for i in core) == {'Zr': 6, 'O': 8}
    assert Counter(entity.elements) == {'Zr': 6, 'O': 32, 'C': 24}
    assert len(set(range(len(entity.elements))) - set(core)) == 48
    return core


def cap_candidates(identity, root, target):
    ind = identity['indices']
    base = [ind['Ccarboxylate'], ind['O1'], ind['O2']]
    result = []
    for conf in identity['conformers']:
        elem, xyz, _ = read_mol(root / 'molecule_library' / conf['mol_file'])
        for flip in [False, True]:
            ii = [base[0], base[2], base[1]] if flip else base
            rot, shift, err = proper_fit(xyz[ii], target)
            result.append({'elements': elem, 'xyz': xyz @ rot + shift,
                           'conformer': conf['conformer'], 'oxygen_swapped': flip,
                           'anchor_rmsd_A': err, 'determinant': float(np.linalg.det(rot))})
    return result


def retained_component(assembly, removed_pair):
    """Exact smaller-fragment deletion on a temporary adjacency list."""
    entities = list(assembly.entities.items)
    a, b = (c.belonging_entity for c in removed_pair)
    adj = {e: list(e.connected_entities) for e in entities}
    adj[a].remove(b)
    adj[b].remove(a)
    seen, stack = {a}, [a]
    while stack:
        for other in adj[stack.pop()]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    if b in seen:
        return set(entities), 'internal_opening'
    other = set(entities) - seen
    return (seen if len(seen) >= len(other) else other), 'fragment_dissociation'


def export_xyz(path, blocks, comment):
    n = sum(len(b['xyz']) for b in blocks)
    rows = [str(n), comment]
    ownership = []
    for block in blocks:
        for element, point in zip(block['elements'], block['xyz']):
            rows.append(f'{element} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}')
            ownership.append({k: block[k] for k in ('role', 'entity', 'site')})
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    path.with_suffix('.atom_ownership.json').write_text(json.dumps(ownership), encoding='utf-8')


def validate_identity(identity, root, seed, core, node_elements, out):
    entities = list(seed.entities.items)
    index = {e: i for i, e in enumerate(entities)}
    linked_sites = {c for p in seed.linked_carboxylate_pairs.items for c in p}
    bare_blocks, cap_slots = [], []
    for e in entities:
        i = index[e]
        if e.entity_type == 'Zr':
            local_core = entity_core_indices(e)
            bare_blocks.append({'elements': [e.elements[k] for k in local_core],
                                'xyz': e.coordinates[local_core].copy(), 'role': 'inorganic_core',
                                'entity': i, 'site': -1})
            for s, c in enumerate(e.carboxylates):
                if c not in linked_sites:
                    cap_slots.append((i, s, c, cap_candidates(identity, root, e.coordinates[c.atom_indexes])))
        else:
            assert list(e.elements).count('H') == 4, 'BDC aromatic hydrogens missing'
            bare_blocks.append({'elements': list(e.elements), 'xyz': e.coordinates.copy(),
                                'role': 'framework_BDC', 'entity': i, 'site': -1})
    bare = np.vstack([b['xyz'] for b in bare_blocks])
    placed, packing = [], []
    # Finite greedy initialization diagnostic only; no conformer free energy or
    # steric coefficient is silently added to KMC chemical constants.
    for i, s, c, candidates in cap_slots:
        environment = np.vstack([bare, *[p['xyz'] for p in placed]])
        scores = [contact_summary(p['xyz'], environment) for p in candidates]
        choice = min(range(len(scores)), key=lambda k: (scores[k]['pairs_below_cutoff'], k))
        p = candidates[choice]
        placed.append({**p, 'role': 'modulator_cap', 'entity': i, 'site': s})
        packing.append({'entity': i, 'site': s, 'candidate_count': len(candidates),
                        'chosen_conformer': p['conformer'], 'oxygen_swapped': p['oxygen_swapped'],
                        **scores[choice]})
    blocks = bare_blocks + placed
    name = identity['identity'].replace('-', '_')
    export_xyz(out / f'{name}_N76_replaced_caps.xyz', blocks,
               'CAP-REPLACEMENT VALIDATION; see JSON contact checks; not a growth result')
    reverses = []
    example_written = False
    for pi, pair in enumerate(seed.linked_carboxylate_pairs.items):
        retained, kind = retained_component(seed, pair)
        mc = next(c for c in pair if c.belonging_entity.entity_type == 'Zr')
        lc = next(c for c in pair if c.belonging_entity.entity_type != 'Zr')
        if mc.belonging_entity not in retained:
            reverses.append({'pair_index': pi, 'kind': kind, 'metal_node_retained': False})
            continue
        i = index[mc.belonging_entity]
        s = mc.belonging_entity.carboxylates.index(mc)
        choices = cap_candidates(identity, root, mc.belonging_entity.coordinates[mc.atom_indexes])
        kept = [b for b in blocks if entities[b['entity']] in retained]
        environment = np.vstack([b['xyz'] for b in kept])
        scores = [contact_summary(p['xyz'], environment) for p in choices]
        choice = min(range(len(scores)), key=lambda k: (scores[k]['pairs_below_cutoff'], k))
        selected = choices[choice]
        partner = (lc.belonging_entity.coordinates if lc.belonging_entity in retained
                   else np.empty((0, 3)))
        partner_contact = contact_summary(selected['xyz'], partner)
        record = {'pair_index': pi, 'kind': kind, 'metal_node_retained': True,
                  'node_entity': i, 'site': s, 'retained_entities': len(retained),
                  'minimum_contacts_over_all_cap_candidates': scores[choice]['pairs_below_cutoff'],
                  'cap_to_remaining_partner_BDC': partner_contact,
                  'all_candidates_tested': len(choices)}
        reverses.append(record)
        if kind == 'internal_opening' and not example_written:
            export_xyz(out / f'{name}_internal_opening_overlap_example.xyz',
                       [{'elements': list(lc.belonging_entity.elements),
                         'xyz': lc.belonging_entity.coordinates,
                         'role': 'remaining_BDC', 'entity': index[lc.belonging_entity], 'site': -1},
                        {**selected, 'role': 'restored_cap', 'entity': i, 'site': s}],
                       'INVALID overlap diagnostic: internal bond open, BDC still present')
            example_written = True
    summary = {'identity': identity['identity'], 'framework_entities': len(entities),
               'Zr6_nodes': sum(e.entity_type == 'Zr' for e in entities),
               'BDC_linkers': sum(e.entity_type != 'Zr' for e in entities),
               'old_acetate_atoms_excluded': 48 * sum(e.entity_type == 'Zr' for e in entities),
               'new_caps_on_unlinked_sites': len(placed),
               'new_caps_on_linked_sites': 0,
               'initial_cap_contact_pairs': sum(r['pairs_below_cutoff'] for r in packing),
               'internal_openings_tested': sum(r['kind'] == 'internal_opening' for r in reverses),
               'internal_openings_with_cap_BDC_overlap': sum(
                   r['kind'] == 'internal_opening' and
                   r.get('cap_to_remaining_partner_BDC', {}).get('pairs_below_cutoff', 0) > 0
                   for r in reverses),
               'fragment_dissociations_tested': sum(r['kind'] == 'fragment_dissociation' for r in reverses),
               'maximum_anchor_rmsd_A': max(p['anchor_rmsd_A'] for p in placed),
               'status': 'STRUCTURAL_AND_REVERSE_STATE_VALIDATION_NOT_GROWTH'}
    (out / f'{name}_validation.json').write_text(json.dumps(
        {'summary': summary, 'initial_packing': packing, 'reverse_transition_tests': reverses}, indent=2))
    return summary


def actual_engine_open_close_test(engine, seed):
    """Exercise existing ring-opening/closing code without changing its rates."""
    for i, pair in enumerate(seed.linked_carboxylate_pairs.items):
        _, kind = retained_component(seed, pair)
        if kind != 'internal_opening':
            continue
        working = copy.deepcopy(seed)
        p = working.linked_carboxylate_pairs.items[i]
        original = [e.coordinates.copy() for e in working.entities.items]
        before = len(working.linked_carboxylate_pairs)
        working.remove_linkage(p)
        after_open = len(working.linked_carboxylate_pairs)
        in_ready = p in working.ready_to_connect_carboxylate_pairs
        unchanged = all(np.array_equal(x, e.coordinates) for x, e in zip(original, working.entities.items))
        working.link_internal_carboxylate(p)
        return {'pair_index': i, 'bonds_before': before, 'bonds_after_open': after_open,
                'bonds_after_reclose': len(working.linked_carboxylate_pairs),
                'entities_after_open_and_close': len(working.entities),
                'opened_pair_in_ready_pool': in_ready,
                'coordinates_unchanged_when_opened': unchanged,
                'test': 'actual reviewed engine methods on a disposable seed copy'}
    raise AssertionError('No internal ring bond available in reviewed seed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--engine-code-dir', type=Path, required=True)
    ap.add_argument('--bundle-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import os
    os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = str(args.bundle_root.resolve())
    sys.path.insert(0, str(args.engine_code_dir.resolve()))
    import common_molecular_runner as runner
    engine, engine_path, redirects = runner._load_engine('uio66')
    engine.MAX_DEPTH = 1000
    # Verify the exact reviewed source, not an older engine or a moving copy.
    assert sha(engine_path) == '4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9'
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'framework_inputs/UiO66_N76_reviewed.pkl'
    assert sha(source) == '9b08834d55e0e1674d6e379eb991594947d000adeea6831d430e1f69b98e48c7'
    with source.open('rb') as f:
        seed = pickle.load(f)
    elem, _, edges = read_mol(root / 'framework_inputs/Zr6_AA_reference.mol')
    core, _ = core_indices(elem, edges)
    lib = json.loads((root / 'molecule_library/molecule_library_index.json').read_text())
    started = time.perf_counter()
    rows = []
    for identity in lib['identities']:
        result = validate_identity(identity, root, seed, core, elem, out)
        rows.append(result)
        print(json.dumps(result), flush=True)
    report = {'identity_results': rows, 'engine_round_trip': actual_engine_open_close_test(engine, seed),
              'engine_path': str(engine_path), 'engine_sha256': sha(engine_path),
              'seed_sha256': sha(source), 'import_redirects': redirects,
              'worker_count': 1, 'wall_seconds': time.perf_counter() - started,
              'growth_simulations_run': 0,
              'note': 'No identity-dependent rate coefficient or reverse-bond accessibility rule introduced.'}
    (out / 'VALIDATION_SUMMARY.json').write_text(json.dumps(report, indent=2))
    files = [{'path': p.relative_to(out).as_posix(), 'sha256': sha(p)}
             for p in sorted(out.iterdir()) if p.is_file()]
    (out / 'REMOTE_SHA256.json').write_text(json.dumps(files, indent=2))
    print(json.dumps(report['engine_round_trip']), flush=True)


if __name__ == '__main__':
    main()
