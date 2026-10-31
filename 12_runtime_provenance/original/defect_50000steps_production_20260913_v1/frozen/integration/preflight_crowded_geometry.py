"""5090-only structural preflight for capped, singly attached BDC states.

NOT a growth simulation. Construct disposable candidate states from the reviewed
N76 seed; never modify the seed or the completed cap-free campaign. A finite
hinge library is a declared geometric approximation, not an optimized barrier.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import socket
import sys
import time

import numpy as np


def rotate_about_axis(xyz, origin, axis, angle_degrees):
    axis = np.asarray(axis, float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError('A finite nonzero hinge axis is required')
    u = axis / np.linalg.norm(axis)
    v = np.asarray(xyz, float) - origin
    theta = math.radians(angle_degrees)
    return (v * math.cos(theta) + np.cross(u, v) * math.sin(theta)
            + np.outer(v @ u, u) * (1 - math.cos(theta)) + origin)


def nonseparating_edge(assembly, pair):
    left, right = (s.belonging_entity for s in pair)
    queue, seen = deque([left]), {left}
    while queue:
        node = queue.popleft()
        for other in node.connected_entities:
            if (node is left and other is right) or (node is right and other is left):
                continue
            if other is right:
                return True
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return False


def local_reversible_rates(rho, addition, closure, penalty, feasible_prior_mass):
    """A resolved F-D-V path with a normalized finite conformer proposal.

    q is an aggregate of explicit candidate prior probabilities, not an empirical
    node-degree factor. Endpoint ratio cancels q and delta for this ONE channel.
    Competing graph deletion/proposal processes are not certified by this algebra.
    """
    if not (all(math.isfinite(x) for x in (rho, addition, closure, penalty, feasible_prior_mass))
            and rho > 0 and addition > 0 and closure > 0 and penalty >= 0
            and 0 <= feasible_prior_mass <= 1):
        raise ValueError('Invalid local rate parameters')
    q = feasible_prior_mass
    entry = q * math.exp(-penalty)
    return {'F_to_D': rho * entry, 'D_to_F': closure,
            'V_to_D': addition * entry, 'D_to_V': rho,
            'F_to_V_eliminated': rho * entry * rho / (closure + rho),
            'V_to_F_eliminated': addition * entry * closure / (closure + rho),
            'endpoint_V_over_F': rho * rho / (addition * closure),
            'endpoint_ratio_scope': 'formal communicating-path ratio; q=0 disconnects F and V',
            'elimination_scope': 'short-lived D, local path only; not full growth kinetics'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True, help='unchanged mixed-cap source root')
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--acetate-source', type=Path, required=True,
                    help='existing Zr6 acetate model with explicit methyl hydrogens')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-pairs', type=int, default=24)
    args = ap.parse_args()
    if socket.gethostname() != 'xmhy-G3261':
        raise RuntimeError('Molecular preflight belongs on 5090')
    if args.max_pairs < 1:
        raise ValueError('max-pairs must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    import os
    os.environ['MOL_GROWTH_REVIEW_BUNDLE'] = str(args.bundle)
    sys.path[:0] = [str(args.bundle / 'code/proposed'), str(args.root),
                   str(args.root / 'solvent_competition_20260912_v1')]
    import common_molecular_runner as runner
    from explicit_cap_kmc import stack_xyz, tree_for, touches
    from acetate_cap_library import AcetateCapLibrary
    from mixed_solvent_kmc import MixedCapAssembly
    from solvent_channel import SiteCompetition
    from validate_replaceable_caps import export_xyz
    started = time.perf_counter()
    engine, engine_path, _ = runner._load_engine('uio66')
    assert sha(engine_path) == '4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9'
    seed_path = args.root / 'framework_inputs/UiO66_N76_reviewed.pkl'
    assert sha(seed_path) == '9b08834d55e0e1674d6e379eb991594947d000adeea6831d430e1f69b98e48c7'
    random.seed(20260912)
    np.random.seed(20260912)
    with seed_path.open('rb') as f:
        assembly = pickle.load(f)
    library = AcetateCapLibrary(args.acetate_source)
    model = MixedCapAssembly(assembly, engine, library,
                             SiteCompetition.from_rho_fraction(1., .5),
                             np.random.default_rng(20260912))
    assert len(assembly.entities) == 76
    initial_coords = {e: e.coordinates.copy() for e in assembly.entities.items}
    initial_caps = len(model.cap)
    angles = [-40., -30., -20., -10., 10., 20., 30., 40.]
    report = {'schema': 'crowded-BDC-geometry-preflight-v1', 'host': socket.gethostname(),
              'not_a_growth_run': True, 'N': 76, 's_close': 1,
              'identity': 'acetate', 'eta_S': .5, 'random_seed': 20260912,
              'cap_geometry_provenance': library.provenance,
              'source_hashes': {'engine': sha(engine_path), 'seed': sha(seed_path),
                                'preflight': sha(__file__),
                                'acetate_library': sha(Path(__file__).with_name('acetate_cap_library.py')),
                                'acetate_source': sha(args.acetate_source)},
              'hinge_angles_degrees': angles, 'collision_cutoff_A': 2.,
              'attached_COO_max_displacement_A': .5,
              'candidate_definition': 'rigid BDC rotation about the attached O-O line; '
                 'both attached O positions exact, all intramolecular distances preserved; '
                 'carboxylate C may move by at most 0.5 A; the receiving site is capped',
              'energy_assignment': 'no geometry-derived energy; delta is a separate prospective screen',
              'pairs': [], 'examples': [],
              'remaining_production_gates': ['coupled incoming-node multiple-intermediate packing',
                 'reverse-paired insertion and deletion with finite pools',
                 'interior lattice vacancy counter controls',
                 'author review of integrated code and candidate structures before long jobs']}
    export_xyz(args.output / 'reference_F_GEOMETRY_ONLY.xyz', model.blocks(),
               'N76 F reference; contains solvent proxy atoms, not physical PXRD input')
    assert all(Counter(c['elements']) == {'C': 2, 'H': 3, 'O': 2}
               for choices in model.candidates.values() for c in choices if c['kind'] == 'M')
    report['initial_caps_by_kind'] = dict(Counter(c['kind'] for c in model.cap.values()))
    export_xyz(args.output / 'acetate_template_GEOMETRY_ONLY.xyz',
               [{'elements': library.elements, 'xyz': library.xyz,
                 'role': 'acetate_template', 'entity': -1, 'site': -1}],
               'Existing explicit-H acetate fragment; geometry only, not optimized intermediate')
    eligible = []
    for pair in assembly.linked_carboxylate_pairs.items:
        ms, ls = sorted(pair, key=lambda s: s.belonging_entity.entity_type != 'Zr')
        linker = ls.belonging_entity
        if len(linker.connected_entities) != 2 or not nonseparating_edge(assembly, pair):
            continue
        eligible.append((pair, ms, ls, linker))
    report['eligible_oriented_bonds'] = len(eligible)
    exported = set()
    for index, (pair, ms, ls, linker) in enumerate(eligible[:args.max_pairs]):
        attached = next(s for s in linker.carboxylates if s is not ls)
        attached_xyz = linker.coordinates[attached.atom_indexes]
        origin = attached_xyz[1]
        axis = attached_xyz[2] - attached_xyz[1]
        fixed_blocks = [b for b in model.blocks()
                        if not (b['role'] == 'framework_BDC'
                                and b['entity'] == model.identifiers[linker])]
        fixed_tree = tree_for(stack_xyz(fixed_blocks))
        stats = Counter()
        mass = {'S': 0., 'M': 0.}
        allowed = []
        for angle in angles:
            xyz = rotate_about_axis(linker.coordinates, origin, axis, angle)
            attached_error = float(np.linalg.norm(xyz[attached.atom_indexes]
                                                  - attached_xyz, axis=1).max())
            if attached_error > .5:
                stats['attached_geometry_outside_0.5A'] += len(model.candidates[ms])
                continue
            if touches(xyz, fixed_tree):
                stats['displaced_BDC_hits_existing_atoms'] += len(model.candidates[ms])
                continue
            for ci, cap in enumerate(model.candidates[ms]):
                if touches(cap['xyz'], fixed_tree):
                    stats['receiving_cap_hits_existing_atoms'] += 1
                    continue
                if touches(cap['xyz'], tree_for(xyz)):
                    stats['receiving_cap_hits_displaced_BDC'] += 1
                    continue
                prior = cap['prior'] / len(angles)
                mass[cap['kind']] += prior
                stats['valid_' + cap['kind']] += 1
                allowed.append({'angle': angle, 'cap_index': ci, 'kind': cap['kind'],
                                'prior': prior, 'attached_max_displacement_A': attached_error})
                if cap['kind'] not in exported:
                    name = 'candidate_D_' + cap['kind'] + '_GEOMETRY_ONLY.xyz'
                    block = {'elements': list(linker.elements), 'xyz': xyz,
                             'role': 'candidate_displaced_BDC',
                             'entity': model.identifiers[linker], 'site': -1}
                    receiving = dict(cap, role='candidate_receiving_cap',
                                     entity=model.identifiers[ms.belonging_entity],
                                     site=ms.belonging_entity.carboxylates.index(ms))
                    export_xyz(args.output / name, fixed_blocks + [block, receiving],
                               'Candidate D hinge geometry, not DFT optimized; solvent proxy may be present')
                    report['examples'].append({'file': name, 'pair_index': index,
                                               'cap_kind': cap['kind'], 'angle': angle})
                    exported.add(cap['kind'])
        report['pairs'].append({'pair_index': index, 'linker_id': model.identifiers[linker],
                                'receiving_node_id': model.identifiers[ms.belonging_entity],
                                'statistics': dict(stats), 'feasible_prior_mass': mass,
                                'allowed_candidates': allowed})
        print(json.dumps({'checked': index + 1, 'feasible_prior_mass': mass}), flush=True)
    for entity, xyz in initial_coords.items():
        assert np.array_equal(entity.coordinates, xyz)
    assert len(model.cap) == initial_caps
    model.assert_bookkeeping()
    report['seed_and_model_unmodified'] = True
    report['pairs_with_any_candidate'] = sum(sum(p['feasible_prior_mass'].values()) > 0
                                             for p in report['pairs'])
    report['local_rate_screen'] = [dict(delta=d, **local_reversible_rates(
        1., 6/7, math.exp(30.9), d, 1.)) for d in [0, 2, 4, 6, 8, 10]]
    report['wall_seconds'] = time.perf_counter() - started
    (args.output / 'geometry_preflight.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'finished': True, 'pairs_with_any_candidate': report['pairs_with_any_candidate'],
                      'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
