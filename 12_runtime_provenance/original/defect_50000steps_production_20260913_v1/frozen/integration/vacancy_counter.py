"""Read-only seed-lattice registration and conservative interior vacancy counts.

This module never selects or rejects a growth event. Complete one/two FCC-shell
enclosure is a conservative eligibility mask, independent of occupancy of the
tested center. Clustered holes and irregular boundaries may remain unclassified.
"""
from collections import Counter
from itertools import combinations, product
import numpy as np


def key(v):
    return tuple(int(x) for x in v)


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def label(v):
    return ','.join(str(x) for x in v)


class FCCRegistration:
    def __init__(self, origin, basis, tolerance=.5):
        self.origin = np.array(origin, float)
        self.basis = np.array(basis, float)
        self.inverse = np.linalg.inv(self.basis)
        self.tolerance = float(tolerance)
        candidates = [np.array(v) for v in product(range(-2, 3), repeat=3) if any(v)]
        lengths = np.array([np.linalg.norm(v @ self.basis) for v in candidates])
        d = float(lengths.min())
        self.nn = [key(v) for v, length in zip(candidates, lengths) if abs(length-d) < .01*d]
        if len(self.nn) != 12:
            raise ValueError('Seed does not define a twelve-neighbor FCC reference')
        self.distance = d
        shell = {(0, 0, 0)}
        self.shells = {}
        for depth in (1, 2):
            shell = shell | {add(k, v) for k in shell for v in self.nn}
            self.shells[depth] = shell - {(0, 0, 0)}

    @classmethod
    def infer(cls, centers, tolerance=.5):
        centers = np.asarray(centers, float)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 4:
            raise ValueError('At least four three-dimensional seed centers required')
        vectors = centers[:, None, :] - centers[None, :, :]
        lengths = np.linalg.norm(vectors, axis=2)
        distance = float(lengths[lengths > 1e-6].min())
        near = vectors[(lengths > .99*distance) & (lengths < 1.01*distance)]
        unique = []
        for vector in near:
            if not any(np.linalg.norm(vector-old) < .01*distance for old in unique):
                unique.append(vector)
        selected = None
        for triple in combinations(unique, 3):
            basis = np.array(triple)
            gram = basis @ basis.T / distance**2
            if (abs(np.linalg.det(basis)) > .1*distance**3 and
                    all(abs(gram[i, j]-.5) < .02 for i, j in combinations(range(3), 2))):
                selected = basis
                break
        if selected is None:
            raise ValueError('No FCC primitive basis can be inferred from seed')
        origin = centers[0]
        integers = np.rint((centers-origin) @ np.linalg.inv(selected))
        fitted, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(centers)), integers]), centers, rcond=None)
        result = cls(fitted[0], fitted[1:], tolerance)
        _, residual = result.register(centers)
        if (residual > tolerance).any():
            raise ValueError('Seed lattice registration exceeds declared tolerance')
        return result

    def register(self, xyz, midpoint=False):
        xyz = np.asarray(xyz, float).reshape((-1, 3))
        scale = 2 if midpoint else 1
        fractional = (xyz-self.origin) @ self.inverse * scale
        integer = np.rint(fractional).astype(int)
        residual = np.linalg.norm(xyz-(self.origin + integer @ self.basis/scale), axis=1)
        return [key(v) for v in integer], residual

    def record(self):
        return {'origin_A': self.origin.tolist(), 'primitive_basis_A': self.basis.tolist(),
                'nearest_node_distance_A': self.distance, 'registration_tolerance_A': self.tolerance,
                'interior_rule': 'complete surrounding FCC shells, excluding tested center; conservative',
                'shell_neighbor_counts': {str(s): len(v) for s, v in self.shells.items()}}

    def count(self, node_centers, linker_centers, linker_degrees=None, shell_depth=1):
        if shell_depth not in self.shells:
            raise ValueError('Choose one or two surrounding shells')
        nk, nr = self.register(node_centers)
        lk, lr = self.register(linker_centers, midpoint=True)
        nodes = {k for k, r in zip(nk, nr) if r <= self.tolerance}
        occupied_midpoints = Counter(k for k, r in zip(lk, lr) if r <= self.tolerance)
        possible = nodes | {add(n, v) for n in nodes for v in self.nn}
        required = self.shells[shell_depth]
        enclosed = {n for n in possible if all(add(n, v) in nodes for v in required)}
        missing_nodes = enclosed - nodes
        edges = {}
        for a in nodes:
            for vector in self.nn:
                b = add(a, vector)
                if b in nodes and a < b:
                    midpoint = add(a, b)
                    if midpoint in edges and edges[midpoint] != (a, b):
                        raise ValueError('Ambiguous nearest-node linker midpoint')
                    edges[midpoint] = (a, b)
        eligible = {m for m, (a, b) in edges.items() if a in enclosed and b in enclosed}
        # An unregistered molecule near an eligible midpoint makes it ambiguous,
        # not automatically vacant. The same tolerance determines local ambiguity.
        ambiguous_midpoints = {k for k, r in zip(lk, lr) if r > self.tolerance and k in eligible}
        eligible -= ambiguous_midpoints
        missing_linkers = eligible - set(occupied_midpoints)
        dangling = 0 if linker_degrees is None else sum(d == 1 for d in linker_degrees)
        return {
            'shell_depth': shell_depth, 'registered_node_count': len(nodes),
            'unregistered_node_count': int(sum(nr > self.tolerance)),
            'duplicate_node_positions': len(nk)-int(sum(nr > self.tolerance))-len(nodes),
            'unregistered_linker_count': int(sum(lr > self.tolerance)),
            'linkers_outside_registered_node_pairs': sum(n for m, n in occupied_midpoints.items() if m not in edges),
            'duplicate_linker_positions': sum(max(0, n-1) for n in occupied_midpoints.values()),
            'interior_node_positions': len(enclosed), 'missing_interior_nodes': len(missing_nodes),
            'interior_linker_positions': len(eligible), 'missing_interior_linkers': len(missing_linkers),
            'missing_node_fraction': len(missing_nodes)/len(enclosed) if enclosed else None,
            'missing_linker_fraction': len(missing_linkers)/len(eligible) if eligible else None,
            'missing_node_keys': sorted(label(n) for n in missing_nodes),
            'missing_linker_keys': sorted(label(m) for m in missing_linkers),
            'eligible_linker_keys': sorted(label(m) for m in eligible),
            'occupied_linker_keys': sorted(label(m) for m in occupied_midpoints),
            'ambiguous_linker_positions': len(ambiguous_midpoints),
            'unenclosed_candidate_positions': len(possible-enclosed),
            'singly_attached_BDC_total': dangling,
            'claim_boundary': 'registered, complete-shell-enclosed positions only; not all clustered/disordered defects',
        }


def assembly_coordinates(assembly):
    nodes, linkers, degrees = [], [], []
    for entity in assembly.entities.items:
        if entity.entity_type == 'Zr':
            node = np.array([x for e, x in zip(entity.elements, entity.coordinates) if e == 'Zr'])
            nodes.append(node.mean(0))
        else:
            linkers.append(entity.coordinates.mean(0))
            degrees.append(len(entity.connected_entities))
    return np.array(nodes).reshape((-1, 3)), np.array(linkers).reshape((-1, 3)), degrees


class VacancyCohort:
    def __init__(self):
        self.seen = set()

    def update(self, measurement):
        current = set(measurement['missing_linker_keys'])
        occupied = set(measurement['occupied_linker_keys'])
        eligible = set(measurement['eligible_linker_keys'])
        previous = set(self.seen)
        result = {'previously_observed_vacancy_positions': len(previous),
                  'previous_positions_now_occupied': len(previous & occupied),
                  'previous_positions_still_missing': len(previous & current),
                  'previous_positions_outside_current_mask': len(previous-occupied-eligible),
                  'newly_observed_vacancy_positions': len(current-previous)}
        self.seen |= current
        return result
