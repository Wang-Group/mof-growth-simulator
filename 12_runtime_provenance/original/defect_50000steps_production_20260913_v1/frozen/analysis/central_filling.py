"""Read-only central-node reference lattice and finite-particle filling model."""
from collections import Counter, deque
from itertools import combinations, product, permutations

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def label(key):
    return ','.join(str(int(x)) for x in key)


def core_center(entity):
    zr = np.asarray([x for symbol, x in zip(entity.elements, entity.coordinates) if symbol == 'Zr'])
    if zr.shape != (6, 3):
        raise ValueError('A six-zirconium core is required')
    return zr.mean(0), zr


def core_rotation(reference, target):
    """Fit the unlabeled six-Zr octahedron; ordering is not physical orientation.

    Cubic-equivalent rotations produce the same full FCC position set. Minimize
    over the 720 assignments rather than relying on atom order after alignment.
    """
    x = np.asarray(reference)-np.mean(reference, axis=0)
    y = np.asarray(target)-np.mean(target, axis=0)
    targets = y[np.array(list(permutations(range(6))))]
    u, _, vt = np.linalg.svd(np.einsum('ni,pnj->pij', x, targets))
    correction = np.repeat(np.eye(3)[None], len(targets), axis=0)
    correction[:, 2, 2] = np.linalg.det(u @ vt)
    rotations = u @ correction @ vt
    errors = np.sqrt(np.mean(np.sum((np.einsum('ni,pij->pnj', x, rotations)-targets)**2, axis=2), axis=1))
    best = int(np.argmin(errors))
    rotation, residual = rotations[best], float(errors[best])
    if residual > .1:
        raise ValueError('Core-template atom correspondence or local distortion requires review')
    return rotation, residual


class CentralLattice:
    def __init__(self, center, basis, metadata=None):
        self.center = np.asarray(center, float)
        self.basis = np.asarray(basis, float)
        self.inverse = np.linalg.inv(self.basis)
        integer = np.array([v for v in product(range(-1, 2), repeat=3) if any(v)], int)
        lengths = np.linalg.norm(integer @ self.basis, axis=1)
        self.distance = float(lengths.min())
        self.nn = integer[np.abs(lengths-self.distance) < .01*self.distance]
        if len(self.nn) != 12:
            raise ValueError('Reference does not define FCC twelve-neighbor positions')
        self.metadata = metadata or {}

    @classmethod
    def from_assemblies(cls, reference_assembly, assembly, anchor_rank=0):
        # Import from the frozen integration path supplied by the caller.
        from vacancy_counter import FCCRegistration
        ref_nodes = [e for e in reference_assembly.entities.items if e.entity_type == 'Zr']
        nodes = [e for e in assembly.entities.items if e.entity_type == 'Zr']
        ref_centers = np.array([core_center(e)[0] for e in ref_nodes])
        centers = np.array([core_center(e)[0] for e in nodes])
        lattice = FCCRegistration.infer(ref_centers)
        ref_index = int(np.argmin(np.linalg.norm(ref_centers-ref_centers.mean(0), axis=1)))
        order = np.argsort(np.linalg.norm(centers-centers.mean(0), axis=1), kind='stable')
        index = int(order[anchor_rank])
        center, zr = core_center(nodes[index])
        rotation, error = core_rotation(core_center(ref_nodes[ref_index])[1], zr)
        return cls(center, lattice.basis @ rotation,
                   {'anchor_rank': anchor_rank, 'anchor_node_index': index,
                    'anchor_distance_to_node_centroid_A': float(np.linalg.norm(center-centers.mean(0))),
                    'orientation_fit_core_RMSD_A': error, 'reference_node_index': ref_index,
                    'orientation_source': 'unlabeled six-Zr octahedron; proper rotation with assignment search'})

    def enumerate(self, observed_nodes):
        frac = (np.asarray(observed_nodes)-self.center) @ self.inverse
        low, high = np.floor(frac.min(0)).astype(int)-2, np.ceil(frac.max(0)).astype(int)+2
        if np.prod(high-low+1) > 500000:
            raise ValueError('Candidate lattice bounding box too large; inspect reference first')
        keys = np.array(list(product(*(range(a, b+1) for a, b in zip(low, high)))), int)
        xyz = self.center + keys @ self.basis
        radius = np.linalg.norm(np.asarray(observed_nodes)-self.center, axis=1).max()+self.distance*1.5
        keep = np.linalg.norm(xyz-self.center, axis=1) <= radius
        self.node_keys, self.node_xyz = keys[keep], xyz[keep]
        lookup = {tuple(k): i for i, k in enumerate(self.node_keys)}
        links = {}
        for i, a in enumerate(self.node_keys):
            for vector in self.nn:
                b = a+vector
                j = lookup.get(tuple(b))
                if j is not None and i < j:
                    midpoint = tuple(a+b)
                    if midpoint in links:
                        raise ValueError('Nonunique FCC edge midpoint')
                    links[midpoint] = (i, j)
        self.link_keys = np.array(list(links), int)
        self.endpoints = np.array(list(links.values()), int)
        self.link_xyz = self.center + self.link_keys @ self.basis/2
        return self

    def record(self):
        return {**self.metadata, 'center_A': self.center.tolist(), 'basis_A': self.basis.tolist(),
                'nearest_node_distance_A': self.distance}


def triangle_distance(points, triangles, batch=96):
    """Euclidean distance to the finite triangular surface, not infinite planes."""
    points, triangles = np.asarray(points), np.asarray(triangles)
    if not len(triangles):
        raise ValueError('No boundary triangles')
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    norm2 = np.einsum('ij,ij->i', normal, normal)
    good = norm2 > 1e-16
    a, b, c, ab, ac, normal, norm2 = [x[good] for x in (a, b, c, ab, ac, normal, norm2)]
    d00, d01, d11 = (np.einsum('ij,ij->i', x, y) for x, y in ((ab, ab), (ab, ac), (ac, ac)))
    denom = d00*d11-d01*d01
    answer = []
    for start in range(0, len(points), batch):
        p = points[start:start+batch, None, :]
        ap = p-a
        height = np.einsum('pij,ij->pi', ap, normal)/norm2
        projected = ap-height[..., None]*normal
        d20 = np.einsum('pij,ij->pi', projected, ab)
        d21 = np.einsum('pij,ij->pi', projected, ac)
        u = (d11*d20-d01*d21)/denom
        v = (d00*d21-d01*d20)/denom
        inside = (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10)
        distance2 = np.where(inside, height*height*norm2, np.inf)
        for begin, end in ((a, b), (b, c), (c, a)):
            edge = end-begin
            t = np.clip(np.einsum('pij,ij->pi', p-begin, edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
            diff = p-(begin+t[..., None]*edge)
            distance2 = np.minimum(distance2, np.einsum('pij,pij->pi', diff, diff))
        answer.extend(np.sqrt(distance2.min(axis=1)))
    return np.array(answer)


class FilledEnvelope:
    def __init__(self, node_xyz, distance, alpha=1.05):
        self.nodes = np.asarray(node_xyz, float)
        self.distance, self.alpha = float(distance), float(alpha)
        self._locations = {}
        if len(self.nodes) < 4:
            raise ValueError('At least four noncoplanar node centers needed')
        self.triangulation = Delaunay(self.nodes)
        vertices = self.nodes[self.triangulation.simplices]
        radii = np.full(len(vertices), np.inf)
        for i, tetra in enumerate(vertices):
            local = tetra[1:]-tetra[0]
            try:
                center = np.linalg.solve(2*local, np.einsum('ij,ij->i', local, local))
                radii[i] = np.linalg.norm(center)
            except np.linalg.LinAlgError:
                pass
        solid = radii <= alpha*distance*(1+1e-10)
        neighbors = self.triangulation.neighbors
        outside = set(int(i) for i in np.flatnonzero(~solid & np.any(neighbors < 0, axis=1)))
        queue = deque(sorted(outside))
        while queue:
            i = queue.popleft()
            for j in neighbors[i]:
                if j >= 0 and not solid[j] and int(j) not in outside:
                    outside.add(int(j))
                    queue.append(int(j))
        self.filled = np.array([i not in outside for i in range(len(solid))])
        self.enclosed_void_tetrahedra_filled = int(np.sum(self.filled & ~solid))
        faces = []
        for i in np.flatnonzero(self.filled):
            tetra = self.triangulation.simplices[i]
            for omit, j in enumerate(neighbors[i]):
                if j < 0 or not self.filled[j]:
                    faces.append(np.delete(tetra, omit))
        self.triangles = self.nodes[np.array(faces, int)] if faces else np.empty((0, 3, 3))

    def locate(self, xyz):
        original = xyz
        cache = self._locations.get(id(original))
        if cache is not None and cache[0] is original:
            return cache[1], cache[2]
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        simplex = self.triangulation.find_simplex(xyz, tol=1e-9)
        inside = (simplex >= 0) & self.filled[np.maximum(simplex, 0)]
        depth = np.zeros(len(xyz))
        if inside.any():
            depth[inside] = triangle_distance(xyz[inside], self.triangles)
        if isinstance(original, np.ndarray):
            self._locations[id(original)] = (original, inside, depth)
        return inside, depth


def assign_observed(xyz, expected, tolerance):
    residual, index = cKDTree(expected).query(np.asarray(xyz).reshape(-1, 3))
    matched = residual <= tolerance
    counts = Counter(int(i) for i in index[matched])
    occupied = set(counts)
    ambiguous = set(int(i) for i in index[~matched])
    duplicates = {i for i, count in counts.items() if count > 1}
    return {'index': index, 'residual': residual, 'occupied': occupied,
            'ambiguous': ambiguous | duplicates, 'duplicates': duplicates}


def analyze(lattice, node_xyz, linker_xyz, linker_degrees=None, *, alpha=1.05,
            boundary=0.35, tolerance=.5, detail=True, envelope=None):
    envelope = envelope or FilledEnvelope(node_xyz, lattice.distance, alpha)
    ni, nd = envelope.locate(lattice.node_xyz)
    li, ld = envelope.locate(lattice.link_xyz)
    na = assign_observed(node_xyz, lattice.node_xyz, tolerance)
    la = assign_observed(linker_xyz, lattice.link_xyz, tolerance)
    nint = set(int(i) for i in np.flatnonzero(ni & (nd > boundary*lattice.distance)))
    lint = set(int(i) for i in np.flatnonzero(li & (ld > boundary*lattice.distance)))
    n_eligible = nint-na['ambiguous']
    n_missing = n_eligible-na['occupied']
    supported = {i for i in lint if all(int(n) in na['occupied'] and int(n) not in na['ambiguous']
                                      for n in lattice.endpoints[i])}
    l_eligible = supported-la['ambiguous']
    l_missing = l_eligible-la['occupied']
    radii = np.linalg.norm(lattice.node_xyz-lattice.center, axis=1)/lattice.distance
    profiles = []
    for lo in np.arange(0, radii[ni].max(initial=0)+.5, .5):
        indices = set(int(i) for i in np.flatnonzero(ni & (radii >= lo-1e-10) & (radii < lo+.5-1e-10)))
        if indices:
            profiles.append({'r_lo_over_dNN': float(lo), 'r_hi_over_dNN': float(lo+.5),
                             'predicted_nodes_in_shape': len(indices),
                             'occupied_nodes': len(indices & na['occupied']),
                             'ambiguous_nodes': len(indices & na['ambiguous'])})
    occupied_linkers = len(l_eligible & la['occupied'])
    result = {'reference': lattice.record(), 'settings': {'alpha': alpha, 'surface_buffer_over_dNN': boundary,
                                                         'registration_tolerance_A': tolerance},
              'node_count': len(node_xyz), 'linker_count': len(linker_xyz),
              'registered_nodes': int(np.sum(na['residual'] <= tolerance)),
              'registered_linkers': int(np.sum(la['residual'] <= tolerance)),
              'node_residual_A_p50_p90_max': np.quantile(na['residual'], [.5, .9, 1.]).tolist(),
              'linker_residual_A_p50_p90_max': np.quantile(la['residual'], [.5, .9, 1.]).tolist(),
              'interior_node_positions': len(n_eligible), 'missing_interior_nodes': len(n_missing),
              'interior_linker_positions': len(l_eligible), 'occupied_interior_linkers': occupied_linkers,
              'missing_interior_linkers': len(l_missing),
              'missing_linker_fraction': len(l_missing)/len(l_eligible) if l_eligible else None,
              'interior_unresolved_node_positions': len(nint & na['ambiguous']),
              'interior_unresolved_linker_positions': len(supported & la['ambiguous']),
              'interior_linker_positions_without_two_registered_nodes': len(lint-supported),
              'outer_layer_predicted_linker_positions': int(li.sum())-len(lint),
              'filled_enclosed_void_tetrahedra': envelope.enclosed_void_tetrahedra_filled,
              'radial_node_filling': profiles,
              'claim_boundary': 'candidate vacancies in the central-reference reconstructed envelope; inspect sensitivity'}
    if linker_degrees is not None:
        result['present_singly_attached_interior_BDC'] = sum(
            int(degree == 1 and int(index) in l_eligible and residual <= tolerance)
            for degree, index, residual in zip(linker_degrees, la['index'], la['residual']))
    if detail:
        for kind, xyz, assignment, eligible, keys, depth in (
                ('node', node_xyz, na, n_eligible, lattice.node_keys, nd),
                ('linker', linker_xyz, la, l_eligible, lattice.link_keys, ld)):
            observed = []
            for entity_index, (position, index, residual) in enumerate(zip(xyz, assignment['index'], assignment['residual'])):
                uncertain = residual > tolerance or int(index) in assignment['ambiguous']
                classification = 'unresolved_mapping' if uncertain else (
                    'interior' if int(index) in eligible else 'surface_or_boundary')
                observed.append({'species_index': entity_index, 'xyz_A': np.asarray(position).tolist(),
                                 'nearest_predicted_key': label(keys[index]), 'residual_A': float(residual),
                                 'radius_from_center_A': float(np.linalg.norm(position-lattice.center)),
                                 'predicted_position_depth_A': float(depth[index]),
                                 'classification': classification})
            result['observed_'+kind+'_rows'] = observed
        result['node_rows'] = [dict(key=label(lattice.node_keys[i]), xyz_A=lattice.node_xyz[i].tolist(),
                                   radius_over_dNN=float(radii[i]), depth_over_dNN=float(nd[i]/lattice.distance),
                                   occupied=i in na['occupied'], ambiguous=i in na['ambiguous'],
                                   inside=bool(ni[i]), interior=i in n_eligible, missing=i in n_missing)
                               for i in range(len(lattice.node_keys))]
        result['linker_rows'] = [dict(key=label(lattice.link_keys[i]), xyz_A=lattice.link_xyz[i].tolist(),
                                     endpoint_keys=[label(lattice.node_keys[n]) for n in lattice.endpoints[i]],
                                     depth_over_dNN=float(ld[i]/lattice.distance),
                                     occupied=i in la['occupied'], ambiguous=i in la['ambiguous'],
                                     inside=bool(li[i]), interior=i in l_eligible, missing=i in l_missing)
                                 for i in range(len(lattice.link_keys))]
    return result
