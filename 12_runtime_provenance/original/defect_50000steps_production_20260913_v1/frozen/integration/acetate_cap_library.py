"""Acetate cap geometry extracted from the existing explicit-H Zr6 model.

Only a complete C2H3O2 fragment is used. No core atoms, regenerated heavy-atom
coordinates, force-field energies, or benzoate geometry enter this library.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np


class AcetateCapLibrary:
    def __init__(self, source):
        self.source = Path(source)
        lines = self.source.read_text().splitlines()
        n, nb = int(lines[3][:3]), int(lines[3][3:6])
        elements = [line.split()[3] for line in lines[4:4+n]]
        xyz = np.array([[float(v) for v in line.split()[:3]] for line in lines[4:4+n]])
        adj = [set() for _ in elements]
        for line in lines[4+n:4+n+nb]:
            a, b = int(line[:3])-1, int(line[3:6])-1
            adj[a].add(b)
            adj[b].add(a)
        anchors, fragments = [], []
        for c, element in enumerate(elements):
            oxygens = sorted(i for i in adj[c] if elements[i] == 'O')
            if element != 'C' or len(oxygens) != 2:
                continue
            found, stack = set(), [c]
            while stack:
                i = stack.pop()
                if i in found or elements[i] == 'Zr':
                    continue
                found.add(i)
                stack.extend(j for j in adj[i] if elements[j] != 'Zr')
            if Counter(elements[i] for i in found) != {'C': 2, 'H': 3, 'O': 2}:
                raise ValueError('Expected complete acetate including three methyl hydrogens')
            methyl = next(i for i in found if elements[i] == 'C' and i != c)
            if {i for i in found if elements[i] == 'H'} != {i for i in adj[methyl] if elements[i] == 'H'}:
                raise ValueError('Acetate hydrogens must be attached to methyl carbon')
            anchors.append([c, *oxygens])
            fragments.append(found)
        if len(fragments) != 12 or any(a & b for i, a in enumerate(fragments) for b in fragments[i+1:]):
            raise ValueError('Expected twelve distinct acetate fragments in source model')
        self.source_indices = sorted(fragments[0])
        self.elements = [elements[i] for i in self.source_indices]
        self.xyz = xyz[self.source_indices].copy()
        self.anchor = [self.source_indices.index(i) for i in anchors[0]]
        self.identity = {'identity': 'acetate', 'atom_counts': dict(Counter(self.elements))}
        self.provenance = {
            'source': str(self.source),
            'source_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
            'source_atom_indices_zero_based': self.source_indices,
            'anchor_indices_in_fragment': self.anchor,
            'geometry': 'first existing explicit-H acetate fragment; all source coordinates retained',
            'candidate_library': 'one rigid source geometry, two oxygen assignments',
            'energy': 'none; not an optimized intermediate or calibrated barrier',
        }

    def candidates(self, site):
        from cap_geometry_preflight import proper_fit
        target = site.belonging_entity.coordinates[site.atom_indexes]
        result = []
        for flip in (False, True):
            anchor = [self.anchor[0], self.anchor[2], self.anchor[1]] if flip else self.anchor
            rot, shift, error = proper_fit(self.xyz[anchor], target)
            result.append({'elements': list(self.elements), 'xyz': self.xyz @ rot + shift,
                           'conformer': 'existing_acetate_fragment_01', 'oxygen_swapped': flip,
                           'anchor_rmsd_A': error})
        return result
