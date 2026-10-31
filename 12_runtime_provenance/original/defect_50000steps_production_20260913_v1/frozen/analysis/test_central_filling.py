from itertools import product
import unittest

import numpy as np

from central_filling import CentralLattice, FilledEnvelope, analyze, core_rotation, triangle_distance, assign_observed


class FillingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = np.array([[0., 10., 10.], [10., 0., 10.], [10., 10., 0.]])
        cls.d = np.linalg.norm(cls.basis[0])
        keys = np.array(list(product(range(-4, 5), repeat=3)))
        xyz = keys @ cls.basis
        keep = np.linalg.norm(xyz, axis=1) <= 2.7*cls.d
        cls.nodes = xyz[keep]
        cls.lattice = CentralLattice([0, 0, 0], cls.basis).enumerate(cls.nodes)
        present = {tuple(k) for k in keys[keep]}
        cls.linkers = np.array([cls.lattice.link_xyz[i] for i, (a, b) in enumerate(cls.lattice.endpoints)
                               if tuple(cls.lattice.node_keys[a]) in present and tuple(cls.lattice.node_keys[b]) in present])
        cls.pristine = analyze(cls.lattice, cls.nodes, cls.linkers)

    def test_pristine_nonzero_interior_and_zero_vacancy(self):
        r = self.pristine
        self.assertGreater(r['interior_linker_positions'], 0)
        self.assertEqual(r['missing_interior_nodes'], 0)
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_single_missing_linker_inside(self):
        index = int(np.argmin(np.linalg.norm(self.linkers, axis=1)))
        r = analyze(self.lattice, self.nodes, np.delete(self.linkers, index, axis=0))
        self.assertEqual(r['missing_interior_linkers'], 1)
        self.assertEqual(r['missing_interior_nodes'], 0)

    def test_missing_node_and_its_linkers_not_double_counted(self):
        index = int(np.argmin(np.linalg.norm(self.nodes, axis=1)))
        # Remove center and its twelve incident midpoint linkers.
        linkers = self.linkers[np.linalg.norm(self.linkers, axis=1) > .51*self.d]
        r = analyze(self.lattice, np.delete(self.nodes, index, axis=0), linkers)
        self.assertEqual(r['missing_interior_nodes'], 1)
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_present_dangling_linker_not_vacancy(self):
        degrees = [2]*len(self.linkers)
        degrees[int(np.argmin(np.linalg.norm(self.linkers, axis=1)))] = 1
        r = analyze(self.lattice, self.nodes, self.linkers, degrees)
        self.assertEqual(r['present_singly_attached_interior_BDC'], 1)
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_removed_surface_linker_not_interior(self):
        surface = next(row for row in self.pristine['linker_rows'] if row['occupied'] and not row['interior'])
        index = int(np.argmin(np.linalg.norm(self.linkers-surface['xyz_A'], axis=1)))
        r = analyze(self.lattice, self.nodes, np.delete(self.linkers, index, axis=0))
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_rigid_motion_invariance(self):
        angle = .48
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        translation = np.array([20., -13., 9.])
        nodes, linkers = self.nodes @ rotation+translation, self.linkers @ rotation+translation
        lattice = CentralLattice(translation, self.basis @ rotation).enumerate(nodes)
        result = analyze(lattice, nodes, linkers)
        for name in ('interior_linker_positions', 'interior_node_positions', 'missing_interior_linkers'):
            self.assertEqual(result[name], self.pristine[name])
        octa = np.vstack([np.eye(3), -np.eye(3)])
        fitted, rmsd = core_rotation(octa, octa @ rotation+translation)
        matched = np.linalg.norm((octa @ fitted)[:, None, :]-(octa @ rotation)[None, :, :], axis=2).min(1)
        np.testing.assert_allclose(matched, 0., atol=1e-12)
        self.assertLess(rmsd, 1e-12)

    def test_enclosed_hole_filled_but_open_notch_not_sphere(self):
        # Large open notch connects the center to outside. Shape reconstruction
        # must not use a single outer radius to fill that missing sector.
        keep = ~((self.nodes[:, 0] > -.5*self.d) &
                 (np.abs(self.nodes[:, 1]) < 1.35*self.d) &
                 (np.abs(self.nodes[:, 2]) < 1.35*self.d))
        envelope = FilledEnvelope(self.nodes[keep], self.d)
        inside, _ = envelope.locate([[.8*self.d, 0, 0]])
        self.assertFalse(inside[0])
        missing_center = np.delete(self.nodes, np.argmin(np.linalg.norm(self.nodes, axis=1)), 0)
        enclosed = FilledEnvelope(missing_center, self.d, alpha=.85)
        inside, depth = enclosed.locate([[0., 0., 0.]])
        self.assertTrue(inside[0])
        self.assertGreater(depth[0], .35*self.d)
        self.assertGreater(enclosed.enclosed_void_tetrahedra_filled, 0)

    def test_unregistered_entity_is_ambiguous_not_false_vacancy(self):
        linkers = self.linkers.copy()
        linkers[np.argmin(np.linalg.norm(linkers, axis=1))] += [.8, 0, 0]
        r = analyze(self.lattice, self.nodes, linkers)
        self.assertEqual(r['missing_interior_linkers'], 0)
        self.assertEqual(r['interior_unresolved_linker_positions'], 1)

    def test_exact_cartesian_assignment_not_fractional_rounding(self):
        # This point is closer to [0,1,0] than [0,0,0] in the FCC skew basis.
        point = np.array([[.49, .49, .0]]) @ self.basis
        expected = np.array(list(product(range(-1, 2), repeat=3))) @ self.basis
        a = assign_observed(point, expected, 20.)
        distance = np.linalg.norm(expected-point, axis=1)
        self.assertAlmostEqual(a['residual'][0], distance.min())
        self.assertLess(a['residual'][0], np.linalg.norm(point))

    def test_triangle_distance_uses_edges(self):
        t = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
        d = triangle_distance([[.2, .2, 1], [2, 0, 0]], t)
        np.testing.assert_allclose(d, [1., 1.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
