import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from tools import audit_fig5c as audit


class Figure5CReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = audit.core()
        cls.frame, cls.axes, cls.cube = cls.mod.load_data()

    def test_packaged_sources_and_exact_probabilities(self):
        self.assertEqual(audit.source_checks(), 5)
        self.assertEqual(self.cube.shape, (10, 14, 9))
        np.testing.assert_array_equal(self.frame.P, self.frame.persistent_assembly_probability)

    def test_relative_svg_paths(self):
        actual = audit.path_points('m 10,20 2,-3 -1,4 z')
        np.testing.assert_array_equal(actual, [[10,20],[12,17],[11,21]])
        np.testing.assert_array_equal(actual, audit.path_points('M 10 20 L 12 17 L 11 21 Z'))
        np.testing.assert_array_equal(audit.path_points('M 1e-3 2E-2 l 1 2'), [[.001,.02],[1.001,2.02]])
        with self.assertRaises(ValueError):
            audit.path_points('M 0 0 C 1 2 3 4 5 6')

    def test_reference_camera_and_author_composite(self):
        _, triangles, camera, result = audit.reference()
        self.assertEqual(triangles.shape, (14488,3,2))
        self.assertEqual(camera.shape, (3,4))
        self.assertLess(result['camera_residual_max_pt'], 1e-5)
        self.assertLess(result['author_vertex_difference_max_pt'], .001)

    def test_exact_roots_and_no_extrapolation(self):
        pairs = np.array([[-2,0], [2,2], [.3,.7]])
        rho = self.mod.exact_roots(pairs, self.axes, self.cube)
        p = RegularGridInterpolator(self.axes, self.cube)(np.c_[rho,pairs])
        np.testing.assert_allclose(p, .5, atol=1e-12)
        with self.assertRaises(ValueError):
            self.mod.exact_roots([[3,1]], self.axes, self.cube)
        axes = (np.arange(4.), np.array([0.,1.]), np.array([0.,1.]))
        plateau = np.broadcast_to(np.array([1.,.5,.5,0.])[:,None,None], (4,2,2))
        np.testing.assert_allclose(self.mod.exact_roots([[.3,.2]], axes, plateau), [2.])

    def test_analytic_plane_and_grid_change(self):
        axes = (np.array([0.,1.]),)*3
        cube = np.broadcast_to(np.array([1.,0.])[:,None,None], (2,2,2))
        for grid in [(5,6,7),(9,10,11)]:
            triangles = self.mod.mesh(axes, cube, grid)
            np.testing.assert_allclose(triangles[:,:,0], .5, atol=1e-7)
            self.assertLess(self.mod.mesh_errors(triangles, axes, cube)['centroid_P_error_max'], 1e-7)

    def test_invalid_grid_and_changed_source_input_fail(self):
        for grid in [(2,3,3), (3,3), (3.,3,3), (200,200,200)]:
            with self.assertRaises(ValueError):
                self.mod.mesh(self.axes, self.cube, grid)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'wrong.csv'
            path.write_text('wrong input', encoding='utf-8')
            with self.assertRaises(ValueError):
                self.mod.load_data(path)

    def test_projection_mask_identity_and_clipping(self):
        triangles = np.array([[[1.,1.],[1.,5.],[5.,1.]]])
        result = audit.overlap(triangles, triangles, np.array([0.,0.]), 4, np.array([30,30]))
        self.assertEqual(result['iou'], 1)
        self.assertEqual(result['edge_max_svg_pt'], 0)
        with self.assertRaises(ValueError):
            audit.mask(triangles, np.array([0.,0.]), 4, np.array([10,10]))


if __name__ == '__main__':
    unittest.main()
