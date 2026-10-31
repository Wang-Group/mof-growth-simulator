import math
from itertools import product
import unittest

import numpy as np

from pool_clock import PoolClock
from vacancy_counter import FCCRegistration, VacancyCohort


class PoolTests(unittest.TestCase):
    def pool(self, mode='finite', policy='retain_oligomers'):
        return PoolClock({'Zr': 1000, 'Ligand': 6000}, {'Zr': 16, 'Ligand': 60}, mode, policy, 1.)

    def test_initial_normalization_and_once_only_depletion(self):
        p = self.pool()
        c0 = p.coefficients()
        for _ in range(500):
            p.incorporate('Zr')
        for _ in range(3000):
            p.incorporate('Ligand')
        c = p.coefficients()
        self.assertAlmostEqual(c['add_node']/c0['add_node'], .5)
        self.assertAlmostEqual(c['add_linker']/c0['add_linker'], .5)
        self.assertAlmostEqual(c['p_Zr'], 1/7)
        self.assertEqual(c['rho_off_total_external'], 2.)
        self.assertEqual(c['rho_off_linker_reference'], 2.)
        for name in ('close_internal', 'remove_bond', 'solvent_to_modulator', 'modulator_to_solvent'):
            self.assertEqual(c0[name], c[name])

    def test_selective_depletion_changes_partition(self):
        p = self.pool()
        for _ in range(500):
            p.incorporate('Zr')
        c = p.coefficients()
        self.assertAlmostEqual(c['p_Zr'], 1/13)
        self.assertEqual(c['rho_off_linker_reference'], 1.)
        self.assertGreater(c['rho_off_total_external'], 1.)

    def test_fragment_policies_and_origin_independence(self):
        for mode in ('fixed', 'finite'):
            for policy in ('retain_oligomers', 'redissolve'):
                p = self.pool(mode, policy)
                p.release(['Zr', 'Ligand'])
                p.release(['Ligand'])
                p.check()
                self.assertEqual(p.tracked, {'Zr': 15, 'Ligand': 58})
                self.assertEqual(p.detached['Zr'], int(policy == 'retain_oligomers'))
                if mode == 'fixed':
                    self.assertEqual(p.free, p.initial_free)
                else:
                    self.assertGreater(p.free['Ligand'], 6000)

    def test_fixed_reservoir_source_sink_ledger(self):
        p = self.pool('fixed')
        p.incorporate('Zr')
        p.release(['Zr'])
        self.assertEqual(p.supplied['Zr'], 1)
        self.assertEqual(p.returned['Zr'], 1)
        self.assertEqual(p.tracked, p.initial_tracked)
        p.check()

    def test_exhaustion_and_invalid_policy(self):
        p = PoolClock({'Zr': 1, 'Ligand': 1}, {'Zr': 0, 'Ligand': 0}, 'finite', 'redissolve', 1)
        p.incorporate('Zr')
        p.incorporate('Ligand')
        self.assertIsNone(p.coefficients()['p_Zr'])
        self.assertEqual(p.coefficients()['add_node'], 0.)
        with self.assertRaises(ValueError):
            p.incorporate('Zr')
        with self.assertRaises(ValueError):
            self.pool(policy='seed_specific')


class LocalThermodynamicTests(unittest.TestCase):
    def test_resolved_microstate_pair_fluxes_with_caps_and_angles(self):
        # Explicit F, D_h,c and V_c,c' states. State measure includes each prior.
        # This test is a closed local reference, not global detailed balance of
        # a growth graph with deletion of detached oligomers.
        for delta, rho, a in product((0., 2., 10.), (.1, 1., 3.), (.2, .8)):
            E = math.exp(30.9)
            for angle_prior in (1/8, 1/16):
                for cap_prior, other_prior in product((.5, .25), repeat=2):
                    f = 1.
                    d = rho/E*math.exp(-delta)*angle_prior*cap_prior
                    v = rho*rho/(E*a)*cap_prior*other_prior
                    fd = rho*math.exp(-delta)*angle_prior*cap_prior
                    df = E
                    dv = rho*other_prior
                    vd = a*math.exp(-delta)*angle_prior
                    self.assertAlmostEqual((f*fd)/(d*df), 1.)
                    self.assertAlmostEqual((d*dv)/(v*vd), 1.)

    def test_joint_state_energy_correction_ratio(self):
        for before, after, delta in product(range(5), range(5), (0., 2., 10.)):
            f = math.exp(-max(0, delta*(after-before)))
            r = math.exp(-max(0, delta*(before-after)))
            self.assertAlmostEqual(math.log(f/r), -delta*(after-before))


class VacancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = FCCRegistration([2., 3., 5.], [[0., 10., 10.], [10., 0., 10.], [10., 10., 0.]])
        cls.keys = list(product(range(-3, 4), repeat=3))
        cls.nodes = cls.reg.origin + np.array(cls.keys) @ cls.reg.basis
        node_set = set(cls.keys)
        cls.midpoints = sorted({tuple(x+y for x, y in zip(a, b)) for a in cls.keys
                                for v in cls.reg.nn
                                for b in [tuple(x+y for x, y in zip(a, v))]
                                if b in node_set and a < b})
        cls.linkers = cls.reg.origin + np.array(cls.midpoints) @ cls.reg.basis/2

    def test_pristine_and_inferred_basis(self):
        inferred = FCCRegistration.infer(self.nodes)
        for reg in (self.reg, inferred):
            for depth in (1, 2):
                result = reg.count(self.nodes, self.linkers, shell_depth=depth)
                self.assertEqual(result['missing_interior_linkers'], 0)
                self.assertEqual(result['missing_interior_nodes'], 0)
                self.assertGreater(result['interior_linker_positions'], 0)

    def test_one_missing_linker_and_cohort_repair(self):
        remove = self.midpoints.index((0, 0, 1))
        for depth in (1, 2):
            r = self.reg.count(self.nodes, np.delete(self.linkers, remove, 0), shell_depth=depth)
            self.assertEqual(r['missing_interior_linkers'], 1)
            self.assertEqual(r['missing_interior_nodes'], 0)
            cohort = VacancyCohort()
            self.assertEqual(cohort.update(r)['newly_observed_vacancy_positions'], 1)
            repaired = self.reg.count(self.nodes, self.linkers, shell_depth=depth)
            self.assertEqual(cohort.update(repaired)['previous_positions_now_occupied'], 1)

    def test_missing_node_does_not_mark_itself_exterior(self):
        remove = self.keys.index((0, 0, 0))
        for depth in (1, 2):
            r = self.reg.count(np.delete(self.nodes, remove, 0), self.linkers, shell_depth=depth)
            self.assertEqual(r['missing_interior_nodes'], 1)
            self.assertEqual(r['missing_interior_linkers'], 0)

    def test_surface_missing_linker_not_interior_defect(self):
        remove = self.midpoints.index((-6, -6, -5))
        r = self.reg.count(self.nodes, np.delete(self.linkers, remove, 0))
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_graph_opening_is_not_absent_molecule(self):
        degrees = [2]*len(self.linkers)
        degrees[self.midpoints.index((0, 0, 1))] = 1
        r = self.reg.count(self.nodes, self.linkers, degrees)
        self.assertEqual(r['singly_attached_BDC_total'], 1)
        self.assertEqual(r['missing_interior_linkers'], 0)

    def test_off_lattice_molecule_near_slot_is_ambiguous(self):
        bad = self.linkers.copy()
        bad[self.midpoints.index((0, 0, 1))] += [1., 0., 0.]
        r = self.reg.count(self.nodes, bad)
        self.assertEqual(r['ambiguous_linker_positions'], 1)
        self.assertEqual(r['missing_interior_linkers'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
