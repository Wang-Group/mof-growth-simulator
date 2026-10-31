from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from common_kmc_selector import (
    OpportunityCounts,
    PhysicalCoordinates,
    bond_removal_pool,
    compute_propensities,
    ideal_node_fraction,
    propensities_from_weights,
)


def test_every_framework_bond_has_equal_opening_weight() -> None:
    pair_a = (object(), object())
    pair_b = (object(), object())
    assembly = SimpleNamespace(
        linked_carboxylate_pairs=SimpleNamespace(items=[pair_a, pair_b])
    )
    bonds, weights = bond_removal_pool(assembly)
    assert bonds == [pair_a, pair_b]
    assert weights == [1.0, 1.0]


def test_site_resolved_propensities() -> None:
    counts = OpportunityCounts(3, 11, 2, 5)
    coordinates = PhysicalCoordinates(
        external_activity=4.0,
        node_fraction=0.25,
        off_weight=0.5,
        closure_entropy_coefficient=0.0,
    )
    prop = compute_propensities(counts, coordinates)
    assert prop.add_node == pytest.approx(3.0)
    assert prop.add_linker == pytest.approx(33.0)
    assert prop.close_internal == pytest.approx(2.0)
    assert prop.remove_bond == pytest.approx(2.5)


def test_zero_activity_disables_corresponding_species() -> None:
    counts = OpportunityCounts(7, 9, 0, 0)
    no_node = PhysicalCoordinates(2.0, 0.0, 0.0, 0.0)
    no_linker = PhysicalCoordinates(2.0, 1.0, 0.0, 0.0)
    assert compute_propensities(counts, no_node).add_node == 0.0
    assert compute_propensities(counts, no_linker).add_linker == 0.0


def test_ideal_cluster_conversion_contains_factor_six() -> None:
    assert ideal_node_fraction(1.0) == pytest.approx(1.0 / 7.0)
    assert ideal_node_fraction(6.0) == pytest.approx(0.5)


def test_constant_closure_has_no_entity_number_input() -> None:
    coordinate = PhysicalCoordinates(1.0, 0.5, 1.0, 0.5)
    expected = math.exp(30.9 * 0.5)
    assert coordinate.closure_weight == pytest.approx(expected)


def test_explicit_and_normalized_propensities_are_equivalent() -> None:
    counts = OpportunityCounts(5, 13, 4, 8)
    physical = PhysicalCoordinates(3.0, 0.2, 0.9, 0.1)
    p1 = compute_propensities(counts, physical)
    p2 = propensities_from_weights(
        counts,
        node_activity=physical.node_fraction,
        linker_activity=1.0 - physical.node_fraction,
        closure_weight=physical.normalized_closure_ratio,
        off_weight=physical.normalized_off_ratio,
    )
    assert p1.add_node / physical.external_activity == pytest.approx(p2.add_node)
    assert p1.add_linker / physical.external_activity == pytest.approx(p2.add_linker)
    assert p1.close_internal / physical.external_activity == pytest.approx(p2.close_internal)
    assert p1.remove_bond / physical.external_activity == pytest.approx(p2.remove_bond)
    n1 = p1.normalized()
    n2 = p2.normalized()
    assert n1.add_node == pytest.approx(n2.add_node)
    assert n1.add_linker == pytest.approx(n2.add_linker)
    assert n1.close_internal == pytest.approx(n2.close_internal)
    assert n1.remove_bond == pytest.approx(n2.remove_bond)


def test_gillespie_mean_for_one_constant_state() -> None:
    # A small stochastic regression catches accidental use of an event count as
    # a physical clock without making the test suite expensive.
    from common_kmc_selector import gillespie_waiting_time

    rng = random.Random(20260830)
    samples = [gillespie_waiting_time(4.0, rng) for _ in range(20_000)]
    assert sum(samples) / len(samples) == pytest.approx(0.25, rel=0.03)
