"""Planar-site utilities for BTB metal-organic-layer KMC.

The archived BTB geometry engine exposes every carboxylate direction of an
M6 node.  Unfiltered use therefore samples a three-dimensional BTB network,
not a single metal-organic layer.  This module supplies a declared layer
plane, identifies the six coplanar M6 directions and the three coplanar BTB
directions, places incoming entities in that plane, and selects events only
from those molecular opportunities.

The out-of-plane M6 carboxylates remain present in the atomistic object as
formate-capped sites.  They are excluded from ordinary layer propagation but
can later be exposed explicitly in a separate secondary-nucleation operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

import numpy as np

from common_kmc_selector import (
    bond_removal_pool,
    choose_weighted,
    EventPropensities,
    OpportunityCounts,
    PhysicalCoordinates,
    SelectedEvent,
    propensities_from_weights,
)


@dataclass(frozen=True)
class LayerPlane:
    origin: np.ndarray
    normal: np.ndarray
    center_tolerance: float = 1.0
    direction_cosine_tolerance: float = 0.08

    def validate(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if origin.shape != (3,) or normal.shape != (3,):
            raise ValueError("origin and normal must be three-vectors")
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("plane normal must have finite nonzero length")
        if self.center_tolerance < 0.0:
            raise ValueError("center_tolerance must be non-negative")
        if not 0.0 <= self.direction_cosine_tolerance < 1.0:
            raise ValueError("direction cosine tolerance must lie in [0, 1)")

    @property
    def unit_normal(self) -> np.ndarray:
        self.validate()
        normal = np.asarray(self.normal, dtype=float)
        return normal / np.linalg.norm(normal)


def carboxylate_center(carboxylate: Any) -> np.ndarray:
    entity = carboxylate.belonging_entity
    return np.mean(entity.coordinates[carboxylate.atom_indexes], axis=0)


def signed_plane_distance(point: np.ndarray, plane: LayerPlane) -> float:
    return float(np.dot(np.asarray(point, dtype=float) - plane.origin, plane.unit_normal))


def infer_six_connected_m6_plane(node: Any) -> LayerPlane:
    """Infer a symmetry plane containing six of the twelve M6 directions."""

    if len(node.carboxylates) != 12:
        raise ValueError("six-connected layer inference requires a 12-site M6 node")
    center = np.asarray(node.center, dtype=float)
    vectors = np.asarray([carboxylate_center(c) - center for c in node.carboxylates])
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    candidates: list[np.ndarray] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            normal = np.cross(vectors[i], vectors[j])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                candidates.append(normal / norm)
    if not candidates:
        raise ValueError("could not infer a layer plane from node directions")

    def score(normal: np.ndarray) -> tuple[int, float, tuple[float, float, float]]:
        projections = np.abs(vectors @ normal)
        count = int(np.sum(projections < 0.08))
        residual = float(np.sum(np.sort(projections)[:6] ** 2))
        canonical = normal.copy()
        first = int(np.argmax(np.abs(canonical) > 1e-10))
        if canonical[first] < 0.0:
            canonical *= -1.0
        return count, -residual, tuple(float(x) for x in canonical)

    best = max(candidates, key=score)
    if score(best)[0] != 6:
        raise ValueError(
            f"expected six coplanar M6 directions; found {score(best)[0]}"
        )
    first = int(np.argmax(np.abs(best) > 1e-10))
    if best[first] < 0.0:
        best = -best
    return LayerPlane(origin=center.copy(), normal=best)


def attach_layer_plane(assembly: Any, plane: LayerPlane) -> None:
    """Persist a minimal, pickle-safe layer-plane record on an assembly."""

    assembly._layer_plane_origin = np.asarray(plane.origin, dtype=float).copy()
    assembly._layer_plane_normal = plane.unit_normal.copy()


def layer_plane_from_assembly(assembly: Any) -> LayerPlane:
    """Recover the declared plane, or infer it from an M6 node."""

    if hasattr(assembly, "_layer_plane_origin") and hasattr(
        assembly, "_layer_plane_normal"
    ):
        return LayerPlane(
            origin=np.asarray(assembly._layer_plane_origin, dtype=float),
            normal=np.asarray(assembly._layer_plane_normal, dtype=float),
        )
    nodes = [
        entity
        for entity in assembly.entities.items
        if getattr(entity, "entity_type", None) == "Zr"
        and len(getattr(entity, "carboxylates", ())) == 12
    ]
    if not nodes:
        raise ValueError("no 12-site M6 node is available to define the layer plane")
    plane = infer_six_connected_m6_plane(nodes[0])
    attach_layer_plane(assembly, plane)
    return plane


def entity_center_is_in_plane(entity: Any, plane: LayerPlane) -> bool:
    return abs(signed_plane_distance(entity.center, plane)) <= plane.center_tolerance


def carboxylate_direction_is_in_plane(carboxylate: Any, plane: LayerPlane) -> bool:
    entity = carboxylate.belonging_entity
    vector = carboxylate_center(carboxylate) - np.asarray(entity.center, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return False
    return abs(float(np.dot(vector / norm, plane.unit_normal))) <= plane.direction_cosine_tolerance


def active_carboxylate(carboxylate: Any, plane: LayerPlane) -> bool:
    entity = carboxylate.belonging_entity
    return entity_center_is_in_plane(entity, plane) and carboxylate_direction_is_in_plane(
        carboxylate, plane
    )


def active_pool(pool: Any, plane: LayerPlane) -> list[Any]:
    return [site for site in pool.items if active_carboxylate(site, plane)]


def active_internal_pairs(assembly: Any, plane: LayerPlane) -> list[Any]:
    return [
        pair
        for pair in assembly.ready_to_connect_carboxylate_pairs.items
        if active_carboxylate(pair[0], plane) and active_carboxylate(pair[1], plane)
    ]


def planar_opportunities(
    assembly: Any,
    plane: LayerPlane,
    bond_accessibility_mode: str = "all_bonds",
) -> tuple[OpportunityCounts, dict[str, list[Any]]]:
    linker_sites = active_pool(assembly.Linker_free_carboxylates, plane)
    metal_sites = active_pool(assembly.MC_free_carboxylates, plane)
    internal_pairs = active_internal_pairs(assembly, plane)
    bonds, bond_weights = bond_removal_pool(
        assembly,
        bond_accessibility_mode,
        active_site_predicate=lambda site: active_carboxylate(site, plane),
    )
    counts = OpportunityCounts(
        linker_free_sites=len(linker_sites),
        metal_free_sites=len(metal_sites),
        ready_internal_pairs=len(internal_pairs),
        existing_bonds=len(bonds),
        removal_opportunity_weight=sum(bond_weights),
    )
    return counts, {
        "linker_sites": linker_sites,
        "metal_sites": metal_sites,
        "internal_pairs": internal_pairs,
        "bonds": bonds,
        "bond_weights": bond_weights,
    }


def select_planar_event(
    assembly: Any,
    coordinates: PhysicalCoordinates,
    plane: LayerPlane,
    rng: random.Random,
    bond_accessibility_mode: str = "all_bonds",
) -> SelectedEvent:
    counts, pools = planar_opportunities(assembly, plane, bond_accessibility_mode)
    prop = propensities_from_weights(
        counts,
        node_activity=coordinates.node_activity,
        linker_activity=coordinates.linker_activity,
        closure_weight=coordinates.closure_weight,
        off_weight=coordinates.off_weight,
    )
    if prop.total <= 0.0:
        raise RuntimeError("no enabled planar event in the current molecular state")
    draw = rng.random() * prop.total
    if draw < prop.add_node:
        return SelectedEvent("add_node", rng.choice(pools["linker_sites"]), None, prop)
    draw -= prop.add_node
    if draw < prop.add_linker:
        return SelectedEvent("add_linker", rng.choice(pools["metal_sites"]), None, prop)
    draw -= prop.add_linker
    if draw < prop.close_internal:
        return SelectedEvent("close_internal", None, rng.choice(pools["internal_pairs"]), prop)
    return SelectedEvent(
        "remove_bond",
        None,
        choose_weighted(pools["bonds"], pools["bond_weights"], rng),
        prop,
    )


def _placement_score(entity: Any, plane: LayerPlane, required_active_sites: int) -> tuple[float, float]:
    center_distance = abs(signed_plane_distance(entity.center, plane))
    active_count = sum(carboxylate_direction_is_in_plane(c, plane) for c in entity.carboxylates)
    direction_penalty = float(abs(required_active_sites - active_count))
    return direction_penalty, center_distance


def _aligned_candidates(
    entity_classes: Iterable[type],
    reference_carboxylate: Any,
    plane: LayerPlane,
    required_active_sites: int,
) -> list[tuple[tuple[float, float], Any, Any]]:
    candidates: list[tuple[tuple[float, float], Any, Any]] = []
    for entity_class in entity_classes:
        template = entity_class()
        site_count = len(template.carboxylates)
        for site_index in range(site_count):
            entity = entity_class()
            chosen = entity.carboxylates[site_index]
            transform = chosen.calculate_rotation_translation_matrix(reference_carboxylate)
            entity.apply_transformation(transform)
            candidates.append(
                (_placement_score(entity, plane, required_active_sites), entity, chosen)
            )
    return candidates


def grow_one_step_planar(
    assembly: Any,
    selected_carboxylate: Any,
    engine: Any,
    plane: LayerPlane,
    *,
    m6_node_fraction: float = 1.0,
    rng: random.Random,
) -> bool:
    """Place and add the best molecular entity consistent with the layer plane."""

    if selected_carboxylate.carboxylate_type == "benzoate":
        if m6_node_fraction >= 1.0:
            classes = [engine.Zr6]
        elif m6_node_fraction <= 0.0:
            classes = [engine.Zr12]
        else:
            classes = [engine.Zr6] if rng.random() < m6_node_fraction else [engine.Zr12]
        required_sites = 6
    else:
        classes = [engine.BTBa, engine.BTBb]
        required_sites = 3

    candidates = _aligned_candidates(
        classes, selected_carboxylate, plane, required_sites
    )
    best_score = min(score for score, _, _ in candidates)
    ties = [(entity, chosen) for score, entity, chosen in candidates if score == best_score]
    new_entity, chosen_site = rng.choice(ties)
    if best_score[0] != 0.0 or best_score[1] > plane.center_tolerance:
        return False

    is_bumping, additional_pairs, additional_new_sites = assembly.find_potential_connection(
        new_entity, selected_carboxylate
    )
    if is_bumping:
        for carboxylate in new_entity.carboxylates:
            carboxylate.belonging_entity = None
        return False
    assembly.add_entity(
        selected_carboxylate,
        new_entity,
        chosen_site,
        additional_pairs,
        additional_new_sites,
    )
    return True


def planarity_metrics(assembly: Any, plane: LayerPlane) -> dict[str, float]:
    centers = np.asarray([entity.center for entity in assembly.entities.items], dtype=float)
    distances = (centers - plane.origin) @ plane.unit_normal
    centered = centers - np.mean(centers, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    return {
        "rms_thickness": float(np.sqrt(np.mean(distances**2))),
        "maximum_absolute_thickness": float(np.max(np.abs(distances))),
        "pca_out_of_plane_fraction": float(
            variance[-1] / np.sum(variance) if np.sum(variance) > 0.0 else 0.0
        ),
    }
