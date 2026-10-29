#!/usr/bin/env python3
"""Rare-event UiO-like growth from a fixed crystallographic MOF-808 host.

The host is a boundary condition, not a second kinetic phase: seed-seed bonds
are excluded from the removal pool, whereas host-daughter and daughter-
daughter bonds use the same reversible event law as the de novo UiO-66 KMC.
Detached daughter fragments are deleted, preserving the model's finite-bath
reference convention without permitting the substrate itself to disappear.

The runner records complete daughter-growth excursions.  A "surface
nucleation event" is declared only when one connected daughter component
reaches a stated entity threshold and internal cycle-rank threshold.  Both
thresholds are stored explicitly so their sensitivity can be analysed rather
than hidden in a qualitative label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import random
import time
from typing import Any

import numpy as np

from common_kmc_selector import (
    EventPropensities,
    OpportunityCounts,
    PhysicalCoordinates,
    SelectedEvent,
    chemistry_record,
    compute_propensities,
    gillespie_waiting_time,
)
from common_molecular_runner import _load_engine, assembly_state, sha256


def is_host(entity: Any) -> bool:
    return bool(getattr(entity, "is_fixed_host", False))


def removable_bonds(assembly: Any) -> list[Any]:
    """All bonds except fixed host-host framework bonds."""

    return [
        pair
        for pair in assembly.linked_carboxylate_pairs.items
        if not (
            is_host(pair[0].belonging_entity)
            and is_host(pair[1].belonging_entity)
        )
    ]


def surface_opportunity_counts(assembly: Any) -> OpportunityCounts:
    return OpportunityCounts(
        linker_free_sites=len(assembly.Linker_free_carboxylates),
        metal_free_sites=len(assembly.MC_free_carboxylates),
        ready_internal_pairs=len(assembly.ready_to_connect_carboxylate_pairs),
        existing_bonds=len(removable_bonds(assembly)),
    )


def select_surface_event(
    assembly: Any, coordinates: PhysicalCoordinates, rng: random.Random
) -> SelectedEvent:
    counts = surface_opportunity_counts(assembly)
    propensities = compute_propensities(counts, coordinates)
    if propensities.total <= 0.0:
        raise RuntimeError("no enabled event in the current surface state")
    draw = rng.random() * propensities.total
    if draw < propensities.add_node:
        return SelectedEvent(
            "add_node",
            rng.choice(assembly.Linker_free_carboxylates.items),
            None,
            propensities,
        )
    draw -= propensities.add_node
    if draw < propensities.add_linker:
        return SelectedEvent(
            "add_linker",
            rng.choice(assembly.MC_free_carboxylates.items),
            None,
            propensities,
        )
    draw -= propensities.add_linker
    if draw < propensities.close_internal:
        return SelectedEvent(
            "close_internal",
            None,
            rng.choice(assembly.ready_to_connect_carboxylate_pairs.items),
            propensities,
        )
    bonds = removable_bonds(assembly)
    if not bonds:
        raise RuntimeError("removal selected with an empty removable-bond pool")
    return SelectedEvent("remove_bond", None, rng.choice(bonds), propensities)


def _add_free_site(assembly: Any, carboxylate: Any) -> None:
    if carboxylate.belonging_entity is None:
        return
    if carboxylate not in assembly.free_carboxylates:
        assembly.free_carboxylates.add(carboxylate)
    if carboxylate.belonging_entity.entity_type == "Zr":
        if carboxylate not in assembly.MC_free_carboxylates:
            assembly.MC_free_carboxylates.add(carboxylate)
    elif carboxylate not in assembly.Linker_free_carboxylates:
        assembly.Linker_free_carboxylates.add(carboxylate)


def _remove_from_pool(pool: Any, value: Any) -> None:
    if value in pool:
        pool.remove(value)


def _linked_pair_in_pool(pool: Any, pair: Any) -> Any | None:
    if pair in pool:
        return pair
    reverse = (pair[1], pair[0])
    if reverse in pool:
        return reverse
    return None


def _component(root: Any, active: set[Any]) -> set[Any]:
    visited = {root}
    stack = [root]
    while stack:
        entity = stack.pop()
        for neighbor in getattr(entity, "connected_entities", ()):
            if neighbor in active and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def _delete_entities(assembly: Any, entities_to_remove: set[Any]) -> None:
    """Delete a detached daughter component and release sites on the host side."""

    ordered_remove = [entity for entity in assembly.entities if entity in entities_to_remove]
    if any(is_host(entity) for entity in ordered_remove):
        raise RuntimeError("the fixed host cannot be deleted")
    for entity in ordered_remove:
        for carboxylate in list(entity.carboxylates):
            _remove_from_pool(assembly.free_carboxylates, carboxylate)
            _remove_from_pool(assembly.MC_free_carboxylates, carboxylate)
            _remove_from_pool(assembly.Linker_free_carboxylates, carboxylate)
            pair = assembly.pair_index.get(carboxylate)
            if pair is None:
                continue
            stored = _linked_pair_in_pool(assembly.ready_to_connect_carboxylate_pairs, pair)
            if stored is not None:
                assembly.ready_to_connect_carboxylate_pairs.remove(stored)
            stored = _linked_pair_in_pool(assembly.linked_carboxylate_pairs, pair)
            if stored is not None:
                assembly.linked_carboxylate_pairs.remove(stored)
            other = pair[0] if carboxylate is pair[1] else pair[1]
            assembly.pair_index.pop(carboxylate, None)
            assembly.pair_index.pop(other, None)
            if (
                other.belonging_entity is not None
                and other.belonging_entity not in entities_to_remove
            ):
                _add_free_site(assembly, other)

    for entity in ordered_remove:
        for neighbor in list(getattr(entity, "connected_entities", ())):
            if neighbor not in entities_to_remove and entity in neighbor.connected_entities:
                neighbor.connected_entities.remove(entity)
        entity.connected_entities = None
        entity.kdtree = None
        cube_index = tuple(np.floor(entity.center / assembly.ROI_range).astype(int))
        if cube_index in assembly.node_mapping and entity in assembly.node_mapping[cube_index]:
            assembly.node_mapping[cube_index].remove(entity)
            if not assembly.node_mapping[cube_index]:
                del assembly.node_mapping[cube_index]
        assembly.entities.remove(entity)
        for carboxylate in entity.carboxylates:
            carboxylate.belonging_entity = None


def remove_surface_bond(assembly: Any, selected_pair: Any) -> dict[str, Any]:
    """Open one reversible bond while retaining the fixed host component."""

    first_carboxylate, second_carboxylate = selected_pair
    first_entity = first_carboxylate.belonging_entity
    second_entity = second_carboxylate.belonging_entity
    if is_host(first_entity) and is_host(second_entity):
        raise RuntimeError("host-host bonds are not removable in the fixed-host assay")
    stored = _linked_pair_in_pool(assembly.linked_carboxylate_pairs, selected_pair)
    if stored is None:
        raise RuntimeError("selected removable bond is not linked")
    assembly.linked_carboxylate_pairs.remove(stored)
    assembly.pair_index.pop(first_carboxylate, None)
    assembly.pair_index.pop(second_carboxylate, None)
    first_entity.connected_entities.remove(second_entity)
    second_entity.connected_entities.remove(first_entity)

    active = set(assembly.entities.items)
    first_component = _component(first_entity, active)
    if second_entity in first_component:
        # Opening a cycle does not detach material.  The geometrically adjacent
        # pair remains available for the ordinary reversible closure event.
        pair = (first_carboxylate, second_carboxylate)
        assembly.ready_to_connect_carboxylate_pairs.add(pair)
        assembly.pair_index[first_carboxylate] = pair
        assembly.pair_index[second_carboxylate] = pair
        return {"opened_cycle": True, "deleted_entities": 0}

    second_component = active - first_component
    first_has_host = any(is_host(entity) for entity in first_component)
    second_has_host = any(is_host(entity) for entity in second_component)
    if first_has_host and second_has_host:
        raise RuntimeError("a connected fixed host cannot occupy both detached components")
    if first_has_host:
        to_delete = second_component
        retained_carboxylate = first_carboxylate
    elif second_has_host:
        to_delete = first_component
        retained_carboxylate = second_carboxylate
    else:
        # This fallback preserves the archived smaller-fragment convention for
        # any mobile-only component encountered in a branched restart.
        if len(first_component) <= len(second_component):
            to_delete = first_component
            retained_carboxylate = second_carboxylate
        else:
            to_delete = second_component
            retained_carboxylate = first_carboxylate
    deleted_count = len(to_delete)
    _add_free_site(assembly, retained_carboxylate)
    _delete_entities(assembly, set(to_delete))
    return {"opened_cycle": False, "deleted_entities": deleted_count}


def daughter_components(assembly: Any) -> list[dict[str, Any]]:
    active = set(assembly.entities.items)
    mobile = {entity for entity in active if not is_host(entity)}
    components: list[dict[str, Any]] = []
    unseen = set(mobile)
    linked_pairs = list(assembly.linked_carboxylate_pairs.items)
    while unseen:
        root = next(entity for entity in assembly.entities if entity in unseen)
        visited = {root}
        stack = [root]
        unseen.remove(root)
        while stack:
            entity = stack.pop()
            for neighbor in getattr(entity, "connected_entities", ()):
                if neighbor in unseen and not is_host(neighbor):
                    unseen.remove(neighbor)
                    visited.add(neighbor)
                    stack.append(neighbor)
        internal_bonds = 0
        host_bonds = 0
        attached_host_entities: set[Any] = set()
        for pair in linked_pairs:
            left = pair[0].belonging_entity
            right = pair[1].belonging_entity
            if left in visited and right in visited:
                internal_bonds += 1
            elif left in visited and is_host(right):
                host_bonds += 1
                attached_host_entities.add(right)
            elif right in visited and is_host(left):
                host_bonds += 1
                attached_host_entities.add(left)
        type_counts = Counter(entity.entity_type for entity in visited)
        components.append(
            {
                "entities": len(visited),
                "zr_entities": int(type_counts.get("Zr", 0)),
                "linker_entities": int(type_counts.get("Ligand", 0)),
                "internal_bonds": internal_bonds,
                "internal_cycle_rank": max(0, internal_bonds - len(visited) + 1),
                "host_attachment_bonds": host_bonds,
                "attached_host_entities": len(attached_host_entities),
            }
        )
    return sorted(
        components,
        key=lambda record: (
            record["entities"],
            record["internal_cycle_rank"],
            record["internal_bonds"],
        ),
        reverse=True,
    )


def daughter_state(assembly: Any) -> dict[str, Any]:
    host_entities = [entity for entity in assembly.entities if is_host(entity)]
    components = daughter_components(assembly)
    mobile_entities = sum(record["entities"] for record in components)
    largest = components[0] if components else None
    return {
        "host_entities": len(host_entities),
        "mobile_entities": mobile_entities,
        "mobile_components": len(components),
        "largest_component": largest,
        "components": components,
        "removable_bonds": len(removable_bonds(assembly)),
    }


def _criterion(state: dict[str, Any], target_entities: int, minimum_cycle_rank: int) -> bool:
    largest = state["largest_component"]
    return bool(
        largest
        and largest["entities"] >= target_entities
        and largest["internal_cycle_rank"] >= minimum_cycle_rank
    )


def run_surface_trace(
    *,
    host_pickle: Path,
    coordinates: PhysicalCoordinates,
    random_seed: int,
    target_mobile_entities: int = 30,
    minimum_mobile_cycle_rank: int = 1,
    milestones: tuple[int, ...] = (1, 2, 4, 8, 16, 30),
    event_cap: int = 200_000,
    wall_time_cap_s: float = 90.0,
    maximum_launches: int = 10_000,
    mol2_output_path: Path | None = None,
) -> dict[str, Any]:
    coordinates.validate()
    if target_mobile_entities < 1 or minimum_mobile_cycle_rank < 0:
        raise ValueError("invalid surface-nucleation thresholds")
    engine, engine_path, redirects = _load_engine("uio66")
    engine.MAX_DEPTH = target_mobile_entities + 100
    with host_pickle.open("rb") as handle:
        assembly = pickle.load(handle)
    # The spatial tree is a machine-specific cache, not molecular state.
    # Rebuild it with the active geometry runtime after deserialization.
    for entity in assembly.entities.items:
        if hasattr(entity, "kdtree"):
            entity.kdtree = None
    host_entities = {entity for entity in assembly.entities if is_host(entity)}
    if not host_entities or len(host_entities) != len(assembly.entities):
        raise ValueError("surface runner requires an all-host crystallographic seed")
    if any(
        not (is_host(pair[0].belonging_entity) and is_host(pair[1].belonging_entity))
        for pair in assembly.linked_carboxylate_pairs
    ):
        raise ValueError("initial seed contains a non-host bond")

    random.seed(random_seed)
    np.random.seed(random_seed % (2**32 - 1))
    rng = random.Random(random_seed ^ 0x8085FACE)
    started = time.perf_counter()
    model_time = 0.0
    event_counts: Counter[str] = Counter()
    excursions: list[dict[str, Any]] = []
    current_excursion: dict[str, Any] | None = None
    milestone_values = tuple(sorted({int(value) for value in milestones if value > 0}))
    milestone_launch_counts = {str(value): 0 for value in milestone_values}
    initial_assembly = assembly_state(assembly)
    termination = "event_cap"
    executed_events = 0

    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        if len(excursions) >= maximum_launches:
            termination = "maximum_launches"
            break
        before = daughter_state(assembly)
        try:
            selection = select_surface_event(assembly, coordinates, rng)
        except RuntimeError as exc:
            if "no enabled event" not in str(exc):
                raise
            termination = "no_enabled_event"
            break
        model_time += gillespie_waiting_time(selection.propensities.total, rng)
        if selection.event in {"add_node", "add_linker"}:
            event_counts[f"{selection.event}_attempt"] += 1
            before_entities = len(assembly.entities)
            assembly.grow_one_step(selection.selected_site)
            if len(assembly.entities) > before_entities:
                event_counts[f"{selection.event}_accepted"] += 1
            else:
                event_counts[f"{selection.event}_collision_rejected"] += 1
        elif selection.event == "close_internal":
            assembly.link_internal_carboxylate(selection.selected_pair)
            event_counts["close_internal"] += 1
        elif selection.event == "remove_bond":
            removal = remove_surface_bond(assembly, selection.selected_pair)
            event_counts["remove_bond"] += 1
            event_counts["remove_bond_deleted_entities"] += int(removal["deleted_entities"])
            if removal["opened_cycle"]:
                event_counts["remove_bond_opened_cycle"] += 1

        executed_events = event_index
        after = daughter_state(assembly)
        if before["mobile_entities"] == 0 and after["mobile_entities"] > 0:
            current_excursion = {
                "launch_index": len(excursions) + 1,
                "start_event": event_index,
                "start_model_time": model_time,
                "launch_site_role": getattr(selection.selected_site, "host_site_role", None),
                "launch_site_status": getattr(selection.selected_site, "host_site_status", None),
                "maximum_mobile_entities": after["mobile_entities"],
                "maximum_largest_component_entities": after["largest_component"]["entities"],
                "maximum_largest_component_cycle_rank": after["largest_component"]["internal_cycle_rank"],
                "first_passage": {},
            }
        if current_excursion is not None and after["largest_component"] is not None:
            current_excursion["maximum_mobile_entities"] = max(
                current_excursion["maximum_mobile_entities"], after["mobile_entities"]
            )
            current_excursion["maximum_largest_component_entities"] = max(
                current_excursion["maximum_largest_component_entities"],
                after["largest_component"]["entities"],
            )
            current_excursion["maximum_largest_component_cycle_rank"] = max(
                current_excursion["maximum_largest_component_cycle_rank"],
                after["largest_component"]["internal_cycle_rank"],
            )
            for milestone in milestone_values:
                key = str(milestone)
                if (
                    key not in current_excursion["first_passage"]
                    and after["largest_component"]["entities"] >= milestone
                ):
                    current_excursion["first_passage"][key] = {
                        "event": event_index,
                        "model_time": model_time,
                        "cycle_rank": after["largest_component"]["internal_cycle_rank"],
                    }
                    milestone_launch_counts[key] += 1
        if _criterion(after, target_mobile_entities, minimum_mobile_cycle_rank):
            assert current_excursion is not None
            current_excursion.update(
                {
                    "outcome": "surface_nucleation_event",
                    "end_event": event_index,
                    "end_model_time": model_time,
                    "terminal_state": after,
                }
            )
            excursions.append(current_excursion)
            current_excursion = None
            termination = "surface_nucleation_event"
            break
        if before["mobile_entities"] > 0 and after["mobile_entities"] == 0:
            assert current_excursion is not None
            current_excursion.update(
                {
                    "outcome": "collapsed_to_host",
                    "end_event": event_index,
                    "end_model_time": model_time,
                    "terminal_state": after,
                }
            )
            excursions.append(current_excursion)
            current_excursion = None

    final_daughter = daughter_state(assembly)
    if current_excursion is not None:
        current_excursion.update(
            {
                "outcome": "censored",
                "end_event": executed_events,
                "end_model_time": model_time,
                "terminal_state": final_daughter,
            }
        )
        excursions.append(current_excursion)
    successful = sum(record["outcome"] == "surface_nucleation_event" for record in excursions)
    completed_launches = sum(record["outcome"] != "censored" for record in excursions)
    mol2_record = None
    if mol2_output_path is not None:
        mol2_output_path.parent.mkdir(parents=True, exist_ok=True)
        assembly.get_mol2_file(str(mol2_output_path))
        mol2_record = {
            "path": mol2_output_path.resolve().as_posix(),
            "sha256": sha256(mol2_output_path),
        }
    return {
        "schema_version": "fixed-host-surface-secondary-nucleation-v1",
        "host": {
            "pickle": host_pickle.resolve().as_posix(),
            "pickle_sha256": sha256(host_pickle),
            "initial_assembly": initial_assembly,
            "host_entities": len(host_entities),
            "host_host_bonds_frozen": len(assembly.linked_carboxylate_pairs)
            - len(removable_bonds(assembly)),
        },
        "coordinates": chemistry_record(coordinates),
        "nucleation_definition": {
            "largest_connected_daughter_component_entities": target_mobile_entities,
            "minimum_internal_cycle_rank": minimum_mobile_cycle_rank,
            "host_daughter_and_daughter_daughter_bonds_reversible": True,
        },
        "clock": {
            "model_time": model_time,
            "reference": "one dimensionless unit equals one declared site-opening time",
        },
        "termination": {
            "status": termination,
            "event_cap": event_cap,
            "executed_events": executed_events,
            "wall_time_cap_s": wall_time_cap_s,
            "maximum_launches": maximum_launches,
        },
        "event_counts": dict(event_counts),
        "launch_summary": {
            "recorded_launches": len(excursions),
            "completed_launches": completed_launches,
            "successful_launches": successful,
            "success_probability_per_completed_launch": (
                successful / completed_launches if completed_launches else None
            ),
            "surface_nucleation_events_per_model_time": (
                successful / model_time if model_time > 0.0 else None
            ),
            "launches_reaching_size_milestones": milestone_launch_counts,
        },
        "excursions": excursions,
        "final_daughter_state": final_daughter,
        "provenance": {
            "engine_path": engine_path.resolve().as_posix(),
            "engine_sha256": sha256(engine_path),
            "engine_import_redirects": redirects,
            "runner_path": Path(__file__).resolve().as_posix(),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "host_boundary": (
                "seed-seed bonds excluded from removal; all other bonds use the common reversible weights"
            ),
            "fragment_rule": (
                "delete the detached non-host component; mobile-only fallback deletes the smaller component"
            ),
            "mol2_output": mol2_record,
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--external-activity", type=float, default=1.0)
    parser.add_argument("--node-fraction", type=float, required=True)
    parser.add_argument("--off-weight", type=float, required=True)
    parser.add_argument("--sc", type=float, required=True)
    parser.add_argument("--target-mobile", type=int, default=30)
    parser.add_argument("--minimum-cycle-rank", type=int, default=1)
    parser.add_argument("--event-cap", type=int, default=200_000)
    parser.add_argument("--wall-time-cap-s", type=float, default=90.0)
    parser.add_argument("--maximum-launches", type=int, default=10_000)
    parser.add_argument("--mol2-output", type=Path)
    args = parser.parse_args()
    coordinates = PhysicalCoordinates(
        external_activity=args.external_activity,
        node_fraction=args.node_fraction,
        off_weight=args.off_weight,
        closure_entropy_coefficient=args.sc,
    )
    result = run_surface_trace(
        host_pickle=args.host_pickle,
        coordinates=coordinates,
        random_seed=args.seed,
        target_mobile_entities=args.target_mobile,
        minimum_mobile_cycle_rank=args.minimum_cycle_rank,
        event_cap=args.event_cap,
        wall_time_cap_s=args.wall_time_cap_s,
        maximum_launches=args.maximum_launches,
        mol2_output_path=args.mol2_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output.resolve().as_posix(), **result["launch_summary"], "termination": result["termination"]}, indent=2))


if __name__ == "__main__":
    main()
