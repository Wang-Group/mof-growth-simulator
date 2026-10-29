#!/usr/bin/env python3
"""Pilot an explicit {111}-facet UiO-66 surface-growth assay.

The n=4 octahedron is the protected parent crystal.  The concentric n=6
octahedron supplies the ideal next-shell registry positions.  Every accepted
post-seed entity is classified by its distance to the nearest ideal position
of the same entity type.  This determines whether the archived geometry engine
can generate an off-registry surface branch before a daughter-domain committor
calculation is attempted.

This script is diagnostic.  It does not call a branch a secondary nucleus and
does not use an arbitrary component-size threshold as a reported rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_DIR = (
    HERE.parents[2]
    / "analysis_workspace"
    / "source_package"
    / "publication_handoff_2026-08-05"
    / "github_release"
    / "source_snapshot"
    / "UiO-66"
    / "UiO_66_from_amorphous"
    / "initialize_structures1"
)
sys.path.insert(0, str(SOURCE_DIR))
import UiO66_Assembly_Large_Correction_conc as engine  # noqa: E402


def load_assembly(path: Path):
    with path.open("rb") as handle:
        assembly = pickle.load(handle)
    engine.rebuild_references_after_load(assembly, verbose=False)
    assembly.ZR6_PERCENTAGE = 1.0
    assembly.ENTROPY_GAIN = 0.0
    assembly.BUMPING_THRESHOLD = 2.0
    return assembly


def centers_by_type(assembly) -> dict[str, np.ndarray]:
    answer: dict[str, list[np.ndarray]] = {}
    for entity in assembly.entities:
        answer.setdefault(str(entity.entity_type), []).append(np.asarray(entity.center, dtype=float))
    return {key: np.asarray(value, dtype=float) for key, value in answer.items()}


def build_registry(parent, outer, match_tolerance: float) -> tuple[dict[str, np.ndarray], dict]:
    parent_points = centers_by_type(parent)
    outer_points = centers_by_type(outer)
    # The even-sized octahedra are related by an integer-cell translation.  A
    # linker-centroid alignment is exact to <1e-3 A for the generated seeds.
    shift = outer_points["Ligand"].mean(axis=0) - parent_points["Ligand"].mean(axis=0)
    registry: dict[str, np.ndarray] = {}
    diagnostics = {"translation_outer_to_parent_A": (-shift).tolist(), "types": {}}
    for entity_type, outer_array in outer_points.items():
        transformed = outer_array - shift
        parent_array = parent_points[entity_type]
        distances = np.sqrt(
            ((transformed[:, None, :] - parent_array[None, :, :]) ** 2).sum(axis=2)
        )
        nearest = distances.min(axis=1)
        registry[entity_type] = transformed[nearest > match_tolerance]
        diagnostics["types"][entity_type] = {
            "parent_sites": int(len(parent_array)),
            "outer_sites": int(len(outer_array)),
            "matched_parent_shell": int(np.sum(nearest <= match_tolerance)),
            "ghost_registry_sites": int(np.sum(nearest > match_tolerance)),
            "maximum_matched_distance_A": float(nearest[nearest <= match_tolerance].max()),
        }
    return registry, diagnostics


def nearest_registry_distance(entity, registry: dict[str, np.ndarray]) -> float:
    points = registry[str(entity.entity_type)]
    if len(points) == 0:
        return math.inf
    delta = points - np.asarray(entity.center, dtype=float)
    return float(np.sqrt(np.sum(delta * delta, axis=1)).min())


def removable_pairs(assembly, parent_entities: set) -> list:
    return [
        pair
        for pair in assembly.linked_carboxylate_pairs
        if not (
            pair[0].belonging_entity in parent_entities
            and pair[1].belonging_entity in parent_entities
        )
    ]


def direct_step(assembly, parent_entities: set, r_sim: float, p_zr: float, e_sim: float):
    removable = removable_pairs(assembly, parent_entities)
    weights = np.asarray(
        [
            p_zr * len(assembly.Linker_free_carboxylates),
            (1.0 - p_zr) * len(assembly.MC_free_carboxylates),
            e_sim * len(assembly.ready_to_connect_carboxylate_pairs),
            r_sim * len(removable),
        ],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0, "stalled", [], None
    before = set(assembly.entities)
    selected_site = None
    draw = random.random() * total
    cut = np.cumsum(weights)
    if draw < cut[0]:
        selected_site = assembly.Linker_free_carboxylates.get_random()
        assembly.grow_one_step(selected_site)
        event = "add_Zr"
    elif draw < cut[1]:
        selected_site = assembly.MC_free_carboxylates.get_random()
        assembly.grow_one_step(selected_site)
        event = "add_BDC"
    elif draw < cut[2]:
        assembly.link_internal_carboxylate(assembly.ready_to_connect_carboxylate_pairs.get_random())
        event = "close"
    else:
        assembly.remove_linkage(random.choice(removable))
        event = "remove"
    new_entities = list(set(assembly.entities) - before)
    dt = -math.log(max(random.random(), 1e-15)) / total
    return dt, event, new_entities, selected_site


def new_components(assembly, parent_entities: set, registry_status: dict) -> list[dict]:
    new_entities = set(assembly.entities) - parent_entities
    seen = set()
    rows = []
    for start in new_entities:
        if start in seen:
            continue
        queue = deque([start])
        component = {start}
        seen.add(start)
        while queue:
            entity = queue.popleft()
            for neighbor in entity.connected_entities:
                if neighbor in new_entities and neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        edges = sum(
            1 for entity in component for neighbor in entity.connected_entities if neighbor in component
        ) // 2
        anchors = {
            neighbor
            for entity in component
            for neighbor in entity.connected_entities
            if neighbor in parent_entities
        }
        off_registry = sum(not registry_status.get(entity, False) for entity in component)
        rows.append(
            {
                "entities": len(component),
                "internal_edges": edges,
                "cycle_rank": max(0, edges - len(component) + 1),
                "seed_anchors": len(anchors),
                "off_registry_entities": off_registry,
            }
        )
    return rows


def run(args) -> tuple[list[dict], dict]:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    parent = load_assembly(args.parent)
    outer = load_assembly(args.outer)
    registry, registry_diagnostics = build_registry(parent, outer, args.parent_match_tolerance)
    parent_entities = set(parent.entities)
    registry_status: dict[object, bool] = {}
    elapsed = 0.0
    additions = []
    site_time = 0.0
    max_component = {"entities": 0, "cycle_rank": 0, "seed_anchors": 0, "off_registry_entities": 0}

    for event_index in range(1, args.event_cap + 1):
        exposed_before = len(parent.free_carboxylates)
        dt, event, new_entities, selected_site = direct_step(
            parent, parent_entities, args.R_sim, args.p_Zr, args.E_sim
        )
        elapsed += dt
        site_time += exposed_before * dt
        if event == "stalled":
            break
        for entity in new_entities:
            distance = nearest_registry_distance(entity, registry)
            registered = distance <= args.registry_tolerance
            registry_status[entity] = registered
            additions.append(
                {
                    "event": event_index,
                    "model_time": elapsed,
                    "event_type": event,
                    "entity_type": str(entity.entity_type),
                    "registry_distance_A": distance,
                    "is_parent_registry": int(registered),
                    "attached_directly_to_seed": int(
                        selected_site is not None
                        and selected_site.belonging_entity in parent_entities
                    ),
                }
            )
        if new_entities or event == "remove" or event_index % args.inspect_every == 0:
            components = new_components(parent, parent_entities, registry_status)
            for component in components:
                for key in max_component:
                    max_component[key] = max(max_component[key], component[key])
        if len(parent.entities) - len(parent_entities) >= args.growth_increment:
            break

    summary = {
        "schema": "uio66-surface-registry-pilot-v1",
        "diagnostic_only": True,
        "parent_seed": str(args.parent.resolve()),
        "outer_registry_seed": str(args.outer.resolve()),
        "R_sim": args.R_sim,
        "p_Zr": args.p_Zr,
        "E_sim": args.E_sim,
        "event_cap": args.event_cap,
        "growth_increment": args.growth_increment,
        "attempted_events": event_index,
        "elapsed_model_time": elapsed,
        "integrated_exposed_site_time": site_time,
        "accepted_additions": len(additions),
        "registered_additions": sum(row["is_parent_registry"] for row in additions),
        "off_registry_additions": sum(not row["is_parent_registry"] for row in additions),
        "maximum_post_seed_component": max_component,
        "registry_diagnostics": registry_diagnostics,
        "interpretation_rule": (
            "If off-registry additions are absent, the archived geometry engine contains only "
            "epitaxial extension and cannot estimate secondary-surface nucleation without a new "
            "adsorbed-embryo or misregistry event channel."
        ),
    }
    return additions, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=HERE / "seed" / "UiO66_n4_111_octahedron.pkl")
    parser.add_argument("--outer", type=Path, default=HERE / "seed" / "UiO66_n6_111_octahedron.pkl")
    parser.add_argument("--R-sim", dest="R_sim", type=float, default=1.0)
    parser.add_argument("--p-Zr", dest="p_Zr", type=float, default=0.5)
    parser.add_argument("--E-sim", dest="E_sim", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--event-cap", type=int, default=20_000)
    parser.add_argument("--growth-increment", type=int, default=100)
    parser.add_argument("--inspect-every", type=int, default=20)
    parser.add_argument("--parent-match-tolerance", type=float, default=0.2)
    parser.add_argument("--registry-tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=HERE / "pilot_registry_additions.csv")
    args = parser.parse_args()
    additions, summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if additions:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(additions[0]))
            writer.writeheader()
            writer.writerows(additions)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
