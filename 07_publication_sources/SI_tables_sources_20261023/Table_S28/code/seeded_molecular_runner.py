"""Grow a frozen molecular seed with the audited common event selector."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time
from typing import Any

import numpy as np

from btb_postgrowth_structure import analyze_btb_assembly
from common_kmc_selector import PhysicalCoordinates, chemistry_record, gillespie_waiting_time, select_event
from common_molecular_runner import (
    _load_engine,
    _record,
    _sample_schedule,
    assembly_state,
    connected_component_sizes,
    sha256,
)
_REGENERATED_SEED_CACHE: dict[
    tuple[str, int, int, tuple[float, float, float, float, float], float],
    tuple[Any, dict[str, Any]],
] = {}


def _largest_stable_layer(metrics: dict[str, Any]) -> dict[str, Any] | None:
    stable = [
        domain
        for domain in metrics.get("layer_domains", ())
        if int(domain.get("local_plane_observations", 0)) >= int(
            metrics.get("minimum_local_planes_for_stable_layer", 2)
        )
    ]
    if not stable:
        return None
    return max(
        stable,
        key=lambda domain: (
            int(domain.get("supporting_metal_nodes", 0)),
            int(domain.get("local_plane_observations", 0)),
        ),
    )


def _layer_domains_relative_to_seed(
    metrics: dict[str, Any],
    seed_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify stable BTB layer domains relative to the original seed plane."""

    if seed_reference is None:
        return {
            "status": "seed_has_no_stable_reference_layer",
            "primary_registered_domains": 0,
            "daughter_layer_domains": None,
            "domains": [],
        }
    reference_normal = np.asarray(seed_reference["normal"], dtype=float)
    reference_normal /= np.linalg.norm(reference_normal)
    reference_center = np.asarray(seed_reference["center_angstrom"], dtype=float)
    stable_threshold = int(metrics.get("minimum_local_planes_for_stable_layer", 2))
    angle_tolerance = float(metrics.get("angle_tolerance_deg", 7.5))
    offset_tolerance = float(metrics.get("offset_tolerance_angstrom", 3.0))
    records: list[dict[str, Any]] = []
    for index, domain in enumerate(metrics.get("layer_domains", ())):
        if int(domain.get("local_plane_observations", 0)) < stable_threshold:
            continue
        normal = np.asarray(domain["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        alignment = float(np.clip(abs(np.dot(normal, reference_normal)), 0.0, 1.0))
        angle = math.degrees(math.acos(alignment))
        center = np.asarray(domain["center_angstrom"], dtype=float)
        offset = abs(float(np.dot(center - reference_center, reference_normal)))
        primary = angle <= angle_tolerance and offset <= offset_tolerance
        records.append(
            {
                "domain_index": index,
                "local_plane_observations": int(domain["local_plane_observations"]),
                "supporting_metal_nodes": int(domain["supporting_metal_nodes"]),
                "angle_from_seed_plane_deg": angle,
                "normal_offset_from_seed_plane_angstrom": offset,
                "registered_with_seed_plane": primary,
            }
        )
    primary_records = [record for record in records if record["registered_with_seed_plane"]]
    daughter_records = [record for record in records if not record["registered_with_seed_plane"]]
    metal_nodes = int(metrics.get("metal_nodes", 0))
    largest_primary_support = max(
        (record["supporting_metal_nodes"] for record in primary_records), default=0
    )
    largest_daughter_support = max(
        (record["supporting_metal_nodes"] for record in daughter_records), default=0
    )
    return {
        "status": "measured",
        "definition": (
            "A stable domain is registered with the original layer when its plane normal is within "
            f"{angle_tolerance:g} degrees and its normal offset is within {offset_tolerance:g} angstrom. "
            "Other stable domains are counted as daughter layer domains."
        ),
        "primary_registered_domains": len(primary_records),
        "daughter_layer_domains": len(daughter_records),
        "largest_primary_supporting_metal_nodes": largest_primary_support,
        "largest_primary_supporting_metal_fraction": (
            largest_primary_support / metal_nodes if metal_nodes else None
        ),
        "largest_daughter_supporting_metal_nodes": largest_daughter_support,
        "largest_daughter_supporting_metal_fraction": (
            largest_daughter_support / metal_nodes if metal_nodes else None
        ),
        "domains": records,
    }


def run_seeded_trace(
    *,
    system: str,
    seed_pickle: Path | None = None,
    seed_generation_random_seed: int | None = None,
    seed_generation_coordinates: PhysicalCoordinates | None = None,
    initial_seed_entities: int = 20,
    coordinates: PhysicalCoordinates,
    random_seed: int,
    target_entities: int = 30,
    event_cap: int = 100_000,
    wall_time_cap_s: float = 120.0,
    site_opening_seconds: float | None = None,
    seed_retention_threshold: float = 0.8,
    m6_node_fraction: float = 1.0,
    structure_metric_milestones: tuple[int, ...] | None = None,
    mol2_output_path: Path | None = None,
    mol2_milestones: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    coordinates.validate()
    if not 0.0 < seed_retention_threshold <= 1.0:
        raise ValueError("seed_retention_threshold must lie in (0, 1]")
    if (seed_pickle is None) == (seed_generation_random_seed is None):
        raise ValueError(
            "Provide exactly one of seed_pickle or seed_generation_random_seed"
        )
    if seed_pickle is not None and seed_generation_coordinates is not None:
        raise ValueError(
            "seed_generation_coordinates is only valid with deterministic seed regeneration"
        )
    if seed_generation_random_seed is not None and seed_generation_coordinates is None:
        raise ValueError(
            "seed_generation_coordinates must be supplied when regenerating a seed"
        )
    if not 0.0 <= m6_node_fraction <= 1.0:
        raise ValueError("m6_node_fraction must lie in [0, 1]")
    function_started = time.perf_counter()
    engine, engine_path, redirects = _load_engine(system)
    engine.MAX_DEPTH = target_entities + 1
    if seed_pickle is not None:
        with seed_pickle.open("rb") as handle:
            assembly = pickle.load(handle)
        # The spatial tree is a machine-specific cache, not molecular state.
        # Rebuild it with the active geometry runtime after deserialization.
        for entity in assembly.entities.items:
            if hasattr(entity, "kdtree"):
                entity.kdtree = None
        seed_source = {
            "kind": "pickle",
            "path": seed_pickle.resolve().as_posix(),
            "sha256": sha256(seed_pickle),
        }
    else:
        from generate_seed_ensemble import generate_one

        seed_generation_coordinates.validate()
        generation_coordinate_key = (
            float(seed_generation_coordinates.external_activity),
            float(seed_generation_coordinates.node_fraction),
            float(seed_generation_coordinates.off_weight),
            float(seed_generation_coordinates.closure_entropy_coefficient),
            float(seed_generation_coordinates.closure_entropy_scale),
        )

        cache_key = (
            system,
            int(seed_generation_random_seed),
            initial_seed_entities,
            generation_coordinate_key,
            float(m6_node_fraction),
        )
        cached = _REGENERATED_SEED_CACHE.get(cache_key)
        if cached is None:
            assembly, generation_record = generate_one(
                system,
                int(seed_generation_random_seed),
                initial_seed_entities,
                max(30.0, wall_time_cap_s),
                seed_generation_coordinates,
                m6_node_fraction,
            )
            _REGENERATED_SEED_CACHE[cache_key] = (
                copy.deepcopy(assembly),
                copy.deepcopy(generation_record),
            )
            cache_hit = False
        else:
            assembly = copy.deepcopy(cached[0])
            generation_record = copy.deepcopy(cached[1])
            cache_hit = True
        seed_source = {
            "kind": "deterministic_regeneration_before_clock_start",
            "generation_random_seed": int(seed_generation_random_seed),
            "m6_node_fraction": float(m6_node_fraction),
            "m12_node_fraction": float(1.0 - m6_node_fraction),
            "generation_record": generation_record,
            "process_local_cache_hit": cache_hit,
        }
    seed_preparation_seconds = time.perf_counter() - function_started
    random.seed(random_seed)
    np.random.seed(random_seed % (2**32 - 1))
    rng = random.Random(random_seed ^ 0x5EEDBEEF)
    started = time.perf_counter()
    initial_state = assembly_state(assembly)
    initial_entities = set(assembly.entities.items)
    initial_count = len(initial_entities)
    if initial_count >= target_entities:
        raise ValueError("target_entities must exceed the frozen-seed size")
    if connected_component_sizes(assembly) != [initial_count]:
        raise ValueError("frozen seed is disconnected")
    requested_structure_milestones = tuple(
        sorted(
            {
                int(value)
                for value in (structure_metric_milestones or ())
                if initial_count < int(value) <= target_entities
            }
        )
    )
    requested_mol2_milestones = tuple(
        sorted(
            {
                int(value)
                for value in (mol2_milestones or ())
                if initial_count < int(value) <= target_entities
            }
        )
    )
    if requested_mol2_milestones and mol2_output_path is None:
        raise ValueError("mol2_milestones require mol2_output_path")
    initial_btb_dimensionality = (
        analyze_btb_assembly(assembly) if system == "btb_mol" else None
    )
    seed_layer_reference = (
        _largest_stable_layer(initial_btb_dimensionality)
        if initial_btb_dimensionality is not None
        else None
    )
    fingerprint_payload = {
        "entity_types": [type(entity).__name__ for entity in assembly.entities.items],
        "centers": [
            [round(float(value), 10) for value in entity.center]
            for entity in assembly.entities.items
        ],
        "adjacency": [
            sorted(
                assembly.entities.items.index(neighbor)
                for neighbor in getattr(entity, "connected_entities", ())
                if neighbor in assembly.entities.items
            )
            for entity in assembly.entities.items
        ],
    }
    seed_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    event_counts = Counter()
    first_passage_model_time = {
        str(level): None for level in range(initial_count + 1, target_entities + 1)
    }
    first_passage_event = {
        str(level): None for level in range(initial_count + 1, target_entities + 1)
    }
    schedule = _sample_schedule(event_cap)
    samples = [_record(initial_state, 0, 0.0)]
    model_time = 0.0
    entity_time_integral = 0.0
    maximum_entities = initial_count
    minimum_seed_retained = initial_count
    first_seed_loss_event = None
    first_seed_loss_model_time = None
    termination = "event_cap"
    executed_events = 0
    structure_evolution: list[dict[str, Any]] = []
    pending_structure_milestones = list(requested_structure_milestones)
    mol2_milestone_records: list[dict[str, Any]] = []
    pending_mol2_milestones = list(requested_mol2_milestones)

    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        before = assembly_state(assembly)
        try:
            selected = select_event(assembly, coordinates, rng)
        except RuntimeError as exc:
            if "no enabled" not in str(exc):
                raise
            termination = "no_enabled_event"
            break
        delta_time = gillespie_waiting_time(selected.propensities.total, rng)
        entity_time_integral += before["entities"] * delta_time
        model_time += delta_time
        if selected.event in {"add_node", "add_linker"}:
            event_counts[f"{selected.event}_attempt"] += 1
            assembly.grow_one_step(selected.selected_site)
            if len(assembly.entities) > before["entities"]:
                event_counts[f"{selected.event}_accepted"] += 1
            else:
                event_counts[f"{selected.event}_collision_rejected"] += 1
        elif selected.event == "close_internal":
            assembly.link_internal_carboxylate(selected.selected_pair)
            event_counts["close_internal"] += 1
        else:
            assembly.remove_linkage(selected.selected_pair)
            event_counts["remove_bond"] += 1
        executed_events = event_index
        after = assembly_state(assembly)
        maximum_entities = max(maximum_entities, after["entities"])
        while (
            pending_mol2_milestones
            and after["entities"] >= pending_mol2_milestones[0]
        ):
            milestone = pending_mol2_milestones.pop(0)
            base = Path(mol2_output_path)
            snapshot = base.with_name(f"{base.stem}_N{milestone:04d}{base.suffix}")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            assembly.get_mol2_file(str(snapshot))
            mol2_milestone_records.append(
                {
                    "requested_entities": milestone,
                    "observed_entities": int(after["entities"]),
                    "event": int(event_index),
                    "model_time": float(model_time),
                    "path": snapshot.resolve().as_posix(),
                    "sha256": sha256(snapshot),
                }
            )
        if system == "btb_mol":
            while (
                pending_structure_milestones
                and after["entities"] >= pending_structure_milestones[0]
            ):
                milestone = pending_structure_milestones.pop(0)
                dimensionality = analyze_btb_assembly(assembly)
                structure_evolution.append(
                    {
                        "requested_entity_milestone": milestone,
                        "observed_entities": after["entities"],
                        "event": event_index,
                        "model_time": model_time,
                        "metrics": dimensionality,
                        "relative_to_seed_layer": _layer_domains_relative_to_seed(
                            dimensionality, seed_layer_reference
                        ),
                    }
                )
        current_entities = set(assembly.entities.items)
        retained = len(initial_entities.intersection(current_entities))
        minimum_seed_retained = min(minimum_seed_retained, retained)
        if (
            first_seed_loss_event is None
            and retained / initial_count < seed_retention_threshold
        ):
            first_seed_loss_event = event_index
            first_seed_loss_model_time = model_time
        for level in range(initial_count + 1, min(after["entities"], target_entities) + 1):
            key = str(level)
            if first_passage_model_time[key] is None:
                first_passage_model_time[key] = model_time
                first_passage_event[key] = event_index
        if event_index in schedule:
            sample = _record(after, event_index, model_time)
            sample["retained_seed_entities"] = retained
            samples.append(sample)
        if after["entities"] >= target_entities:
            termination = "target_reached"
            break

    final_state = assembly_state(assembly)
    final_state["connected_component_sizes"] = connected_component_sizes(assembly)
    final_state["is_single_connected_component"] = (
        final_state["connected_component_sizes"] == [final_state["entities"]]
    )
    final_entities = set(assembly.entities.items)
    final_retained = len(initial_entities.intersection(final_entities))
    final_btb_dimensionality = (
        analyze_btb_assembly(assembly) if system == "btb_mol" else None
    )
    final_btb_relative_to_seed = (
        _layer_domains_relative_to_seed(final_btb_dimensionality, seed_layer_reference)
        if final_btb_dimensionality is not None
        else None
    )
    if final_btb_dimensionality is not None:
        final_state["post_growth_dimensionality"] = final_btb_dimensionality
        final_state["layer_domains_relative_to_seed"] = final_btb_relative_to_seed
    mol2_record: dict[str, Any] | None = None
    if mol2_output_path is not None:
        destination = Path(mol2_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assembly.get_mol2_file(str(destination))
        mol2_record = {
            "path": destination.resolve().as_posix(),
            "sha256": sha256(destination),
            "entities": final_state["entities"],
        }
    if samples[-1]["event"] != executed_events:
        sample = _record(final_state, executed_events, model_time)
        sample["retained_seed_entities"] = final_retained
        samples.append(sample)
    clock = {
        "model_time": model_time,
        "reference": "one dimensionless unit equals one declared site-opening time",
        "site_opening_seconds": site_opening_seconds,
        "physical_seconds": (
            model_time * site_opening_seconds
            if site_opening_seconds is not None
            else None
        ),
    }
    return {
        "schema_version": "common-seeded-molecular-kmc-v1",
        "system": system,
        "topology_mode": "unrestricted",
        "bond_opening_model": "one equal per-bond opening weight",
        "random_seed": random_seed,
        "seed": {
            "source": seed_source,
            "structural_fingerprint_sha256": seed_fingerprint,
            "preparation_seconds_excluded_from_model_clock": seed_preparation_seconds,
            "initial_entity_count": initial_count,
            "initial_state": initial_state,
            "retention_threshold": seed_retention_threshold,
            "minimum_retained_entities": minimum_seed_retained,
            "final_retained_entities": final_retained,
            "final_retained_fraction": final_retained / initial_count,
            "first_loss_event": first_seed_loss_event,
            "first_loss_model_time": first_seed_loss_model_time,
        },
        "coordinates": chemistry_record(coordinates),
        "clock": clock,
        "termination": {
            "status": termination,
            "target_entities": target_entities,
            "event_cap": event_cap,
            "executed_events": executed_events,
            "wall_time_cap_s": wall_time_cap_s,
            "maximum_entities": maximum_entities,
            "effective_fragment_search_depth": target_entities + 1,
        },
        "event_counts": dict(event_counts),
        "first_passage_model_time": first_passage_model_time,
        "first_passage_event": first_passage_event,
        "sampled_growth_curve": samples,
        "derived": {
            "target_reached": termination == "target_reached",
            "net_entity_change": final_state["entities"] - initial_count,
            "mean_entities_time_weighted": (
                entity_time_integral / model_time
                if model_time > 0.0
                else float(initial_count)
            ),
            "initial_btb_dimensionality": initial_btb_dimensionality,
            "seed_layer_reference": seed_layer_reference,
            "post_growth_structure_evolution": structure_evolution,
        },
        "final_state": final_state,
        "provenance": {
            "engine_path": engine_path.as_posix(),
            "engine_sha256": sha256(engine_path),
            "engine_import_redirects": redirects,
            "runner_path": Path(__file__).resolve().as_posix(),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "selector_path": (Path(__file__).resolve().parent / "common_kmc_selector.py").as_posix(),
            "selector_sha256": sha256(Path(__file__).resolve().parent / "common_kmc_selector.py"),
            "fragment_removal": "exact search depth at least target_entities + 1",
            "btb_layer_domain_measurement": (
                "post-growth measurement only; growth remained unrestricted"
                if system == "btb_mol"
                else "not applicable"
            ),
            "mol2_output": mol2_record,
            "mol2_milestones": mol2_milestone_records,
        },
        "wall_seconds": time.perf_counter() - started,
    }
