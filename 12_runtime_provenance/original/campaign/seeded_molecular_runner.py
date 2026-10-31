"""Propagate a frozen molecular seed with the audited common event selector."""

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

from common_kmc_selector import PhysicalCoordinates, chemistry_record, gillespie_waiting_time, select_event
from common_molecular_runner import (
    _load_engine,
    _record,
    _sample_schedule,
    assembly_state,
    connected_component_sizes,
    sha256,
)
from planar_mol_topology import (
    grow_one_step_planar,
    layer_plane_from_assembly,
    planarity_metrics,
    select_planar_event,
)


_REGENERATED_SEED_CACHE: dict[
    tuple[str, int, int, float, str], tuple[Any, dict[str, Any]]
] = {}


def run_seeded_trace(
    *,
    system: str,
    seed_pickle: Path | None = None,
    seed_generation_random_seed: int | None = None,
    initial_seed_entities: int = 20,
    coordinates: PhysicalCoordinates,
    random_seed: int,
    target_entities: int = 30,
    event_cap: int = 100_000,
    wall_time_cap_s: float = 120.0,
    site_opening_seconds: float | None = None,
    seed_retention_threshold: float = 0.8,
    m6_node_fraction: float = 1.0,
    topology_mode: str = "unrestricted",
    bond_accessibility_mode: str = "all_bonds",
) -> dict[str, Any]:
    coordinates.validate()
    if not 0.0 < seed_retention_threshold <= 1.0:
        raise ValueError("seed_retention_threshold must lie in (0, 1]")
    if (seed_pickle is None) == (seed_generation_random_seed is None):
        raise ValueError(
            "Provide exactly one of seed_pickle or seed_generation_random_seed"
        )
    if not 0.0 <= m6_node_fraction <= 1.0:
        raise ValueError("m6_node_fraction must lie in [0, 1]")
    if topology_mode not in {"unrestricted", "planar_mol"}:
        raise ValueError("topology_mode must be unrestricted or planar_mol")
    if bond_accessibility_mode not in {"all_bonds", "undercoordinated_metal", "dangling_only"}:
        raise ValueError("unsupported bond_accessibility_mode")
    if topology_mode == "planar_mol" and system != "btb_mol":
        raise ValueError("planar_mol topology is defined only for btb_mol")
    if topology_mode == "planar_mol" and m6_node_fraction != 1.0:
        raise ValueError("the current planar_mol construction requires pure M6 nodes")
    function_started = time.perf_counter()
    engine, engine_path, redirects = _load_engine(system)
    engine.MAX_DEPTH = target_entities + 1
    if seed_pickle is not None:
        with seed_pickle.open("rb") as handle:
            assembly = pickle.load(handle)
        seed_source = {
            "kind": "pickle",
            "path": seed_pickle.resolve().as_posix(),
            "sha256": sha256(seed_pickle),
        }
    else:
        from generate_seed_ensemble import generate_one

        cache_key = (
            system,
            int(seed_generation_random_seed),
            initial_seed_entities,
            float(m6_node_fraction),
            topology_mode,
        )
        cached = _REGENERATED_SEED_CACHE.get(cache_key)
        if cached is None:
            assembly, generation_record = generate_one(
                system,
                int(seed_generation_random_seed),
                initial_seed_entities,
                max(30.0, wall_time_cap_s),
                m6_node_fraction,
                topology_mode,
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
    plane = layer_plane_from_assembly(assembly) if topology_mode == "planar_mol" else None
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

    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        before = assembly_state(assembly)
        try:
            selected = (
                select_planar_event(
                    assembly, coordinates, plane, rng, bond_accessibility_mode
                )
                if plane is not None
                else select_event(assembly, coordinates, rng, bond_accessibility_mode)
            )
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
            if plane is None:
                assembly.grow_one_step(selected.selected_site)
            else:
                grow_one_step_planar(
                    assembly,
                    selected.selected_site,
                    engine,
                    plane,
                    m6_node_fraction=m6_node_fraction,
                    rng=rng,
                )
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
    if plane is not None:
        final_state.update(planarity_metrics(assembly, plane))
    final_entities = set(assembly.entities.items)
    final_retained = len(initial_entities.intersection(final_entities))
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
        "topology_mode": topology_mode,
        "bond_accessibility_mode": bond_accessibility_mode,
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
        },
        "wall_seconds": time.perf_counter() - started,
    }
