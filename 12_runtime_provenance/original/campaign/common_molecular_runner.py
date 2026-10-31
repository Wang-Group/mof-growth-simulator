#!/usr/bin/env python3
"""Run UiO-66 or BTB-MOL molecular KMC with one audited event definition.

The archived molecular geometry engines are retained for alignment, collision
checking, bond formation, internal closure, and bond removal.  Their archived
event selector and recipe-to-rate conversion are never called.
"""

from __future__ import annotations

import argparse
import builtins
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types
from typing import Any

import numpy as np

from common_kmc_selector import (
    PhysicalCoordinates,
    chemistry_record,
    counts_from_assembly,
    gillespie_waiting_time,
    ideal_node_fraction,
    select_event,
)
from planar_mol_topology import (
    attach_layer_plane,
    grow_one_step_planar,
    infer_six_connected_m6_plane,
    planarity_metrics,
    select_planar_event,
)


HERE = Path(__file__).resolve().parent
_WORKSPACE_SNAPSHOT = (
    HERE.parents[1]
    / "source_package"
    / "publication_handoff_2026-08-05"
    / "github_release"
    / "source_snapshot"
)
SNAPSHOT = Path(os.environ.get("MOL_GROWTH_SOURCE_SNAPSHOT", _WORKSPACE_SNAPSHOT))
ENGINE_SPECS = {
    "uio66": {
        "path": SNAPSHOT
        / "UiO-66"
        / "code"
        / "UiO66_Assembly_Large_Correction_20250811.py",
        "initial_class": "Zr6_AA",
        "monomer_roots": [
            SNAPSHOT / "UiO-66" / "Zr-Ligand-ExplicitChannel" / "input_monomers",
            SNAPSHOT / "KMC_example" / "input_monomers",
        ],
        "monomers": {"Zr6_AA.mol", "Zr12_AA.mol", "BDC.mol"},
    },
    "btb_mol": {
        "path": SNAPSHOT
        / "KMC_example"
        / "MOL_KMC"
        / "MOL_Assembly_Large_Correction_20250811.py",
        "initial_class": "Zr6",
        "monomer_roots": [SNAPSHOT / "KMC_example" / "input_monomers"],
        "monomers": {"Zr6.mol", "Zr12.mol", "BTBa.mol", "BTBb.mol"},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompatibleKDTree:
    """Pickle-safe NumPy substitute for the cKDTree API used by old engines.

    The class must remain at module scope.  Assemblies retain their spatial
    index, and a function-local compatibility class cannot be serialized for
    frozen-seed propagation runs on hosts where SciPy is unavailable.
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def query_ball_tree(self, other, r):
        if self.data.size == 0 or other.data.size == 0:
            return [[] for _ in range(len(self.data))]
        radius_squared = float(r) ** 2
        differences = self.data[:, None, :] - other.data[None, :, :]
        distance_squared = np.sum(differences * differences, axis=2)
        return [
            np.where(row <= radius_squared)[0].tolist()
            for row in distance_squared
        ]


def _install_dependency_shims_if_needed() -> list[str]:
    """Provide only the numerical interfaces used by the archived engines."""

    installed: list[str] = []
    if "scipy" not in sys.modules and importlib.util.find_spec("scipy") is None:
        scipy_module = types.ModuleType("scipy")
        spatial_module = types.ModuleType("scipy.spatial")
        linalg_module = types.ModuleType("scipy.linalg")

        spatial_module.cKDTree = CompatibleKDTree
        linalg_module.svd = np.linalg.svd
        scipy_module.spatial = spatial_module
        scipy_module.linalg = linalg_module
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.spatial"] = spatial_module
        sys.modules["scipy.linalg"] = linalg_module
        installed.append("numpy-compatible scipy.spatial.cKDTree and scipy.linalg.svd shim")

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        matplotlib_module = types.ModuleType("matplotlib")
        pyplot_module = types.ModuleType("matplotlib.pyplot")
        matplotlib_module.pyplot = pyplot_module
        sys.modules["matplotlib"] = matplotlib_module
        sys.modules["matplotlib.pyplot"] = pyplot_module
        installed.append("unused matplotlib.pyplot import shim")
    return installed


def _load_engine(system: str):
    spec_record = ENGINE_SPECS[system]
    engine_path = Path(spec_record["path"])
    if not engine_path.is_file():
        raise FileNotFoundError(f"Archived geometry engine not found: {engine_path}")
    module_name = f"common_kmc_{system}_{sha256(engine_path)[:12]}"
    if module_name in sys.modules:
        return sys.modules[module_name], engine_path, []

    dependency_adaptations = _install_dependency_shims_if_needed()
    module_spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import {engine_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module

    original_open = builtins.open
    redirects: list[str] = []

    def redirected_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            candidate = Path(file)
            if (
                candidate.as_posix().startswith("/public/home/yibinjiang/workspace/mol/")
                and candidate.name in spec_record["monomers"]
            ):
                replacement = next(
                    (
                        Path(root) / candidate.name
                        for root in spec_record["monomer_roots"]
                        if (Path(root) / candidate.name).is_file()
                    ),
                    None,
                )
                if replacement is None:
                    raise FileNotFoundError(f"No portable monomer for {candidate.name}")
                redirects.append(f"{candidate.as_posix()} -> {replacement.as_posix()}")
                file = replacement
        return original_open(file, *args, **kwargs)

    builtins.open = redirected_open
    try:
        module_spec.loader.exec_module(module)
    finally:
        builtins.open = original_open
    module._common_runner_redirects = dependency_adaptations + sorted(set(redirects))
    return module, engine_path, module._common_runner_redirects


def connected_component_sizes(assembly: Any) -> list[int]:
    """Return connected entity-component sizes in descending order."""

    entities = list(assembly.entities)
    unseen = set(entities)
    sizes: list[int] = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        size = 0
        while stack:
            entity = stack.pop()
            size += 1
            for neighbor in getattr(entity, "connected_entities", ()): 
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def assembly_state(assembly: Any) -> dict[str, Any]:
    entity_count = len(assembly.entities)
    linked_bonds = len(assembly.linked_carboxylate_pairs)
    entity_types = Counter(str(getattr(x, "entity_type", "unknown")) for x in assembly.entities)
    counts = counts_from_assembly(assembly)
    return {
        "entities": int(entity_count),
        "linked_bonds": int(linked_bonds),
        "cycle_rank": int(max(0, linked_bonds - entity_count + 1)),
        "linker_free_sites": int(counts.linker_free_sites),
        "metal_free_sites": int(counts.metal_free_sites),
        "ready_internal_pairs": int(counts.ready_internal_pairs),
        "entity_type_counts": dict(entity_types),
    }


def _sample_schedule(event_cap: int, points: int = 180) -> set[int]:
    schedule = set(range(1, min(event_cap, 50) + 1))
    if event_cap <= 50:
        return schedule
    lo = math.log(50.0)
    hi = math.log(float(event_cap))
    for index in range(points):
        fraction = index / max(points - 1, 1)
        schedule.add(int(round(math.exp(lo + fraction * (hi - lo)))))
    schedule.add(event_cap)
    return schedule


def _record(state: dict[str, Any], event: int, model_time: float) -> dict[str, Any]:
    return {"event": int(event), "model_time": float(model_time), **state}


def _update_excursion_outcomes(
    excursion: dict[str, Any],
    outcome: str,
    by_size: dict[str, dict[str, int]],
    by_size_cycle: dict[str, dict[str, int]],
) -> None:
    for size in excursion["visited_sizes"]:
        record = by_size.setdefault(
            str(size), {"reached": 0, "target": 0, "collapse": 0, "censored": 0}
        )
        record["reached"] += 1
        record[outcome] += 1
    for state_key in excursion["visited_size_cycle"]:
        record = by_size_cycle.setdefault(
            state_key, {"reached": 0, "target": 0, "collapse": 0, "censored": 0}
        )
        record["reached"] += 1
        record[outcome] += 1


def run_trace(
    *,
    system: str,
    coordinates: PhysicalCoordinates,
    seed: int,
    target_entities: int = 30,
    event_cap: int = 100_000,
    wall_time_cap_s: float = 120.0,
    bumping_threshold: float = 2.0,
    m6_node_fraction: float = 1.0,
    max_depth: int = 3,
    site_opening_seconds: float | None = None,
    topology_mode: str = "unrestricted",
    bond_accessibility_mode: str = "all_bonds",
) -> dict[str, Any]:
    """Execute one trace and return analysis-ready molecular observables."""

    coordinates.validate()
    if system not in ENGINE_SPECS:
        raise ValueError(f"Unsupported system {system!r}")
    if target_entities < 2 or event_cap < 1 or wall_time_cap_s <= 0.0:
        raise ValueError("Invalid target, event cap, or wall-time cap")
    if not 0.0 <= m6_node_fraction <= 1.0:
        raise ValueError("m6_node_fraction must lie in [0, 1]")
    if site_opening_seconds is not None and site_opening_seconds <= 0.0:
        raise ValueError("site_opening_seconds must be positive")
    if topology_mode not in {"unrestricted", "planar_mol"}:
        raise ValueError("topology_mode must be unrestricted or planar_mol")
    if bond_accessibility_mode not in {"all_bonds", "undercoordinated_metal", "dangling_only"}:
        raise ValueError("unsupported bond_accessibility_mode")
    if topology_mode == "planar_mol" and system != "btb_mol":
        raise ValueError("planar_mol topology is defined only for btb_mol")
    if topology_mode == "planar_mol" and m6_node_fraction != 1.0:
        raise ValueError("the current planar_mol construction requires pure M6 nodes")

    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    rng = random.Random(seed ^ 0x5EEDC0DE)
    engine, engine_path, redirects = _load_engine(system)
    # ``remove_linkage`` in the archived geometry engines uses a bounded
    # bidirectional graph search to decide whether breaking a bond detaches a
    # fragment.  The archived default (three graph layers) is only an
    # optimization and is not exact for the larger candidates analysed here.
    # A trace stops as soon as ``target_entities`` is reached, so searching at
    # least target + 1 layers is sufficient to resolve every component in that
    # trace exactly.  ``max_depth`` is retained as a backward-compatible lower
    # bound and is recorded in the output.
    effective_fragment_search_depth = max(int(max_depth), int(target_entities) + 1)
    engine.MAX_DEPTH = effective_fragment_search_depth
    initial_entity = getattr(engine, ENGINE_SPECS[system]["initial_class"])()
    assembly = engine.Assembly(
        initial_entity,
        m6_node_fraction,
        0.0,
        bumping_threshold,
    )
    plane = None
    if topology_mode == "planar_mol":
        plane = infer_six_connected_m6_plane(initial_entity)
        attach_layer_plane(assembly, plane)

    event_counts = Counter()
    first_passage_time: dict[str, float | None] = {
        str(level): None for level in range(2, target_entities + 1)
    }
    first_passage_event: dict[str, int | None] = {
        str(level): None for level in range(2, target_entities + 1)
    }
    outcome_by_size: dict[str, dict[str, int]] = {}
    outcome_by_size_cycle: dict[str, dict[str, int]] = {}
    excursion_counts = Counter()
    active_excursion: dict[str, Any] | None = None
    samples = [_record(assembly_state(assembly), 0, 0.0)]
    schedule = _sample_schedule(event_cap)
    model_time = 0.0
    entity_time_integral = 0.0
    maximum_entities = 1
    termination = "event_cap"
    executed_events = 0

    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        before = assembly_state(assembly)
        try:
            selection = (
                select_planar_event(
                    assembly, coordinates, plane, rng, bond_accessibility_mode
                )
                if plane is not None
                else select_event(assembly, coordinates, rng, bond_accessibility_mode)
            )
        except RuntimeError as exc:
            if "no enabled event" not in str(exc):
                raise
            termination = "no_enabled_event"
            break

        delta_time = gillespie_waiting_time(selection.propensities.total, rng)
        entity_time_integral += before["entities"] * delta_time
        model_time += delta_time

        if selection.event in {"add_node", "add_linker"}:
            event_counts[f"{selection.event}_attempt"] += 1
            if plane is None:
                assembly.grow_one_step(selection.selected_site)
            else:
                grow_one_step_planar(
                    assembly,
                    selection.selected_site,
                    engine,
                    plane,
                    m6_node_fraction=m6_node_fraction,
                    rng=rng,
                )
            if len(assembly.entities) > before["entities"]:
                event_counts[f"{selection.event}_accepted"] += 1
            else:
                event_counts[f"{selection.event}_collision_rejected"] += 1
        elif selection.event == "close_internal":
            assembly.link_internal_carboxylate(selection.selected_pair)
            event_counts["close_internal"] += 1
        elif selection.event == "remove_bond":
            assembly.remove_linkage(selection.selected_pair)
            event_counts["remove_bond"] += 1
        else:  # pragma: no cover - protected by selector type
            raise RuntimeError(f"Unknown selected event {selection.event}")

        executed_events = event_index
        after = assembly_state(assembly)
        maximum_entities = max(maximum_entities, after["entities"])
        for level in range(2, min(after["entities"], target_entities) + 1):
            key = str(level)
            if first_passage_time[key] is None:
                first_passage_time[key] = model_time
                first_passage_event[key] = event_index

        if active_excursion is None and before["entities"] <= 1 < after["entities"]:
            excursion_counts["total"] += 1
            active_excursion = {
                "start_event": event_index,
                "start_model_time": model_time,
                "visited_sizes": set(),
                "visited_size_cycle": set(),
            }
        if active_excursion is not None:
            active_excursion["visited_sizes"].add(after["entities"])
            active_excursion["visited_size_cycle"].add(
                f"{after['entities']}|{after['cycle_rank']}"
            )

        if event_index in schedule:
            samples.append(_record(after, event_index, model_time))

        if after["entities"] >= target_entities:
            termination = "target_reached"
            excursion_counts["target"] += 1
            if active_excursion is not None:
                _update_excursion_outcomes(
                    active_excursion, "target", outcome_by_size, outcome_by_size_cycle
                )
                active_excursion = None
            break
        if active_excursion is not None and after["entities"] <= 1:
            excursion_counts["collapse"] += 1
            _update_excursion_outcomes(
                active_excursion, "collapse", outcome_by_size, outcome_by_size_cycle
            )
            active_excursion = None

    if active_excursion is not None:
        excursion_counts["censored"] += 1
        _update_excursion_outcomes(
            active_excursion, "censored", outcome_by_size, outcome_by_size_cycle
        )

    final_state = assembly_state(assembly)
    final_state["connected_component_sizes"] = connected_component_sizes(assembly)
    final_state["is_single_connected_component"] = (
        final_state["connected_component_sizes"] == [final_state["entities"]]
    )
    if plane is not None:
        final_state.update(planarity_metrics(assembly, plane))
    if samples[-1]["event"] != executed_events:
        samples.append(_record(final_state, executed_events, model_time))
    wall_seconds = time.perf_counter() - started
    clock = {
        "model_time": model_time,
        "reference": "one dimensionless unit equals one declared site-opening time",
        "site_opening_seconds": site_opening_seconds,
        "physical_seconds": (
            model_time * site_opening_seconds if site_opening_seconds is not None else None
        ),
    }
    return {
        "schema_version": "common-molecular-kmc-v1",
        "system": system,
        "topology_mode": topology_mode,
        "bond_accessibility_mode": bond_accessibility_mode,
        "seed": int(seed),
        "coordinates": chemistry_record(coordinates),
        "node_population": {
            "m6_fraction": float(m6_node_fraction),
            "m12_fraction": float(1.0 - m6_node_fraction),
            "interpretation": (
                "Probability that an added metal node is the engine's M6 class; "
                "this is not a reactive-node fraction."
            ),
        },
        "clock": clock,
        "termination": {
            "status": termination,
            "target_entities": target_entities,
            "event_cap": event_cap,
            "executed_events": executed_events,
            "wall_time_cap_s": wall_time_cap_s,
            "maximum_entities": maximum_entities,
            "requested_fragment_search_depth": int(max_depth),
            "effective_fragment_search_depth": effective_fragment_search_depth,
        },
        "event_counts": dict(event_counts),
        "first_passage_model_time": first_passage_time,
        "first_passage_event": first_passage_event,
        "excursions": {
            "counts": dict(excursion_counts),
            "outcome_by_size": outcome_by_size,
            "outcome_by_size_cycle_rank": outcome_by_size_cycle,
            "interpretation": (
                "Each excursion begins when a one-entity candidate first reaches two entities. "
                "Target, return to one entity, and capped paths are resolved as target, collapse, "
                "and right-censored outcomes."
            ),
        },
        "derived": {
            "mean_entities_time_weighted": (
                entity_time_integral / model_time if model_time > 0.0 else 1.0
            ),
            "target_reached": termination == "target_reached",
        },
        "final_state": final_state,
        "sampled_growth_curve": samples,
        "provenance": {
            "engine_path": engine_path.as_posix(),
            "engine_sha256": sha256(engine_path),
            "engine_import_redirects": redirects,
            "runner_path": Path(__file__).resolve().as_posix(),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "selector_path": (HERE / "common_kmc_selector.py").as_posix(),
            "selector_sha256": sha256(HERE / "common_kmc_selector.py"),
            "event_selector": "compatible-site physical propensities; no fallback",
            "closure_model": "constant exp(30.9*SC); no entity-number correction",
            "fragment_removal": (
                "archived component-removal routine with exact search depth "
                "at least target_entities + 1"
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "wall_seconds": wall_seconds,
    }


def _coordinates_from_args(args: argparse.Namespace) -> tuple[PhysicalCoordinates, dict[str, Any]]:
    if args.node_fraction is not None:
        node_fraction = args.node_fraction
        mapping = {"kind": "direct_node_fraction", "node_fraction": node_fraction}
    else:
        node_fraction = ideal_node_fraction(args.atomic_metal_linker_ratio)
        mapping = {
            "kind": "ideal_equal_capture",
            "atomic_metal_to_linker_ratio": args.atomic_metal_linker_ratio,
            "formula": "f_node=q/(q+6)",
            "node_fraction": node_fraction,
        }
    return (
        PhysicalCoordinates(
            external_activity=args.external_activity,
            node_fraction=node_fraction,
            off_weight=args.off_weight,
            closure_entropy_coefficient=args.sc,
        ),
        mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(ENGINE_SPECS), required=True)
    composition = parser.add_mutually_exclusive_group(required=True)
    composition.add_argument("--node-fraction", type=float)
    composition.add_argument("--atomic-metal-linker-ratio", type=float)
    parser.add_argument("--external-activity", type=float, required=True)
    parser.add_argument("--off-weight", type=float, required=True)
    parser.add_argument("--sc", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-entities", type=int, default=30)
    parser.add_argument("--event-cap", type=int, default=100_000)
    parser.add_argument("--wall-time-cap-s", type=float, default=120.0)
    parser.add_argument("--bumping-threshold", type=float, default=2.0)
    parser.add_argument("--m6-node-fraction", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--site-opening-seconds", type=float)
    parser.add_argument(
        "--topology-mode", choices=("unrestricted", "planar_mol"), default="unrestricted"
    )
    parser.add_argument(
        "--bond-accessibility-mode",
        choices=("all_bonds", "undercoordinated_metal", "dangling_only"),
        default="all_bonds",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coordinates, composition_mapping = _coordinates_from_args(args)
    try:
        result = run_trace(
            system=args.system,
            coordinates=coordinates,
            seed=args.seed,
            target_entities=args.target_entities,
            event_cap=args.event_cap,
            wall_time_cap_s=args.wall_time_cap_s,
            bumping_threshold=args.bumping_threshold,
            m6_node_fraction=args.m6_node_fraction,
            max_depth=args.max_depth,
            site_opening_seconds=args.site_opening_seconds,
            topology_mode=args.topology_mode,
            bond_accessibility_mode=args.bond_accessibility_mode,
        )
        result["composition_mapping"] = composition_mapping
    except Exception as exc:
        result = {
            "schema_version": "common-molecular-kmc-error-v1",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    if result.get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
