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
import importlib
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
from btb_postgrowth_structure import analyze_btb_assembly
HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = Path(
    os.environ.get("MOL_GROWTH_REVIEW_BUNDLE", HERE.parents[1])
).resolve()
CANDIDATE_CODE = BUNDLE_ROOT / "code" / "candidate_corrected"
MONOMER_DIR = BUNDLE_ROOT / "models" / "workstation_source_20260831" / "mol"
APPROVED_DERIVED_MODEL_DIR = BUNDLE_ROOT / "models" / "approved_derived"
ENGINE_SPECS = {
    "uio66": {
        "path": CANDIDATE_CODE / "UiO66_Assembly_Large_Correction_20250811_candidate.py",
        "initial_class": "Zr6_AA",
        "monomer_files": {
            "Zr6_AA.mol": MONOMER_DIR / "Zr6_AA.mol",
            "Zr12_AA.mol": MONOMER_DIR / "Zr12_AA.mol",
            "BDC.mol": APPROVED_DERIVED_MODEL_DIR / "BDC_explicit_H.mol",
        },
    },
    "btb_mol": {
        "path": CANDIDATE_CODE / "MOL_Assembly_Large_Correction_20250811_candidate.py",
        "initial_class": "Zr6",
        "monomer_files": {
            "Zr6.mol": MONOMER_DIR / "Zr6.mol",
            "Zr12.mol": MONOMER_DIR / "Zr12.mol",
            "BTBa.mol": MONOMER_DIR / "BTBa.mol",
            "BTBb.mol": MONOMER_DIR / "BTBb.mol",
        },
    },
    "mof808_btc": {
        # Reuse the current audited UiO geometry machinery, but supply the
        # uncapped 12-direction Zr6 model and the explicit-H tritopic BTC
        # linker.  Growth remains unrestricted; six-connected MOF-808-like
        # topology must emerge and is assessed after growth.
        "path": CANDIDATE_CODE / "UiO66_Assembly_Large_Correction_20250811_candidate.py",
        "initial_class": "Zr6_AA",
        "monomer_files": {
            "Zr6_AA.mol": MONOMER_DIR / "Zr6.mol",
            "Zr12_AA.mol": MONOMER_DIR / "Zr12.mol",
            "BDC.mol": MONOMER_DIR / "BTC.mol",
        },
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


class DiscardedSerializedKDTree:
    """Unpickle-only placeholder for a cached SciPy spatial index."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


def _install_dependency_shims_if_needed() -> list[str]:
    """Provide only the numerical interfaces used by the archived engines."""

    installed: list[str] = []
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = importlib.import_module("numpy.core")
        for module_name in (
            "numeric",
            "multiarray",
            "umath",
            "_multiarray_umath",
        ):
            try:
                sys.modules[f"numpy._core.{module_name}"] = importlib.import_module(
                    f"numpy.core.{module_name}"
                )
            except ModuleNotFoundError:
                continue
        installed.append("NumPy-2 pickle module aliases for NumPy-1 runtimes")
    if "scipy" not in sys.modules and importlib.util.find_spec("scipy") is None:
        scipy_module = types.ModuleType("scipy")
        spatial_module = types.ModuleType("scipy.spatial")
        linalg_module = types.ModuleType("scipy.linalg")

        spatial_module.cKDTree = CompatibleKDTree
        spatial_module._mol_growth_compatibility_shim = True
        linalg_module.svd = np.linalg.svd
        scipy_module.spatial = spatial_module
        scipy_module.linalg = linalg_module
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.spatial"] = spatial_module
        sys.modules["scipy.linalg"] = linalg_module
        installed.append("numpy-compatible scipy.spatial.cKDTree and scipy.linalg.svd shim")

    spatial_module = sys.modules.get("scipy.spatial")
    if getattr(spatial_module, "_mol_growth_compatibility_shim", False):
        if not hasattr(spatial_module, "__path__"):
            spatial_module.__path__ = []
        if "scipy.spatial._ckdtree" not in sys.modules:
            ckdtree_module = types.ModuleType("scipy.spatial._ckdtree")
            ckdtree_module.cKDTree = DiscardedSerializedKDTree
            sys.modules["scipy.spatial._ckdtree"] = ckdtree_module
            installed.append("discard-only serialized SciPy cKDTree cache shim")

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
            if candidate.as_posix().startswith(
                "/public/home/yibinjiang/workspace/mol/"
            ) and candidate.name in spec_record["monomer_files"]:
                replacement = Path(spec_record["monomer_files"][candidate.name])
                if not replacement.is_file():
                    raise FileNotFoundError(
                        f"Frozen review-bundle monomer is missing: {replacement}"
                    )
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


def entity_degree_record(assembly: Any) -> dict[str, Any]:
    """Return entity-graph degrees without imposing a topology classifier."""

    by_type: dict[str, list[int]] = {}
    all_degrees: list[int] = []
    active_entities = set(assembly.entities.items)
    for entity in assembly.entities:
        degree = sum(
            neighbor in active_entities
            for neighbor in getattr(entity, "connected_entities", ())
        )
        label = getattr(entity, "entity_type", type(entity).__name__)
        by_type.setdefault(label, []).append(degree)
        all_degrees.append(degree)

    def summarize(values: list[int]) -> dict[str, Any]:
        histogram: dict[str, int] = {}
        for value in values:
            histogram[str(value)] = histogram.get(str(value), 0) + 1
        return {
            "count": len(values),
            "mean_degree": sum(values) / len(values) if values else None,
            "minimum_degree": min(values) if values else None,
            "maximum_degree": max(values) if values else None,
            "degree_histogram": histogram,
        }

    return {
        "all_entities": summarize(all_degrees),
        "by_entity_type": {
            label: summarize(values) for label, values in sorted(by_type.items())
        },
    }


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
    structure_metric_milestones: tuple[int, ...] | None = None,
    mol2_output_path: Path | None = None,
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
    requested_structure_milestones = tuple(
        sorted(
            {
                int(value)
                for value in (structure_metric_milestones or ())
                if int(value) >= 3 and int(value) <= target_entities
            }
        )
    )

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
    structure_evolution: list[dict[str, Any]] = []
    pending_structure_milestones = list(requested_structure_milestones)

    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        before = assembly_state(assembly)
        try:
            selection = select_event(assembly, coordinates, rng)
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
            assembly.grow_one_step(selection.selected_site)
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
        if system == "btb_mol":
            while (
                pending_structure_milestones
                and after["entities"] >= pending_structure_milestones[0]
            ):
                milestone = pending_structure_milestones.pop(0)
                structure_evolution.append(
                    {
                        "requested_entity_milestone": milestone,
                        "observed_entities": after["entities"],
                        "event": event_index,
                        "model_time": model_time,
                        "metrics": analyze_btb_assembly(assembly),
                    }
                )
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
    final_state["entity_graph_degrees"] = entity_degree_record(assembly)
    final_state["is_single_connected_component"] = (
        final_state["connected_component_sizes"] == [final_state["entities"]]
    )
    if system == "btb_mol":
        final_state["post_growth_dimensionality"] = analyze_btb_assembly(assembly)
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
        "topology_mode": "unrestricted",
        "bond_opening_model": "one equal per-bond opening weight",
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
            "post_growth_structure_evolution": structure_evolution,
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
            "btb_dimensionality": (
                "post-growth measurement from unrestricted all-12-site M6 growth"
                if system == "btb_mol"
                else "not applicable"
            ),
            "fragment_removal": (
                "archived component-removal routine with exact search depth "
                "at least target_entities + 1"
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "mol2_output": mol2_record,
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
        "--structure-metric-milestones",
        help="Comma-separated entity counts for unrestricted BTB post-growth measurements",
    )
    parser.add_argument("--mol2-output", type=Path)
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
            structure_metric_milestones=(
                tuple(int(value) for value in args.structure_metric_milestones.split(","))
                if args.structure_metric_milestones
                else None
            ),
            mol2_output_path=args.mol2_output,
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
