"""Population KMC for molecular seeds sharing finite node/linker reservoirs.

The fixed-bath runners follow one assembly.  This module follows a population
of independent molecular assemblies while all of them deplete the same
solution reservoir.  It is intended for the seeded digestion-NMR experiments,
not for a volumetric primary-nucleation claim.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import random
import sys
import time
import types
from typing import Any, Iterable

import numpy as np

from chemistry_mapping import CompetitiveBath, external_and_removal_coordinates
from btb_postgrowth_structure import analyze_btb_assembly
from common_kmc_selector import (
    PhysicalCoordinates,
    compute_propensities,
    counts_from_assembly,
    gillespie_waiting_time,
    select_event,
)
from common_molecular_runner import (
    _install_dependency_shims_if_needed,
    _load_engine,
    connected_component_sizes,
    sha256,
)


class _DiscardedSerializedKDTree:
    """Unpickle-only placeholder for a cached SciPy cKDTree.

    The spatial index is cleared immediately after deserialization and rebuilt
    by the active geometry engine.  Its serialized implementation is therefore
    not part of the reviewed molecular state.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


def _install_numpy_pickle_compatibility() -> None:
    """Allow NumPy-2 pickles to load under the NumPy-1 runtime on 5090.

    NumPy 2 serializes selected array helpers below ``numpy._core`` whereas
    NumPy 1.26 exposes the same implementations below ``numpy.core``.  The
    reviewed seed files were created with NumPy 2.  Installing module aliases
    preserves their bytes and molecular contents; it does not alter any KMC
    probability, geometry, or thermodynamic rule.
    """

    if "numpy._core" in sys.modules:
        return
    core = importlib.import_module("numpy.core")
    sys.modules["numpy._core"] = core
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


def _install_spatial_pickle_compatibility() -> None:
    """Load SciPy-free shims and discard serialized cKDTree caches safely."""

    _install_dependency_shims_if_needed()
    spatial = sys.modules.get("scipy.spatial")
    if spatial is not None and not hasattr(spatial, "__path__"):
        spatial.__path__ = []
    module_name = "scipy.spatial._ckdtree"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.cKDTree = _DiscardedSerializedKDTree
        sys.modules[module_name] = module


@dataclass(frozen=True)
class FiniteBathInputs:
    volume_mL: float
    seed_reference_linker_umol: float
    analytical_atomic_metal_mM: float
    free_linker_mM: float
    reactive_m6_fraction: float
    node_capture_coefficient_per_mM: float
    tethered_linker_weight: float
    opened_site_rate_ratio: float = 1.0
    bond_opening_rate_ratio: float = 1.0
    solvent_displacement_efficiency: float = 1.0
    closure_coefficient: float = 0.5
    free_linker_exchange_fate: str = "topology_loss"

    def validate(self) -> None:
        nonnegative = (
            "analytical_atomic_metal_mM",
            "free_linker_mM",
            "reactive_m6_fraction",
            "node_capture_coefficient_per_mM",
            "tethered_linker_weight",
            "opened_site_rate_ratio",
            "bond_opening_rate_ratio",
            "solvent_displacement_efficiency",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.volume_mL) or self.volume_mL <= 0.0:
            raise ValueError("volume_mL must be finite and positive")
        if (
            not math.isfinite(self.seed_reference_linker_umol)
            or self.seed_reference_linker_umol <= 0.0
        ):
            raise ValueError("seed_reference_linker_umol must be finite and positive")
        if self.reactive_m6_fraction > 1.0:
            raise ValueError("reactive_m6_fraction must lie in [0, 1]")
        if not math.isfinite(self.closure_coefficient):
            raise ValueError("closure_coefficient must be finite")
        if self.free_linker_exchange_fate not in {
            "topology_loss",
            "protective_replacement",
        }:
            raise ValueError("unknown free-linker exchange fate")

    @property
    def initial_reactive_m6_mM(self) -> float:
        return self.analytical_atomic_metal_mM * self.reactive_m6_fraction / 6.0


@dataclass
class Reservoir:
    node_initial: int
    linker_initial: int
    node_remaining: int
    linker_remaining: int
    node_expected: float
    linker_expected: float

    def validate(self) -> None:
        for name in (
            "node_initial",
            "linker_initial",
            "node_remaining",
            "linker_remaining",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.node_remaining > self.node_initial:
            raise ValueError("node reservoir exceeds its initial inventory")
        if self.linker_remaining > self.linker_initial:
            raise ValueError("linker reservoir exceeds its initial inventory")


def entity_role(system: str, entity: Any) -> str:
    name = type(entity).__name__.lower()
    if "zr6" in name or "zr12" in name or "hf6" in name or "hf12" in name:
        return "node"
    if system == "uio66" and "bdc" in name:
        return "linker"
    if system == "btb_mol" and "btb" in name:
        return "linker"
    raise ValueError(f"Cannot classify {type(entity).__name__!r} in {system}")


def stochastic_round(value: float, rng: random.Random) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("inventory expectation must be finite and non-negative")
    lower = math.floor(value)
    return lower + int(rng.random() < value - lower)


def reservoir_from_experiment(
    inputs: FiniteBathInputs,
    *,
    total_model_seed_linkers: int,
    rng: random.Random,
) -> Reservoir:
    inputs.validate()
    if total_model_seed_linkers <= 0:
        raise ValueError("model seed ensemble must contain at least one linker")
    reactive_node_umol = inputs.initial_reactive_m6_mM * inputs.volume_mL
    incoming_linker_umol = inputs.free_linker_mM * inputs.volume_mL
    node_expected = (
        reactive_node_umol
        / inputs.seed_reference_linker_umol
        * total_model_seed_linkers
    )
    linker_expected = (
        incoming_linker_umol
        / inputs.seed_reference_linker_umol
        * total_model_seed_linkers
    )
    node_count = stochastic_round(node_expected, rng)
    linker_count = stochastic_round(linker_expected, rng)
    reservoir = Reservoir(
        node_initial=node_count,
        linker_initial=linker_count,
        node_remaining=node_count,
        linker_remaining=linker_count,
        node_expected=node_expected,
        linker_expected=linker_expected,
    )
    reservoir.validate()
    return reservoir


def dynamic_coordinates(
    base_bath: CompetitiveBath,
    inputs: FiniteBathInputs,
    reservoir: Reservoir,
) -> tuple[PhysicalCoordinates, dict[str, float]]:
    inputs.validate()
    reservoir.validate()
    linker_fraction = (
        reservoir.linker_remaining / reservoir.linker_initial
        if reservoir.linker_initial > 0
        else 0.0
    )
    node_fraction = (
        reservoir.node_remaining / reservoir.node_initial
        if reservoir.node_initial > 0
        else 0.0
    )
    linker_concentration = inputs.free_linker_mM * linker_fraction
    reactive_node_concentration = inputs.initial_reactive_m6_mM * node_fraction
    bath = replace(
        base_bath,
        linker=replace(base_bath.linker, total_concentration=linker_concentration),
    )
    mapped = dict(
        external_and_removal_coordinates(
            bath,
            reactive_node_activity=reactive_node_concentration,
            node_capture_coefficient=inputs.node_capture_coefficient_per_mM,
            tethered_linker_weight=inputs.tethered_linker_weight,
            opened_site_rate_ratio=inputs.opened_site_rate_ratio,
            bond_opening_rate_ratio=inputs.bond_opening_rate_ratio,
            solvent_displacement_efficiency=inputs.solvent_displacement_efficiency,
            free_linker_exchange_fate=inputs.free_linker_exchange_fate,
        )
    )
    if reservoir.node_remaining <= 0:
        mapped["node_activity"] = 0.0
    if reservoir.linker_remaining <= 0:
        mapped["linker_activity"] = 0.0
    external = mapped["node_activity"] + mapped["linker_activity"]
    share = mapped["node_activity"] / external if external > 0.0 else 0.0
    coordinates = PhysicalCoordinates(
        external_activity=external,
        node_fraction=share,
        off_weight=mapped["off_weight"],
        closure_entropy_coefficient=inputs.closure_coefficient,
    )
    mapped.update(
        {
            "dynamic_linker_concentration_mM": linker_concentration,
            "dynamic_reactive_m6_concentration_mM": reactive_node_concentration,
            "dynamic_external_activity": external,
            "dynamic_node_activity_share": share,
        }
    )
    return coordinates, mapped


def structural_fingerprint(assembly: Any) -> str:
    payload = {
        "types": [type(entity).__name__ for entity in assembly.entities.items],
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_seed_population(
    system: str,
    seed_pickles: Iterable[Path],
    *,
    batch_seed_count: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    paths = [Path(path) for path in seed_pickles]
    if not paths:
        raise ValueError("At least one frozen seed pickle is required")
    _install_numpy_pickle_compatibility()
    _install_spatial_pickle_compatibility()
    templates: list[Any] = []
    source_records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as handle:
            assembly = pickle.load(handle)
        # A frozen seed may have been serialized on a host using the NumPy
        # cKDTree compatibility class and resumed on a host with SciPy (or the
        # reverse).  Spatial trees are caches, not molecular state; clear them
        # so the active geometry engine rebuilds one consistent implementation.
        for entity in assembly.entities.items:
            if hasattr(entity, "kdtree"):
                entity.kdtree = None
        components = connected_component_sizes(assembly)
        if components != [len(assembly.entities)]:
            raise ValueError(f"Disconnected frozen seed {path}: {components}")
        for entity in assembly.entities.items:
            entity_role(system, entity)
        templates.append(assembly)
        source_records.append(
            {
                "path": path.resolve().as_posix(),
                "sha256": sha256(path),
                "fingerprint": structural_fingerprint(assembly),
            }
        )
    population = [copy.deepcopy(templates[i % len(templates)]) for i in range(batch_seed_count)]
    return population, source_records


def _weighted_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("cannot select from zero aggregate propensity")
    draw = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(weights) - 1


def assembly_total_propensity(
    assembly: Any,
    coordinates: PhysicalCoordinates,
) -> float:
    """Use the unrestricted, equal-per-bond opportunity set for the clock."""

    return compute_propensities(counts_from_assembly(assembly), coordinates).total


def run_population_trace(
    *,
    system: str,
    seed_pickles: Iterable[Path],
    batch_seed_count: int,
    base_bath: CompetitiveBath,
    inputs: FiniteBathInputs,
    random_seed: int,
    model_time_horizon: float,
    event_cap: int = 500_000,
    wall_time_cap_s: float = 300.0,
) -> dict[str, Any]:
    if system not in {"uio66", "btb_mol"}:
        raise ValueError("Unknown molecular system")
    if batch_seed_count <= 0:
        raise ValueError("batch_seed_count must be positive")
    if not math.isfinite(model_time_horizon) or model_time_horizon <= 0.0:
        raise ValueError("model_time_horizon must be finite and positive")
    inputs.validate()
    base_bath.validate()
    started = time.perf_counter()
    random.seed(random_seed)
    np.random.seed(random_seed % (2**32 - 1))
    rng = random.Random(random_seed ^ 0xF1B17E)
    engine, engine_path, redirects = _load_engine(system)
    engine.MAX_DEPTH = max(64, event_cap + 1)
    assemblies, seed_sources = _load_seed_population(
        system, seed_pickles, batch_seed_count=batch_seed_count
    )
    initial_entity_sets = [set(assembly.entities.items) for assembly in assemblies]
    initial_linker_sets = [
        {entity for entity in assembly.entities.items if entity_role(system, entity) == "linker"}
        for assembly in assemblies
    ]
    total_model_seed_linkers = sum(len(items) for items in initial_linker_sets)
    reservoir = reservoir_from_experiment(
        inputs, total_model_seed_linkers=total_model_seed_linkers, rng=rng
    )
    event_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    model_time = 0.0
    termination = "event_cap"

    def population_record(event: int, mapping: dict[str, float]) -> dict[str, Any]:
        retained_reference = []
        new_linkers = []
        sizes = []
        for assembly, initial_entities, initial_linkers in zip(
            assemblies, initial_entity_sets, initial_linker_sets
        ):
            current = set(assembly.entities.items)
            retained_reference.append(len(current.intersection(initial_linkers)))
            new_linkers.append(
                sum(
                    entity_role(system, entity) == "linker"
                    and entity not in initial_entities
                    for entity in current
                )
            )
            sizes.append(len(current))
        reference_total = sum(retained_reference)
        new_total = sum(new_linkers)
        return {
            "event": event,
            "model_time": model_time,
            "node_remaining": reservoir.node_remaining,
            "linker_remaining": reservoir.linker_remaining,
            "reference_seed_linkers_retained": reference_total,
            "new_linkers_in_solids": new_total,
            "new_to_retained_reference_linker_ratio": (
                new_total / reference_total if reference_total > 0 else None
            ),
            "seed_count_with_net_entity_growth": sum(
                size > len(initial) for size, initial in zip(sizes, initial_entity_sets)
            ),
            "mean_entities_per_seed": float(np.mean(sizes)),
            "median_entities_per_seed": float(np.median(sizes)),
            "maximum_entities": max(sizes),
            "minimum_entities": min(sizes),
            **{
                key: mapping[key]
                for key in (
                    "dynamic_linker_concentration_mM",
                    "dynamic_reactive_m6_concentration_mM",
                    "dynamic_external_activity",
                    "dynamic_node_activity_share",
                    "off_weight",
                    "normalized_off_ratio",
                )
                if key in mapping
            },
        }

    coordinates, mapping = dynamic_coordinates(base_bath, inputs, reservoir)
    samples.append(population_record(0, mapping))
    sample_events = set(
        int(value)
        for value in np.unique(
            np.geomspace(1, max(2, event_cap), num=80, dtype=int)
        )
    )
    executed_events = 0
    for event_index in range(1, event_cap + 1):
        if time.perf_counter() - started >= wall_time_cap_s:
            termination = "wall_time_cap"
            break
        coordinates, mapping = dynamic_coordinates(base_bath, inputs, reservoir)
        totals = [assembly_total_propensity(assembly, coordinates) for assembly in assemblies]
        aggregate = sum(totals)
        if aggregate <= 0.0:
            termination = "no_enabled_event"
            break
        delta_time = gillespie_waiting_time(aggregate, rng)
        if model_time + delta_time > model_time_horizon:
            model_time = model_time_horizon
            termination = "model_time_horizon"
            break
        model_time += delta_time
        selected_index = _weighted_index(totals, rng)
        assembly = assemblies[selected_index]
        selection = select_event(assembly, coordinates, rng)
        before_entities = set(assembly.entities.items)
        before_roles = Counter(entity_role(system, entity) for entity in before_entities)
        if selection.event in {"add_node", "add_linker"}:
            assembly.grow_one_step(selection.selected_site)
            accepted = len(assembly.entities) > len(before_entities)
            event_counts[f"{selection.event}_attempt"] += 1
            event_counts[f"{selection.event}_{'accepted' if accepted else 'collision_rejected'}"] += 1
            if accepted:
                if selection.event == "add_node":
                    if reservoir.node_remaining <= 0:
                        raise RuntimeError("accepted node addition with empty node reservoir")
                    reservoir.node_remaining -= 1
                else:
                    if reservoir.linker_remaining <= 0:
                        raise RuntimeError("accepted linker addition with empty linker reservoir")
                    reservoir.linker_remaining -= 1
        elif selection.event == "close_internal":
            assembly.link_internal_carboxylate(selection.selected_pair)
            event_counts["close_internal"] += 1
        else:
            assembly.remove_linkage(selection.selected_pair)
            event_counts["remove_bond"] += 1
        after_entities = set(assembly.entities.items)
        after_roles = Counter(entity_role(system, entity) for entity in after_entities)
        # Detached smaller fragments remain deleted and are not returned to the
        # active reservoir. This avoids treating material differently according
        # to whether it originated in the seed or was added during the trace.
        reservoir.validate()
        if any(value < 0 for value in after_roles.values()) or any(
            value < 0 for value in before_roles.values()
        ):
            raise RuntimeError("negative molecular role count")
        executed_events = event_index
        if event_index in sample_events:
            _, post_event_mapping = dynamic_coordinates(base_bath, inputs, reservoir)
            samples.append(population_record(event_index, post_event_mapping))
    else:
        termination = "event_cap"

    coordinates, mapping = dynamic_coordinates(base_bath, inputs, reservoir)
    if samples[-1]["event"] != executed_events or samples[-1]["model_time"] != model_time:
        samples.append(population_record(executed_events, mapping))
    final_components = [connected_component_sizes(assembly) for assembly in assemblies]
    if any(component != [len(assembly.entities)] for component, assembly in zip(final_components, assemblies)):
        raise RuntimeError("population contains a disconnected tracked assembly")
    final_record = samples[-1]
    per_seed_final = []
    for assembly, initial_entities, initial_linkers in zip(
        assemblies, initial_entity_sets, initial_linker_sets
    ):
        current = set(assembly.entities.items)
        retained_reference = len(current.intersection(initial_linkers))
        new_linkers = sum(
            entity_role(system, entity) == "linker" and entity not in initial_entities
            for entity in current
        )
        per_seed_final.append(
            {
                "initial_entities": len(initial_entities),
                "final_entities": len(current),
                "retained_reference_linkers": retained_reference,
                "new_linkers": new_linkers,
                "new_linkers_per_initial_reference_linker": (
                    new_linkers / len(initial_linkers) if initial_linkers else None
                ),
                "new_to_retained_reference_linker_ratio": (
                    new_linkers / retained_reference if retained_reference else None
                ),
                "btb_unrestricted_dimensionality": (
                    analyze_btb_assembly(assembly) if system == "btb_mol" else None
                ),
            }
        )
    return {
        "schema_version": "finite-bath-population-unrestricted-v2",
        "system": system,
        "topology_mode": "unrestricted",
        "bond_accessibility_mode": "all_bonds_equal_per_bond_weight",
        "detached_fragment_fate": "deleted_without_recycling_to_active_reservoir",
        "random_seed": random_seed,
        "termination": {
            "status": termination,
            "executed_events": executed_events,
            "model_time": model_time,
            "model_time_horizon": model_time_horizon,
            "wall_seconds": time.perf_counter() - started,
        },
        "experimental_inputs": {
            **inputs.__dict__,
            "initial_reactive_m6_mM": inputs.initial_reactive_m6_mM,
            "base_bath": {
                "linker": base_bath.linker.__dict__,
                "modulator": base_bath.modulator.__dict__,
                "solvent_weight": base_bath.solvent_weight,
                "operational_ph": base_bath.operational_ph,
                "formic_acid_pka": base_bath.formic_acid_pka,
                "acid_to_carboxylate_solvent_exchange_ratio": (
                    base_bath.acid_to_carboxylate_solvent_exchange_ratio
                ),
            },
        },
        "population": {
            "batch_seed_count": batch_seed_count,
            "seed_sources": seed_sources,
            "unique_seed_fingerprints": sorted({item["fingerprint"] for item in seed_sources}),
            "initial_model_seed_linkers": total_model_seed_linkers,
            "final_connected_component_sizes": final_components,
            "per_seed_final": per_seed_final,
        },
        "reservoir": reservoir.__dict__,
        "event_counts": dict(event_counts),
        "final": final_record,
        "samples": samples,
        "provenance": {
            "engine_path": engine_path.as_posix(),
            "engine_sha256": sha256(engine_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "dependency_adaptations": redirects,
        },
        "claim_boundary": (
            "This is a finite shared-bath seeded-propagation calculation. "
            "It does not contain homogeneous primary nucleation or a calibrated "
            "hot-solvent reactive-M6 fraction. BTB growth is unrestricted; "
            "dimensionality is measured only after growth."
        ),
    }
