"""Generate auditable connected molecular seeds for propagation calculations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random
import time

import numpy as np

from common_kmc_selector import PhysicalCoordinates, gillespie_waiting_time, ideal_node_fraction, select_event
from common_molecular_runner import ENGINE_SPECS, _load_engine, assembly_state, connected_component_sizes, sha256
from planar_mol_topology import (
    attach_layer_plane,
    grow_one_step_planar,
    infer_six_connected_m6_plane,
    planarity_metrics,
    select_planar_event,
)


def planarity_record(assembly) -> dict[str, float]:
    centers = np.asarray([entity.center for entity in assembly.entities], dtype=float)
    centered = centers - centers.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False) if len(centers) > 1 else np.zeros(3)
    total = float(np.sum(singular))
    return {
        "center_singular_value_1": float(singular[0]),
        "center_singular_value_2": float(singular[1]),
        "center_singular_value_3": float(singular[2]),
        "planarity_ratio_s3_over_sum": float(singular[2] / total) if total > 0 else 0.0,
    }


def generate_one(
    system: str,
    seed: int,
    target_entities: int,
    wall_time_cap_s: float,
    m6_node_fraction: float = 1.0,
    topology_mode: str = "unrestricted",
):
    if not 0.0 <= m6_node_fraction <= 1.0:
        raise ValueError("m6_node_fraction must lie in [0, 1]")
    if topology_mode not in {"unrestricted", "planar_mol"}:
        raise ValueError("topology_mode must be unrestricted or planar_mol")
    if topology_mode == "planar_mol" and system != "btb_mol":
        raise ValueError("planar_mol topology is defined only for btb_mol")
    if topology_mode == "planar_mol" and m6_node_fraction != 1.0:
        raise ValueError("the current planar_mol construction requires pure M6 nodes")
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    rng = random.Random(seed ^ 0x51EED)
    engine, engine_path, redirects = _load_engine(system)
    engine.MAX_DEPTH = target_entities + 1
    q = 1.0 if system == "uio66" else 3.0
    coordinates = PhysicalCoordinates(
        external_activity=1.0,
        node_fraction=ideal_node_fraction(q),
        off_weight=0.1,
        closure_entropy_coefficient=0.5,
    )
    m6_class = "Zr6_AA" if system == "uio66" else "Zr6"
    m12_class = "Zr12_AA" if system == "uio66" else "Zr12"
    initial_class = m6_class if rng.random() < m6_node_fraction else m12_class
    initial_entity = getattr(engine, initial_class)()
    assembly = engine.Assembly(initial_entity, m6_node_fraction, 0.0, 2.0)
    plane = None
    if topology_mode == "planar_mol":
        plane = infer_six_connected_m6_plane(initial_entity)
        attach_layer_plane(assembly, plane)
    model_time = 0.0
    event_counts: dict[str, int] = {}
    for event_index in range(1, 200_001):
        if time.perf_counter() - started > wall_time_cap_s:
            raise TimeoutError(f"Seed {seed} did not reach {target_entities} entities")
        selected = (
            select_planar_event(assembly, coordinates, plane, rng)
            if plane is not None
            else select_event(assembly, coordinates, rng)
        )
        model_time += gillespie_waiting_time(selected.propensities.total, rng)
        if selected.event in {"add_node", "add_linker"}:
            before = len(assembly.entities)
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
            label = f"{selected.event}_{'accepted' if len(assembly.entities) > before else 'collision_rejected'}"
        elif selected.event == "close_internal":
            assembly.link_internal_carboxylate(selected.selected_pair)
            label = "close_internal"
        else:
            assembly.remove_linkage(selected.selected_pair)
            label = "remove_bond"
        event_counts[label] = event_counts.get(label, 0) + 1
        if len(assembly.entities) >= target_entities:
            break
    else:
        raise RuntimeError("Seed generation reached the event cap")
    components = connected_component_sizes(assembly)
    if components != [len(assembly.entities)]:
        raise RuntimeError(f"Generated seed is disconnected: {components}")
    state = assembly_state(assembly)
    return assembly, {
        "schema_version": "common-molecular-seed-v1",
        "system": system,
        "topology_mode": topology_mode,
        "random_seed": seed,
        "generation_coordinates": {
            "atomic_metal_to_linker_ratio": q,
            "external_activity": 1.0,
            "node_activity_share": ideal_node_fraction(q),
            "bond_removal_weight": 0.1,
            "closure_coefficient": 0.5,
            "m6_node_fraction": m6_node_fraction,
            "m12_node_fraction": 1.0 - m6_node_fraction,
            "initial_node_class": initial_class,
        },
        "generation_model_time": model_time,
        "event_counts": event_counts,
        "state": state,
        "connected_component_sizes": components,
        "geometry": {
            **planarity_record(assembly),
            **(planarity_metrics(assembly, plane) if plane is not None else {}),
        },
        "provenance": {
            "engine_path": engine_path.as_posix(),
            "engine_sha256": sha256(engine_path),
            "engine_import_redirects": redirects,
            "generator_sha256": sha256(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("uio66", "btb_mol"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--target-entities", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260831)
    parser.add_argument("--wall-time-cap-s", type=float, default=60.0)
    parser.add_argument("--m6-node-fraction", type=float, default=1.0)
    parser.add_argument(
        "--topology-mode", choices=("unrestricted", "planar_mol"), default="unrestricted"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_records = []
    for index in range(args.count):
        random_seed = args.base_seed + index * 7919
        assembly, record = generate_one(
            args.system,
            random_seed,
            args.target_entities,
            args.wall_time_cap_s,
            args.m6_node_fraction,
            args.topology_mode,
        )
        seed_id = f"{args.system}_seed_{index:02d}"
        pickle_path = args.output_dir / f"{seed_id}.pkl"
        temporary_pickle_path = args.output_dir / f".{seed_id}.pkl.tmp"
        json_path = args.output_dir / f"{seed_id}.json"
        with temporary_pickle_path.open("wb") as handle:
            pickle.dump(assembly, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary_pickle_path.open("rb") as handle:
            round_trip_assembly = pickle.load(handle)
        round_trip_components = connected_component_sizes(round_trip_assembly)
        if round_trip_components != [len(round_trip_assembly.entities)]:
            raise RuntimeError(
                f"Serialized seed failed connected round trip: {round_trip_components}"
            )
        temporary_pickle_path.replace(pickle_path)
        record["seed_id"] = seed_id
        record["pickle_sha256"] = sha256(pickle_path)
        record["pickle_round_trip_validation"] = "passed"
        json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        index_records.append(record)
    (args.output_dir / f"{args.system}_seed_index.json").write_text(
        json.dumps(index_records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"system": args.system, "seed_count": len(index_records)}, indent=2))


if __name__ == "__main__":
    main()
