import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from UiO66_Assembly_Large_Correction_conc import Assembly, Zr6_AA, safe_pickle_save


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a large mixed seed with internally embedded, under-coordinated Zr12 clusters, "
            "then grow the outer shell using only Zr6_AA and BDC."
        )
    )
    parser.add_argument("--rng-seed", type=int, default=31)
    parser.add_argument("--target-entities", type=int, default=180)
    parser.add_argument("--attempts-per-step", type=int, default=500)
    parser.add_argument("--internal-link-probability", type=float, default=0.10)
    parser.add_argument(
        "--internal-zr12-count",
        type=int,
        default=2,
        help="Number of internally embedded Zr12_AA clusters to build into the seed core.",
    )
    parser.add_argument(
        "--seed-zr6-branches",
        type=int,
        default=4,
        help="Number of early Zr6_AA branches grown from the initial seed to avoid chain-like growth.",
    )
    parser.add_argument(
        "--zr6-branches-per-zr12",
        type=int,
        default=3,
        help="Number of Zr6_AA branches sprouted from each internal Zr12_AA cluster.",
    )
    parser.add_argument("--max-zr12-coordination", type=int, default=3)
    parser.add_argument(
        "--max-axis-ratio-1-3",
        type=float,
        default=None,
        help="Optional upper bound on principal_axis_ratio_1_3 for compact-seed screening.",
    )
    parser.add_argument("--output-dir", default="output/mixed_nuclei")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def entity_kind(entity):
    if getattr(entity, "entity_type", None) == "Ligand":
        return "BDC"
    if getattr(entity, "entity_type", None) == "Zr" and getattr(entity, "entity_subtype", None) == 0:
        return "Zr6_AA"
    if getattr(entity, "entity_type", None) == "Zr" and getattr(entity, "entity_subtype", None) == 1:
        return "Zr12_AA"
    return "Unknown"


def cluster_capacity(entity):
    kind = entity_kind(entity)
    if kind == "Zr6_AA":
        return 12
    if kind == "Zr12_AA":
        return 18
    return None


def zr_entities(assembly):
    return [entity for entity in assembly.entities if entity_kind(entity).startswith("Zr")]


def cluster_centroid(assembly):
    centers = [np.asarray(entity.center, dtype=float) for entity in zr_entities(assembly)]
    return np.mean(np.asarray(centers), axis=0)


def octant_key(vector):
    return tuple(1 if value >= 0 else -1 for value in vector)


def cluster_octant_populations(assembly):
    center = cluster_centroid(assembly)
    counts = {}
    for entity in zr_entities(assembly):
        vector = np.asarray(entity.center, dtype=float) - center
        if np.linalg.norm(vector) < 1e-8:
            continue
        key = octant_key(vector)
        counts[key] = counts.get(key, 0) + 1
    return center, counts


def cluster_shape_metrics(assembly):
    centers = np.asarray([np.asarray(entity.center, dtype=float) for entity in zr_entities(assembly)])
    if len(centers) < 3:
        return None

    centered = centers - np.mean(centers, axis=0)
    covariance = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    spans = centers.max(axis=0) - centers.min(axis=0)
    safe_eigs = np.maximum(eigenvalues, 1e-8)
    safe_spans = np.maximum(spans, 1e-8)

    return {
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
        "principal_axis_ratio_1_2": float(safe_eigs[0] / safe_eigs[1]),
        "principal_axis_ratio_2_3": float(safe_eigs[1] / safe_eigs[2]),
        "principal_axis_ratio_1_3": float(safe_eigs[0] / safe_eigs[2]),
        "span_xyz": [float(value) for value in spans],
        "span_ratio_max_min": float(np.max(safe_spans) / np.min(safe_spans)),
    }


def assembly_center(assembly):
    centers = [np.asarray(entity.center, dtype=float) for entity in assembly.entities]
    return np.mean(np.asarray(centers), axis=0)


def owner_radius(selected_carboxylate, center):
    owner_center = np.asarray(selected_carboxylate.belonging_entity.center, dtype=float)
    return float(np.linalg.norm(owner_center - center))


def linked_coordination_map(assembly):
    zr_entities = [entity for entity in assembly.entities if entity_kind(entity).startswith("Zr")]
    counts = {id(entity): 0 for entity in zr_entities}
    for pair in assembly.linked_carboxylate_pairs:
        for carb in pair:
            entity = getattr(carb, "belonging_entity", None)
            if entity is not None and id(entity) in counts:
                counts[id(entity)] += 1
    return counts


def cluster_summary(assembly):
    zr_entities = [entity for entity in assembly.entities if entity_kind(entity).startswith("Zr")]
    center = np.mean(np.asarray([np.asarray(entity.center, dtype=float) for entity in zr_entities]), axis=0)
    linked_counts = linked_coordination_map(assembly)

    rows = []
    for idx, entity in enumerate(zr_entities):
        kind = entity_kind(entity)
        radius = float(np.linalg.norm(np.asarray(entity.center, dtype=float) - center))
        capacity = cluster_capacity(entity)
        coord = linked_counts[id(entity)]
        rows.append(
            {
                "cluster_id": idx,
                "kind": kind,
                "coordination": coord,
                "max_sites": capacity,
                "free_sites": capacity - coord,
                "radius_from_cluster_centroid": radius,
                "center": np.asarray(entity.center, dtype=float).round(4).tolist(),
                "entity_ref": entity,
            }
        )

    rows.sort(key=lambda row: row["radius_from_cluster_centroid"])
    n = len(rows)
    for rank, row in enumerate(rows):
        row["radial_rank_inner_to_outer"] = rank
        row["radial_fraction"] = 0.0 if n == 1 else rank / (n - 1)
    return rows


def free_sites(
    assembly,
    *,
    carboxylate_type,
    owner_entity=None,
    owner_kind_filter=None,
    owner_connected_to=None,
    owner_connected_kind=None,
):
    out = []
    for carb in assembly.free_carboxylates:
        if carb.carboxylate_type != carboxylate_type:
            continue
        owner = carb.belonging_entity
        if owner is None:
            continue
        if owner_entity is not None and owner is not owner_entity:
            continue
        if owner_kind_filter is not None and entity_kind(owner) != owner_kind_filter:
            continue
        if owner_connected_to is not None:
            neighbors = getattr(owner, "connected_entities", [])
            if owner_connected_to not in neighbors:
                continue
        if owner_connected_kind is not None:
            neighbors = getattr(owner, "connected_entities", [])
            if not any(entity_kind(neighbor) == owner_connected_kind for neighbor in neighbors):
                continue
        out.append(carb)
    return out


def pick_site(candidates, assembly, preference):
    if not candidates:
        raise RuntimeError("No candidate growth sites matched the requested constraints.")
    if preference == "random":
        return random.choice(candidates)

    center = assembly_center(assembly)
    ranked = sorted(candidates, key=lambda carb: owner_radius(carb, center))
    if preference == "outermost":
        return ranked[-1]
    if preference == "innermost":
        return ranked[0]
    if preference == "sparse_outer":
        octant_center, octant_counts = cluster_octant_populations(assembly)
        ranked_rows = []
        for carb in candidates:
            owner_center = np.asarray(carb.belonging_entity.center, dtype=float)
            vector = owner_center - octant_center
            radius = float(np.linalg.norm(vector))
            occupancy = octant_counts.get(octant_key(vector), 0) if radius >= 1e-8 else 10**9
            ranked_rows.append((occupancy, -radius, carb))

        min_occupancy = min(row[0] for row in ranked_rows)
        sparse_rows = [row for row in ranked_rows if row[0] <= min_occupancy + 1]
        sparse_rows.sort(key=lambda row: (row[0], row[1]))
        top_n = max(1, min(8, int(np.ceil(len(sparse_rows) * 0.35))))
        return random.choice([row[2] for row in sparse_rows[:top_n]])
    raise ValueError(f"Unsupported preference: {preference}")


def grow_on_site(assembly, selected_carboxylate, next_kind):
    before_ids = {id(entity) for entity in assembly.entities}
    before_n = len(assembly.entities)
    original_ratio = assembly.ZR6_PERCENTAGE
    if next_kind == "zr6":
        assembly.ZR6_PERCENTAGE = 1.0
    elif next_kind == "zr12":
        assembly.ZR6_PERCENTAGE = 0.0

    try:
        assembly.grow_one_step(selected_carboxylate)
    finally:
        assembly.ZR6_PERCENTAGE = original_ratio

    if len(assembly.entities) != before_n + 1:
        return None

    new_entities = [entity for entity in assembly.entities if id(entity) not in before_ids]
    if len(new_entities) != 1:
        raise RuntimeError(f"Expected exactly one new entity, found {len(new_entities)}.")
    return new_entities[0]


def grow_with_retries(
    assembly,
    *,
    next_kind,
    attempts_per_step,
    carboxylate_type,
    owner_entity=None,
    owner_kind_filter=None,
    owner_connected_to=None,
    owner_connected_kind=None,
    preference="random",
):
    for _ in range(attempts_per_step):
        candidates = free_sites(
            assembly,
            carboxylate_type=carboxylate_type,
            owner_entity=owner_entity,
            owner_kind_filter=owner_kind_filter,
            owner_connected_to=owner_connected_to,
            owner_connected_kind=owner_connected_kind,
        )
        selected = pick_site(candidates, assembly, preference)
        new_entity = grow_on_site(assembly, selected, next_kind)
        if new_entity is not None:
            return new_entity
    raise RuntimeError(f"Failed to place {next_kind} after {attempts_per_step} attempts.")


def append_unique_entity(entities, entity):
    if all(existing is not entity for existing in entities):
        entities.append(entity)


def grow_cluster_from_anchor(assembly, anchor_entity, cluster_kind, attempts_per_step):
    linker = grow_with_retries(
        assembly,
        next_kind="bdc",
        attempts_per_step=attempts_per_step,
        carboxylate_type="formate",
        owner_entity=anchor_entity,
        preference="random",
    )
    cluster = grow_with_retries(
        assembly,
        next_kind=cluster_kind,
        attempts_per_step=attempts_per_step,
        carboxylate_type="benzoate",
        owner_entity=linker,
        preference="random",
    )
    return linker, cluster


def pick_shell_seed_entity(shell_seed_entities, assembly, anchor_usage):
    unique_entities = []
    seen_ids = set()
    for entity in shell_seed_entities:
        entity_id = id(entity)
        if entity_id in seen_ids:
            continue
        unique_entities.append(entity)
        seen_ids.add(entity_id)

    if not unique_entities:
        raise RuntimeError("No shell-seed Zr6 entities are available for internal Zr12 placement.")

    octant_center, octant_counts = cluster_octant_populations(assembly)
    ranked = []
    for entity in unique_entities:
        vector = np.asarray(entity.center, dtype=float) - octant_center
        radius = float(np.linalg.norm(vector))
        occupancy = octant_counts.get(octant_key(vector), 0) if radius >= 1e-8 else 0
        ranked.append(
            (
                anchor_usage.get(id(entity), 0),
                occupancy,
                radius,
                random.random(),
                entity,
            )
        )

    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    top_n = max(1, min(6, int(np.ceil(len(ranked) * 0.40))))
    selected = random.choice([row[4] for row in ranked[:top_n]])
    anchor_usage[id(selected)] = anchor_usage.get(id(selected), 0) + 1
    return selected


def maybe_link_internal_pair(assembly, probability):
    if len(assembly.ready_to_connect_carboxylate_pairs) == 0:
        return False
    if random.random() >= probability:
        return False
    pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
    assembly.link_internal_carboxylate(pair)
    return True


def build_low_coordination_internal_zr12_core(
    assembly,
    attempts_per_step,
    internal_zr12_count,
    seed_zr6_branches,
    zr6_branches_per_zr12,
):
    build_log = []

    seed_zr6 = next(entity for entity in assembly.entities if entity_kind(entity) == "Zr6_AA")
    build_log.append({"step": "seed", "entity": entity_kind(seed_zr6)})

    if internal_zr12_count < 2:
        raise ValueError("internal_zr12_count must be at least 2.")
    if seed_zr6_branches < 2:
        raise ValueError("seed_zr6_branches must be at least 2.")
    if zr6_branches_per_zr12 < 1:
        raise ValueError("zr6_branches_per_zr12 must be at least 1.")

    shell_seed_entities = [seed_zr6]
    anchor_usage = {id(seed_zr6): 0}

    for branch_idx in range(seed_zr6_branches):
        _, z6_branch = grow_cluster_from_anchor(assembly, seed_zr6, "zr6", attempts_per_step)
        append_unique_entity(shell_seed_entities, z6_branch)
        anchor_usage[id(z6_branch)] = 0
        build_log.append(
            {
                "step": "seed_branch",
                "branch_index": branch_idx,
                "anchor_kind": entity_kind(seed_zr6),
                "entity": entity_kind(z6_branch),
            }
        )

    z12_core_entities = []
    for z12_idx in range(internal_zr12_count):
        anchor_entity = pick_shell_seed_entity(shell_seed_entities, assembly, anchor_usage)
        _, z12_entity = grow_cluster_from_anchor(assembly, anchor_entity, "zr12", attempts_per_step)
        z12_core_entities.append(z12_entity)
        build_log.append(
            {
                "step": "core_zr12",
                "z12_index": z12_idx,
                "anchor_kind": entity_kind(anchor_entity),
                "entity": entity_kind(z12_entity),
            }
        )

        for branch_idx in range(zr6_branches_per_zr12):
            _, z6_branch = grow_cluster_from_anchor(assembly, z12_entity, "zr6", attempts_per_step)
            append_unique_entity(shell_seed_entities, z6_branch)
            anchor_usage[id(z6_branch)] = 0
            build_log.append(
                {
                    "step": "z12_branch",
                    "z12_index": z12_idx,
                    "branch_index": branch_idx,
                    "anchor_kind": entity_kind(z12_entity),
                    "entity": entity_kind(z6_branch),
                }
            )

    return {
        "z12_core_entities": z12_core_entities,
        "shell_seed_entities": shell_seed_entities,
        "build_log": build_log,
    }


def grow_outer_shell_zr6_only(assembly, target_entities, attempts_per_step, internal_link_probability):
    snapshots = []
    while len(assembly.entities) < target_entities:
        maybe_link_internal_pair(assembly, internal_link_probability)

        z6_branch_candidates = free_sites(
            assembly,
            carboxylate_type="benzoate",
            owner_kind_filter="BDC",
            owner_connected_kind="Zr6_AA",
        )
        z6_formate_candidates = free_sites(
            assembly,
            carboxylate_type="formate",
            owner_kind_filter="Zr6_AA",
        )

        growth_attempts = []
        if z6_branch_candidates and z6_formate_candidates:
            if random.random() < 0.35:
                growth_attempts = ["zr6", "bdc"]
            else:
                growth_attempts = ["bdc", "zr6"]
        elif z6_formate_candidates:
            growth_attempts = ["bdc"]
        elif z6_branch_candidates:
            growth_attempts = ["zr6"]
        else:
            raise RuntimeError("Outer-shell growth stalled: no Zr6 formate or shell BDC benzoate sites available.")

        placed = False
        for next_kind in growth_attempts:
            try:
                if next_kind == "zr6":
                    grow_with_retries(
                        assembly,
                        next_kind="zr6",
                        attempts_per_step=attempts_per_step,
                        carboxylate_type="benzoate",
                        owner_kind_filter="BDC",
                        owner_connected_kind="Zr6_AA",
                        preference="random",
                    )
                else:
                    grow_with_retries(
                        assembly,
                        next_kind="bdc",
                        attempts_per_step=attempts_per_step,
                        carboxylate_type="formate",
                        owner_kind_filter="Zr6_AA",
                        preference="random",
                    )
                placed = True
                break
            except RuntimeError:
                continue

        if not placed:
            if len(assembly.ready_to_connect_carboxylate_pairs) > 0:
                pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
                assembly.link_internal_carboxylate(pair)
                continue
            raise RuntimeError(
                "Outer-shell growth stalled after both Zr6 and BDC placement attempts failed."
            )

        if len(assembly.entities) % 20 == 0 or len(assembly.entities) == target_entities:
            counts = entity_counts(assembly)
            snapshots.append(
                {
                    "total_entities": counts["total_entities"],
                    "Zr6_AA": counts["Zr6_AA"],
                    "Zr12_AA": counts["Zr12_AA"],
                    "BDC": counts["BDC"],
                    "ready_pairs": counts["ready_pairs"],
                }
            )
            print(json.dumps(snapshots[-1]))

    return snapshots


def entity_counts(assembly):
    counts = {"Zr6_AA": 0, "Zr12_AA": 0, "BDC": 0}
    for entity in assembly.entities:
        counts[entity_kind(entity)] += 1
    counts["total_entities"] = len(assembly.entities)
    counts["linked_pairs"] = len(assembly.linked_carboxylate_pairs)
    counts["ready_pairs"] = len(assembly.ready_to_connect_carboxylate_pairs)
    counts["free_carboxylates"] = len(assembly.free_carboxylates)
    return counts


def validate_seed(assembly, max_zr12_coordination, internal_zr12_count, max_axis_ratio_1_3=None):
    counts = entity_counts(assembly)
    summary = cluster_summary(assembly)
    shape = cluster_shape_metrics(assembly)
    zr12_rows = [row for row in summary if row["kind"] == "Zr12_AA"]
    if len(zr12_rows) < internal_zr12_count:
        raise RuntimeError(
            f"Expected at least {internal_zr12_count} internal Zr12 clusters, got {len(zr12_rows)}."
        )
    if any(row["coordination"] > max_zr12_coordination for row in zr12_rows):
        raise RuntimeError(
            f"At least one Zr12 cluster exceeded max coordination {max_zr12_coordination}. Summary: {summary}"
        )
    if any(row["radial_fraction"] > 0.70 for row in zr12_rows):
        raise RuntimeError(
            "At least one Zr12 cluster is still too close to the outer shell. "
            f"Summary: {summary}"
        )
    if max_axis_ratio_1_3 is not None and shape is not None:
        if shape["principal_axis_ratio_1_3"] > max_axis_ratio_1_3:
            raise RuntimeError(
                "Seed remained too anisotropic for the requested compactness screen. "
                f"Shape: {shape}"
            )
    return counts, summary, shape


def main():
    args = parse_args()
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    assembly = Assembly(Zr6_AA(), ZR6_PERCENTAGE=1.0, ENTROPY_GAIN=30.9, BUMPING_THRESHOLD=2.0)
    core_info = build_low_coordination_internal_zr12_core(
        assembly,
        args.attempts_per_step,
        args.internal_zr12_count,
        args.seed_zr6_branches,
        args.zr6_branches_per_zr12,
    )
    growth_snapshots = grow_outer_shell_zr6_only(
        assembly=assembly,
        target_entities=args.target_entities,
        attempts_per_step=args.attempts_per_step,
        internal_link_probability=args.internal_link_probability,
    )
    counts, coordination, cluster_shape = validate_seed(
        assembly,
        args.max_zr12_coordination,
        args.internal_zr12_count,
        args.max_axis_ratio_1_3,
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basename = args.basename or (
        f"internal_zr12_seed_{timestamp}_entities_{counts['total_entities']}_zr12_{counts['Zr12_AA']}"
    )

    mol2_path = output_dir / f"{basename}.mol2"
    pkl_path = output_dir / f"{basename}.pkl"
    json_path = output_dir / f"{basename}.json"

    assembly.get_mol2_file(str(mol2_path))
    if not safe_pickle_save(assembly, str(pkl_path), rebuild_after_save=False):
        raise RuntimeError(f"Failed to save pickle file to {pkl_path}")

    summary = {
        "rng_seed": args.rng_seed,
        "target_entities": args.target_entities,
        "attempts_per_step": args.attempts_per_step,
        "internal_link_probability": args.internal_link_probability,
        "internal_zr12_count": args.internal_zr12_count,
        "seed_zr6_branches": args.seed_zr6_branches,
        "zr6_branches_per_zr12": args.zr6_branches_per_zr12,
        "max_zr12_coordination": args.max_zr12_coordination,
        "max_axis_ratio_1_3": args.max_axis_ratio_1_3,
        "counts": counts,
        "cluster_shape": cluster_shape,
        "cluster_coordination": [
            {key: value for key, value in row.items() if key != "entity_ref"} for row in coordination
        ],
        "core_build_log": core_info["build_log"],
        "growth_snapshots": growth_snapshots,
        "outputs": {
            "mol2": str(mol2_path),
            "pkl": str(pkl_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
