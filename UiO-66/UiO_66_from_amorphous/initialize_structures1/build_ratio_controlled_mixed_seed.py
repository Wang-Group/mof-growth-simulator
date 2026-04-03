import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
from UiO66_Assembly_Large_Correction_conc import Assembly, Zr6_AA, safe_pickle_save


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact mixed seed with a user-controlled Zr12:Zr6 ratio. "
            "Growth uses the existing AA geometry while preventing Zr12 clusters "
            "from exceeding a coordination cap during stochastic growth."
        )
    )
    parser.add_argument("--rng-seed", type=int, default=612)
    parser.add_argument("--target-entities", type=int, default=140)
    parser.add_argument("--attempts-per-step", type=int, default=1000)
    parser.add_argument("--max-growth-steps", type=int, default=30000)
    parser.add_argument("--cluster-add-probability", type=float, default=0.65)
    parser.add_argument("--internal-link-probability", type=float, default=0.20)
    parser.add_argument("--target-zr12-fraction", type=float, default=0.55)
    parser.add_argument("--initial-internal-zr12-count", type=int, default=5)
    parser.add_argument("--seed-zr6-branches", type=int, default=4)
    parser.add_argument("--initial-zr6-branches-per-zr12", type=int, default=1)
    parser.add_argument("--inner-zr6-fill-count", type=int, default=0)
    parser.add_argument("--inner-zr6-radial-min", type=float, default=0.08)
    parser.add_argument("--inner-zr6-radial-max", type=float, default=0.58)
    parser.add_argument("--inner-zr6-target-radial", type=float, default=0.32)
    parser.add_argument("--max-zr12-coordination", type=int, default=8)
    parser.add_argument("--max-axis-ratio-1-3", type=float, default=None)
    parser.add_argument("--pick-preference", default="sparse_outer", choices=["random", "outermost", "innermost", "sparse_outer"])
    parser.add_argument("--output-dir", default="output/mixed_nuclei")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def linked_coordination_map(assembly):
    return core_builder.linked_coordination_map(assembly)


def filtered_formate_sites(assembly, max_zr12_coordination):
    coordination = linked_coordination_map(assembly)
    sites = []
    for carboxylate in assembly.free_carboxylates:
        if carboxylate.carboxylate_type != "formate":
            continue
        owner = getattr(carboxylate, "belonging_entity", None)
        if owner is None:
            continue
        owner_kind = core_builder.entity_kind(owner)
        if not owner_kind.startswith("Zr"):
            continue
        if owner_kind == "Zr12_AA" and coordination[id(owner)] >= max_zr12_coordination:
            continue
        sites.append(carboxylate)
    return sites


def filtered_benzoate_sites(assembly, max_zr12_coordination):
    coordination = linked_coordination_map(assembly)
    sites = []
    for carboxylate in assembly.free_carboxylates:
        if carboxylate.carboxylate_type != "benzoate":
            continue
        owner = getattr(carboxylate, "belonging_entity", None)
        if owner is None or core_builder.entity_kind(owner) != "BDC":
            continue

        blocked = False
        for neighbor in getattr(owner, "connected_entities", []):
            if core_builder.entity_kind(neighbor) == "Zr12_AA" and coordination[id(neighbor)] >= max_zr12_coordination:
                blocked = True
                break

        if not blocked:
            sites.append(carboxylate)
    return sites


def allowed_ready_pairs(assembly, max_zr12_coordination):
    coordination = linked_coordination_map(assembly)
    allowed = []
    for pair in assembly.ready_to_connect_carboxylate_pairs:
        blocked = False
        for carboxylate in pair:
            owner = getattr(carboxylate, "belonging_entity", None)
            if owner is None:
                continue
            if core_builder.entity_kind(owner) == "Zr12_AA" and coordination[id(owner)] >= max_zr12_coordination:
                blocked = True
                break
        if not blocked:
            allowed.append(pair)
    return allowed


def choose_cluster_kind(counts, target_zr12_fraction):
    z6_count = counts["Zr6_AA"]
    z12_count = counts["Zr12_AA"]
    current_fraction = z12_count / max(1, z6_count + z12_count)
    if current_fraction < target_zr12_fraction - 0.02:
        return "zr12"
    if current_fraction > target_zr12_fraction + 0.02:
        return "zr6"
    return "zr12" if random.random() < target_zr12_fraction else "zr6"


def consume_allowed_ready_pairs(assembly, max_zr12_coordination, max_links=5000):
    linked = 0
    while linked < max_links:
        ready_pairs = allowed_ready_pairs(assembly, max_zr12_coordination)
        if not ready_pairs:
            break
        assembly.link_internal_carboxylate(random.choice(ready_pairs))
        linked += 1
    return linked


def candidate_inner_zr6_anchor_rows(
    assembly,
    *,
    max_zr12_coordination,
    radial_min,
    radial_max,
    target_radial,
):
    rows = []
    for row in core_builder.cluster_summary(assembly):
        if row["kind"] not in {"Zr12_AA", "Zr6_AA"}:
            continue
        if row["radial_fraction"] < radial_min or row["radial_fraction"] > radial_max:
            continue
        free_formates = core_builder.free_sites(
            assembly,
            carboxylate_type="formate",
            owner_entity=row["entity_ref"],
        )
        if not free_formates:
            continue
        if row["kind"] == "Zr12_AA" and row["coordination"] >= max_zr12_coordination:
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            0 if row["kind"] == "Zr12_AA" else 1,
            abs(row["radial_fraction"] - target_radial),
            row["coordination"],
            row["radius_from_cluster_centroid"],
        )
    )
    return rows


def fill_inner_zr6_clusters(
    assembly,
    *,
    fill_count,
    attempts_per_step,
    max_zr12_coordination,
    radial_min,
    radial_max,
    target_radial,
):
    inserted_rows = []
    used_anchor_ids = set()

    for insert_index in range(fill_count):
        anchor_rows = candidate_inner_zr6_anchor_rows(
            assembly,
            max_zr12_coordination=max_zr12_coordination,
            radial_min=radial_min,
            radial_max=radial_max,
            target_radial=target_radial,
        )
        selected = None
        last_error = None

        for row in anchor_rows[:30]:
            anchor = row["entity_ref"]
            if id(anchor) in used_anchor_ids:
                continue
            try:
                _, new_zr6 = core_builder.grow_cluster_from_anchor(
                    assembly,
                    anchor,
                    "zr6",
                    attempts_per_step,
                )
                selected = (row, new_zr6)
                break
            except Exception as exc:
                last_error = exc

        if selected is None:
            raise RuntimeError(
                f"Failed to insert inner Zr6 cluster {insert_index}. Last error: {last_error}"
            )

        row, _new_zr6 = selected
        used_anchor_ids.add(id(row["entity_ref"]))
        inserted_rows.append(
            {
                "insert_index": insert_index,
                "anchor_kind": row["kind"],
                "anchor_coordination_before": row["coordination"],
                "anchor_radial_fraction": row["radial_fraction"],
                "anchor_radius": row["radius_from_cluster_centroid"],
            }
        )

    cleanup_links = consume_allowed_ready_pairs(assembly, max_zr12_coordination)
    return {
        "requested_fill_count": fill_count,
        "inserted_fill_count": len(inserted_rows),
        "cleanup_links_consumed": cleanup_links,
        "insertions": inserted_rows,
    }


def ratio_controlled_growth(
    assembly,
    *,
    target_entities,
    max_growth_steps,
    attempts_per_step,
    cluster_add_probability,
    internal_link_probability,
    target_zr12_fraction,
    max_zr12_coordination,
    pick_preference,
):
    snapshots = []
    stalled_external_attempts = 0

    for step_idx in range(1, max_growth_steps + 1):
        if len(assembly.entities) >= target_entities:
            break

        ready_pairs = allowed_ready_pairs(assembly, max_zr12_coordination)
        formate_sites = filtered_formate_sites(assembly, max_zr12_coordination)
        benzoate_sites = filtered_benzoate_sites(assembly, max_zr12_coordination)

        if ready_pairs and random.random() < internal_link_probability:
            assembly.link_internal_carboxylate(random.choice(ready_pairs))
            stalled_external_attempts = 0
        else:
            if benzoate_sites and formate_sites:
                next_kind = (
                    choose_cluster_kind(core_builder.entity_counts(assembly), target_zr12_fraction)
                    if random.random() < cluster_add_probability
                    else "bdc"
                )
            elif benzoate_sites:
                next_kind = choose_cluster_kind(core_builder.entity_counts(assembly), target_zr12_fraction)
            elif formate_sites:
                next_kind = "bdc"
            elif ready_pairs:
                assembly.link_internal_carboxylate(random.choice(ready_pairs))
                stalled_external_attempts = 0
                next_kind = None
            else:
                break

            if next_kind is not None:
                candidates = benzoate_sites if next_kind in {"zr6", "zr12"} else formate_sites
                if not candidates:
                    stalled_external_attempts += 1
                else:
                    selected_carboxylate = core_builder.pick_site(candidates, assembly, pick_preference)
                    new_entity = core_builder.grow_on_site(assembly, selected_carboxylate, next_kind)
                    stalled_external_attempts = 0 if new_entity is not None else stalled_external_attempts + 1

        if len(assembly.entities) % 20 == 0 or len(assembly.entities) == target_entities:
            counts = core_builder.entity_counts(assembly)
            snapshots.append(
                {
                    "step": step_idx,
                    "total_entities": counts["total_entities"],
                    "Zr6_AA": counts["Zr6_AA"],
                    "Zr12_AA": counts["Zr12_AA"],
                    "BDC": counts["BDC"],
                    "ready_pairs": counts["ready_pairs"],
                    "stalled_external_attempts": stalled_external_attempts,
                }
            )
            print(json.dumps(snapshots[-1]))

    cleanup_links = consume_allowed_ready_pairs(assembly, max_zr12_coordination)
    return {
        "cleanup_links_consumed": cleanup_links,
        "progress_snapshots": snapshots,
        "target_reached": len(assembly.entities) >= target_entities,
    }


def validate_final_seed(assembly, max_zr12_coordination, max_axis_ratio_1_3=None):
    counts = core_builder.entity_counts(assembly)
    summary = core_builder.cluster_summary(assembly)
    shape = core_builder.cluster_shape_metrics(assembly)
    zr12_rows = [row for row in summary if row["kind"] == "Zr12_AA"]

    if not zr12_rows:
        raise RuntimeError("No Zr12 clusters were present in the final seed.")
    if any(row["coordination"] > max_zr12_coordination for row in zr12_rows):
        raise RuntimeError(
            f"At least one Zr12 cluster exceeded max coordination {max_zr12_coordination}. Summary: {summary}"
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
    core_info = core_builder.build_low_coordination_internal_zr12_core(
        assembly,
        attempts_per_step=args.attempts_per_step,
        internal_zr12_count=args.initial_internal_zr12_count,
        seed_zr6_branches=args.seed_zr6_branches,
        zr6_branches_per_zr12=args.initial_zr6_branches_per_zr12,
    )
    growth_summary = ratio_controlled_growth(
        assembly,
        target_entities=args.target_entities,
        max_growth_steps=args.max_growth_steps,
        attempts_per_step=args.attempts_per_step,
        cluster_add_probability=args.cluster_add_probability,
        internal_link_probability=args.internal_link_probability,
        target_zr12_fraction=args.target_zr12_fraction,
        max_zr12_coordination=args.max_zr12_coordination,
        pick_preference=args.pick_preference,
    )
    inner_fill_summary = None
    if args.inner_zr6_fill_count > 0:
        inner_fill_summary = fill_inner_zr6_clusters(
            assembly,
            fill_count=args.inner_zr6_fill_count,
            attempts_per_step=args.attempts_per_step,
            max_zr12_coordination=args.max_zr12_coordination,
            radial_min=args.inner_zr6_radial_min,
            radial_max=args.inner_zr6_radial_max,
            target_radial=args.inner_zr6_target_radial,
        )
    counts, coordination, cluster_shape = validate_final_seed(
        assembly,
        args.max_zr12_coordination,
        args.max_axis_ratio_1_3,
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basename = args.basename or (
        f"ratio_controlled_mixed_seed_{timestamp}_entities_{counts['total_entities']}_zr12_{counts['Zr12_AA']}"
    )

    mol2_path = output_dir / f"{basename}.mol2"
    pkl_path = output_dir / f"{basename}.pkl"
    json_path = output_dir / f"{basename}.json"

    assembly.get_mol2_file(str(mol2_path))
    if not safe_pickle_save(assembly, str(pkl_path), rebuild_after_save=False):
        raise RuntimeError(f"Failed to save pickle file to {pkl_path}")

    zr12_rows = [row for row in coordination if row["kind"] == "Zr12_AA"]
    summary = {
        "rng_seed": args.rng_seed,
        "target_entities": args.target_entities,
        "attempts_per_step": args.attempts_per_step,
        "max_growth_steps": args.max_growth_steps,
        "cluster_add_probability": args.cluster_add_probability,
        "internal_link_probability": args.internal_link_probability,
        "target_zr12_fraction": args.target_zr12_fraction,
        "initial_internal_zr12_count": args.initial_internal_zr12_count,
        "seed_zr6_branches": args.seed_zr6_branches,
        "initial_zr6_branches_per_zr12": args.initial_zr6_branches_per_zr12,
        "inner_zr6_fill_count": args.inner_zr6_fill_count,
        "inner_zr6_radial_min": args.inner_zr6_radial_min,
        "inner_zr6_radial_max": args.inner_zr6_radial_max,
        "inner_zr6_target_radial": args.inner_zr6_target_radial,
        "max_zr12_coordination": args.max_zr12_coordination,
        "max_axis_ratio_1_3": args.max_axis_ratio_1_3,
        "pick_preference": args.pick_preference,
        "counts": counts,
        "cluster_shape": cluster_shape,
        "zr12_to_zr6_ratio": counts["Zr12_AA"] / max(1, counts["Zr6_AA"]),
        "zr12_radial_fraction_stats": {
            "min": min(row["radial_fraction"] for row in zr12_rows),
            "median": float(np.median([row["radial_fraction"] for row in zr12_rows])),
            "max": max(row["radial_fraction"] for row in zr12_rows),
        },
        "cluster_coordination": [
            {key: value for key, value in row.items() if key != "entity_ref"} for row in coordination
        ],
        "core_build_log": core_info["build_log"],
        "growth_summary": growth_summary,
        "inner_fill_summary": inner_fill_summary,
        "outputs": {
            "mol2": str(mol2_path),
            "pkl": str(pkl_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
