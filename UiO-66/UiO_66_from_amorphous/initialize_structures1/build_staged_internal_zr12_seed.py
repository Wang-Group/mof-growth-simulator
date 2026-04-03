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
            "Build a compact mixed seed by first constructing a stable low-Zr12 core, "
            "then inserting additional under-coordinated internal Zr12 clusters at an "
            "intermediate growth stage before finishing Zr6-only shell growth."
        )
    )
    parser.add_argument("--rng-seed", type=int, default=326)
    parser.add_argument("--target-entities", type=int, default=260)
    parser.add_argument("--mid-growth-entities", type=int, default=120)
    parser.add_argument("--attempts-per-step", type=int, default=1000)
    parser.add_argument("--internal-link-probability", type=float, default=0.18)
    parser.add_argument("--initial-internal-zr12-count", type=int, default=2)
    parser.add_argument("--extra-internal-zr12-count", type=int, default=2)
    parser.add_argument("--seed-zr6-branches", type=int, default=4)
    parser.add_argument("--initial-zr6-branches-per-zr12", type=int, default=3)
    parser.add_argument("--extra-zr6-branches-per-zr12", type=int, default=2)
    parser.add_argument("--anchor-max-radial-fraction", type=float, default=0.58)
    parser.add_argument("--anchor-target-radial-fraction", type=float, default=0.28)
    parser.add_argument("--max-zr12-coordination", type=int, default=8)
    parser.add_argument("--max-axis-ratio-1-3", type=float, default=4.8)
    parser.add_argument("--output-dir", default="output/mixed_nuclei")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def candidate_anchor_rows(assembly, used_anchor_ids, max_radial_fraction, target_radial_fraction):
    rows = []
    for row in core_builder.cluster_summary(assembly):
        if row["kind"] != "Zr6_AA":
            continue
        if id(row["entity_ref"]) in used_anchor_ids:
            continue
        if row["radial_fraction"] > max_radial_fraction:
            continue
        free_sites = core_builder.free_sites(
            assembly,
            carboxylate_type="formate",
            owner_entity=row["entity_ref"],
        )
        if not free_sites:
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            abs(row["radial_fraction"] - target_radial_fraction),
            row["radius_from_cluster_centroid"],
        )
    )
    return rows


def insert_extra_zr12(
    assembly,
    attempts_per_step,
    extra_count,
    extra_zr6_branches_per_zr12,
    anchor_max_radial_fraction,
    anchor_target_radial_fraction,
):
    used_anchor_ids = set()
    inserted_rows = []

    for insert_index in range(extra_count):
        rows = candidate_anchor_rows(
            assembly,
            used_anchor_ids,
            anchor_max_radial_fraction,
            anchor_target_radial_fraction,
        )
        selected_anchor = None
        inserted_z12 = None
        last_error = None

        for row in rows[:20]:
            try:
                _, inserted_z12 = core_builder.grow_cluster_from_anchor(
                    assembly,
                    row["entity_ref"],
                    "zr12",
                    attempts_per_step,
                )
                selected_anchor = row
                break
            except Exception as exc:
                last_error = exc

        if inserted_z12 is None:
            raise RuntimeError(f"Failed to insert extra Zr12 cluster {insert_index}: {last_error}")

        used_anchor_ids.add(id(selected_anchor["entity_ref"]))
        branch_successes = 0
        for _ in range(extra_zr6_branches_per_zr12):
            try:
                core_builder.grow_cluster_from_anchor(
                    assembly,
                    inserted_z12,
                    "zr6",
                    max(250, attempts_per_step // 2),
                )
                branch_successes += 1
            except Exception:
                pass

        inserted_rows.append(
            {
                "insert_index": insert_index,
                "anchor_radial_fraction": selected_anchor["radial_fraction"],
                "anchor_radius": selected_anchor["radius_from_cluster_centroid"],
                "successful_zr6_branches": branch_successes,
            }
        )

    return inserted_rows


def link_all_ready_pairs(assembly, max_links=10000):
    linked_pairs = 0
    while len(assembly.ready_to_connect_carboxylate_pairs) > 0 and linked_pairs < max_links:
        pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
        assembly.link_internal_carboxylate(pair)
        linked_pairs += 1
    return linked_pairs


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
    growth_snapshots_before_insert = core_builder.grow_outer_shell_zr6_only(
        assembly,
        target_entities=args.mid_growth_entities,
        attempts_per_step=args.attempts_per_step,
        internal_link_probability=args.internal_link_probability,
    )

    inserted_rows = insert_extra_zr12(
        assembly,
        attempts_per_step=args.attempts_per_step,
        extra_count=args.extra_internal_zr12_count,
        extra_zr6_branches_per_zr12=args.extra_zr6_branches_per_zr12,
        anchor_max_radial_fraction=args.anchor_max_radial_fraction,
        anchor_target_radial_fraction=args.anchor_target_radial_fraction,
    )
    counts_after_insert = core_builder.entity_counts(assembly)

    growth_snapshots_after_insert = core_builder.grow_outer_shell_zr6_only(
        assembly,
        target_entities=args.target_entities,
        attempts_per_step=args.attempts_per_step,
        internal_link_probability=args.internal_link_probability,
    )
    counts_before_link_cleanup = core_builder.entity_counts(assembly)
    cleanup_links = link_all_ready_pairs(assembly)

    total_internal_zr12_count = args.initial_internal_zr12_count + args.extra_internal_zr12_count
    counts, coordination, cluster_shape = core_builder.validate_seed(
        assembly,
        args.max_zr12_coordination,
        total_internal_zr12_count,
        args.max_axis_ratio_1_3,
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basename = args.basename or (
        f"staged_internal_zr12_seed_{timestamp}_entities_{counts['total_entities']}_zr12_{counts['Zr12_AA']}"
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
        "mid_growth_entities": args.mid_growth_entities,
        "attempts_per_step": args.attempts_per_step,
        "internal_link_probability": args.internal_link_probability,
        "initial_internal_zr12_count": args.initial_internal_zr12_count,
        "extra_internal_zr12_count": args.extra_internal_zr12_count,
        "seed_zr6_branches": args.seed_zr6_branches,
        "initial_zr6_branches_per_zr12": args.initial_zr6_branches_per_zr12,
        "extra_zr6_branches_per_zr12": args.extra_zr6_branches_per_zr12,
        "anchor_max_radial_fraction": args.anchor_max_radial_fraction,
        "anchor_target_radial_fraction": args.anchor_target_radial_fraction,
        "max_zr12_coordination": args.max_zr12_coordination,
        "max_axis_ratio_1_3": args.max_axis_ratio_1_3,
        "counts_after_insert": counts_after_insert,
        "counts_before_link_cleanup": counts_before_link_cleanup,
        "cleanup_links_consumed": cleanup_links,
        "counts": counts,
        "cluster_shape": cluster_shape,
        "cluster_coordination": [
            {key: value for key, value in row.items() if key != "entity_ref"} for row in coordination
        ],
        "core_build_log": core_info["build_log"],
        "extra_zr12_insertions": inserted_rows,
        "growth_snapshots_before_insert": growth_snapshots_before_insert,
        "growth_snapshots_after_insert": growth_snapshots_after_insert,
        "outputs": {
            "mol2": str(mol2_path),
            "pkl": str(pkl_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
