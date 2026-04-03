import argparse
import contextlib
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import run_outer_zr12_zr6_only_case as outer_stage
from UiO66_Assembly_Large_Correction_conc import Assembly, Zr6_AA


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a sacrificial spoked seed with a small Zr6 hub scaffold, many dedicated Zr6 shell anchors, "
            "and terminal Zr12 shell clusters, then run long Zr6-only secondary growth."
        )
    )
    parser.add_argument("--seed-rng-seed", type=int, default=2402)
    parser.add_argument("--hub-zr6-count", type=int, default=8)
    parser.add_argument("--shell-zr12-count", type=int, default=30)
    parser.add_argument("--target-total-entities", type=int, default=200)
    parser.add_argument("--min-total-entities", type=int, default=185)
    parser.add_argument("--attempts-per-step", type=int, default=2400)
    parser.add_argument("--shell-zr12-max-coordination", type=int, default=1)
    parser.add_argument("--min-zr12-to-zr6-ratio", type=float, default=0.84)
    parser.add_argument("--max-zr12-to-zr6-ratio", type=float, default=1.05)
    parser.add_argument("--min-zr12-radial-min", type=float, default=0.40)
    parser.add_argument("--min-zr12-radial-median", type=float, default=0.75)
    parser.add_argument("--min-zr12-radial-mean", type=float, default=0.74)
    parser.add_argument("--max-hub-branch-usage", type=int, default=10)
    parser.add_argument("--max-hub-decoration-usage", type=int, default=8)
    parser.add_argument("--max-shell-anchor-decoration-usage", type=int, default=1)
    parser.add_argument("--max-zr12-per-shell-anchor", type=int, default=2)
    parser.add_argument("--shell-batch-size", type=int, default=4)
    parser.add_argument("--min-dedicated-shell-anchor-fraction", type=float, default=0.75)
    parser.add_argument(
        "--disable-shell-anchor-decorations",
        action="store_true",
        help="Only decorate hub Zr6 nodes. Shell anchors remain single-purpose supports for Zr12.",
    )
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument(
        "--followup-stages",
        type=int,
        default=2,
        help=(
            "Number of additional best-path growth stages to run after the initial stage. "
            "Each stage starts from the previous stage's best Zr12-loss replicate."
        ),
    )
    parser.add_argument(
        "--followup-replicates",
        type=int,
        default=None,
        help="Replicates per follow-up stage. Defaults to --replicates.",
    )
    parser.add_argument("--base-rng-seed", type=int, default=67000)
    parser.add_argument("--growth-total-steps", type=int, default=20000)
    parser.add_argument("--growth-max-entities-delta", type=int, default=360)
    parser.add_argument(
        "--growth-exchange-rxn-time-seconds",
        type=float,
        default=outer_stage.DEFAULT_GROWTH_CHEMISTRY["exchange_rxn_time_seconds"],
    )
    parser.add_argument(
        "--growth-zr-conc",
        type=float,
        default=outer_stage.DEFAULT_GROWTH_CHEMISTRY["zr_conc"],
    )
    parser.add_argument(
        "--growth-linker-conc",
        type=float,
        default=outer_stage.DEFAULT_GROWTH_CHEMISTRY["linker_conc"],
    )
    parser.add_argument(
        "--growth-capping-agent-conc",
        type=float,
        default=outer_stage.DEFAULT_GROWTH_CHEMISTRY["capping_agent_conc"],
    )
    parser.add_argument(
        "--growth-equilibrium-constant-coefficient",
        type=float,
        default=outer_stage.DEFAULT_GROWTH_CHEMISTRY["equilibrium_constant_coefficient"],
    )
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=outer_stage.probe.RUN_DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=outer_stage.probe.RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=outer_stage.probe.RUN_DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    parser.add_argument("--bumping-threshold", type=float, default=outer_stage.probe.RUN_DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument("--output-dir", default="output/mixed_nuclei/spoked_sacrificial_zr12_zr6only")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def radial_stats(rows):
    radial_fractions = [row["radial_fraction"] for row in rows]
    return {
        "count": len(rows),
        "min": min(radial_fractions),
        "median": float(np.median(radial_fractions)),
        "mean": float(np.mean(radial_fractions)),
        "max": max(radial_fractions),
        "outer_half_fraction": sum(1 for value in radial_fractions if value >= 0.5) / len(rows),
        "outer_70_fraction": sum(1 for value in radial_fractions if value >= 0.7) / len(rows),
    }


def row_map_for_kind(assembly, kind):
    return {
        id(row["entity_ref"]): row
        for row in core_builder.cluster_summary(assembly)
        if row["kind"] == kind
    }


def entity_candidate_rows(
    assembly,
    entities,
    usage_map,
    *,
    max_usage=None,
    prefer_outer=True,
    prioritize_inner=False,
):
    row_map = row_map_for_kind(assembly, "Zr6_AA")
    octant_center, octant_counts = core_builder.cluster_octant_populations(assembly)
    rows = []

    for entity in entities:
        row = row_map.get(id(entity))
        if row is None:
            continue

        free_formates = core_builder.free_sites(
            assembly,
            carboxylate_type="formate",
            owner_entity=entity,
        )
        if not free_formates:
            continue

        usage = usage_map.get(id(entity), 0)
        if max_usage is not None and usage >= max_usage:
            continue

        vector = np.asarray(entity.center, dtype=float) - octant_center
        radius = float(np.linalg.norm(vector))
        occupancy = 0 if radius < 1e-8 else octant_counts.get(core_builder.octant_key(vector), 0)
        rows.append(
            {
                "entity_ref": entity,
                "coordination": row["coordination"],
                "radial_fraction": row["radial_fraction"],
                "radius_from_cluster_centroid": row["radius_from_cluster_centroid"],
                "usage": usage,
                "free_formates": len(free_formates),
                "octant_occupancy": occupancy,
            }
        )

    if prioritize_inner:
        rows.sort(
            key=lambda row: (
                row["usage"],
                row["coordination"],
                row["octant_occupancy"],
                row["radius_from_cluster_centroid"],
                -row["free_formates"],
                random.random(),
            )
        )
    elif prefer_outer:
        rows.sort(
            key=lambda row: (
                row["usage"],
                row["octant_occupancy"],
                -row["radius_from_cluster_centroid"],
                row["coordination"],
                -row["free_formates"],
                random.random(),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                row["usage"],
                row["coordination"],
                row["octant_occupancy"],
                -row["radius_from_cluster_centroid"],
                -row["free_formates"],
                random.random(),
            )
        )

    return rows


def attach_cluster_to_parent(assembly, parent_entity, cluster_kind, attempts_per_step):
    return core_builder.grow_cluster_from_anchor(
        assembly,
        parent_entity,
        cluster_kind,
        attempts_per_step,
    )


def grow_dedicated_zr6_children(
    assembly,
    *,
    parent_entities,
    child_count,
    attempts_per_step,
    usage_map,
    max_usage=None,
    prefer_outer=True,
    prioritize_inner=False,
    stage_label,
    allow_partial=False,
):
    build_log = []
    children = []
    stall_info = None

    for child_index in range(child_count):
        candidate_rows = entity_candidate_rows(
            assembly,
            parent_entities,
            usage_map,
            max_usage=max_usage,
            prefer_outer=prefer_outer,
            prioritize_inner=prioritize_inner,
        )
        if not candidate_rows:
            raise RuntimeError(f"No parent Zr6 candidates remained while building {stage_label} child {child_index}.")

        selected_row = None
        last_error = None
        for row in candidate_rows[:40]:
            try:
                linker, child = attach_cluster_to_parent(
                    assembly,
                    row["entity_ref"],
                    "zr6",
                    attempts_per_step,
                )
                usage_map[id(row["entity_ref"])] = usage_map.get(id(row["entity_ref"]), 0) + 1
                selected_row = row
                children.append(child)
                build_log.append(
                    {
                        "stage": stage_label,
                        "child_index": child_index,
                        "parent_radius": row["radius_from_cluster_centroid"],
                        "parent_radial_fraction": row["radial_fraction"],
                        "parent_coordination_before": row["coordination"],
                        "parent_octant_occupancy": row["octant_occupancy"],
                        "parent_usage_before": row["usage"],
                        "linker_center": np.asarray(linker.center, dtype=float).round(4).tolist(),
                        "child_center": np.asarray(child.center, dtype=float).round(4).tolist(),
                    }
                )
                break
            except Exception as exc:
                last_error = exc

        if selected_row is None:
            if allow_partial:
                stall_info = {
                    "stage": stage_label,
                    "child_index": child_index,
                    "last_error": str(last_error),
                }
                break
            raise RuntimeError(
                f"Failed to build {stage_label} child {child_index}. Last error: {last_error}"
            )

    return children, build_log, stall_info


def attach_zr12_to_shell_anchors(
    assembly,
    *,
    shell_anchor_entities,
    target_count,
    attempts_per_step,
    max_zr12_per_anchor,
    initial_usage_map=None,
    attach_label="shell",
    allow_partial=False,
):
    attach_log = []
    zr12_entities = []
    usage_map = (
        dict(initial_usage_map)
        if initial_usage_map is not None
        else {id(entity): 0 for entity in shell_anchor_entities}
    )
    blocked_anchor_ids = set()
    stall_info = None

    for shell_index in range(target_count):
        candidate_entities = [
            entity for entity in shell_anchor_entities if id(entity) not in blocked_anchor_ids
        ]
        candidate_rows = entity_candidate_rows(
            assembly,
            candidate_entities,
            usage_map,
            max_usage=max_zr12_per_anchor,
            prefer_outer=True,
            prioritize_inner=False,
        )
        if not candidate_rows:
            if allow_partial:
                stall_info = {
                    "attach_label": attach_label,
                    "shell_index": shell_index,
                    "last_error": "No eligible shell anchors remained.",
                }
                break
            raise RuntimeError(
                f"No eligible shell anchors remained for Zr12 placement at index {shell_index}."
            )

        placed = False
        last_error = None
        for row in candidate_rows[:40]:
            try:
                linker, zr12 = attach_cluster_to_parent(
                    assembly,
                    row["entity_ref"],
                    "zr12",
                    attempts_per_step,
                )
                usage_map[id(row["entity_ref"])] = usage_map.get(id(row["entity_ref"]), 0) + 1
                zr12_entities.append(zr12)
                attach_log.append(
                    {
                        "attach_label": attach_label,
                        "shell_index": shell_index,
                        "parent_radius": row["radius_from_cluster_centroid"],
                        "parent_radial_fraction": row["radial_fraction"],
                        "parent_coordination_before": row["coordination"],
                        "parent_usage_before": row["usage"],
                        "linker_center": np.asarray(linker.center, dtype=float).round(4).tolist(),
                        "zr12_center": np.asarray(zr12.center, dtype=float).round(4).tolist(),
                    }
                )
                placed = True
                break
            except Exception as exc:
                blocked_anchor_ids.add(id(row["entity_ref"]))
                last_error = exc

        if not placed:
            if allow_partial:
                stall_info = {
                    "attach_label": attach_label,
                    "shell_index": shell_index,
                    "last_error": str(last_error),
                }
                break
            raise RuntimeError(
                f"Failed to place shell Zr12 {shell_index}. Last error: {last_error}"
            )

    return zr12_entities, attach_log, usage_map, stall_info


def add_dangling_bdc_decorations(
    assembly,
    *,
    hub_entities,
    shell_anchor_entities,
    decoration_count,
    attempts_per_step,
    allow_shell_anchor_decorations,
    max_hub_usage,
    max_shell_anchor_usage,
):
    build_log = []
    hub_usage = {id(entity): 0 for entity in hub_entities}
    shell_usage = {id(entity): 0 for entity in shell_anchor_entities}

    for decoration_index in range(decoration_count):
        selected_pool = None
        candidate_rows = entity_candidate_rows(
            assembly,
            hub_entities,
            hub_usage,
            max_usage=max_hub_usage,
            prefer_outer=False,
            prioritize_inner=True,
        )
        if candidate_rows:
            selected_pool = "hub"
            usage_map = hub_usage
        elif allow_shell_anchor_decorations:
            candidate_rows = entity_candidate_rows(
                assembly,
                shell_anchor_entities,
                shell_usage,
                max_usage=max_shell_anchor_usage,
                prefer_outer=False,
                prioritize_inner=False,
            )
            if candidate_rows:
                selected_pool = "shell_anchor"
                usage_map = shell_usage
            else:
                usage_map = None
        else:
            usage_map = None

        if not candidate_rows:
            raise RuntimeError(
                f"Could not place decoration BDC {decoration_index}; no eligible Zr6 formate sites remained."
            )

        placed = False
        last_error = None
        for row in candidate_rows[:40]:
            try:
                bdc = core_builder.grow_with_retries(
                    assembly,
                    next_kind="bdc",
                    attempts_per_step=attempts_per_step,
                    carboxylate_type="formate",
                    owner_entity=row["entity_ref"],
                    preference="random",
                )
                usage_map[id(row["entity_ref"])] = usage_map.get(id(row["entity_ref"]), 0) + 1
                build_log.append(
                    {
                        "decoration_index": decoration_index,
                        "pool": selected_pool,
                        "parent_radius": row["radius_from_cluster_centroid"],
                        "parent_radial_fraction": row["radial_fraction"],
                        "parent_coordination_before": row["coordination"],
                        "parent_usage_before": row["usage"],
                        "bdc_center": np.asarray(bdc.center, dtype=float).round(4).tolist(),
                    }
                )
                placed = True
                break
            except Exception as exc:
                last_error = exc

        if not placed:
            raise RuntimeError(
                f"Failed to place decoration BDC {decoration_index}. Last error: {last_error}"
            )

    return build_log


def build_spoked_seed(args):
    random.seed(args.seed_rng_seed)
    np.random.seed(args.seed_rng_seed)

    assembly = Assembly(Zr6_AA(), ZR6_PERCENTAGE=1.0, ENTROPY_GAIN=30.9, BUMPING_THRESHOLD=args.bumping_threshold)
    seed_hub = next(entity for entity in assembly.entities if core_builder.entity_kind(entity) == "Zr6_AA")

    hub_usage = {id(seed_hub): 0}
    hub_entities = [seed_hub]
    hub_children, hub_log, hub_stall = grow_dedicated_zr6_children(
        assembly,
        parent_entities=hub_entities,
        child_count=max(0, args.hub_zr6_count - 1),
        attempts_per_step=args.attempts_per_step,
        usage_map=hub_usage,
        max_usage=args.max_hub_branch_usage,
        prefer_outer=False,
        prioritize_inner=True,
        stage_label="hub",
        allow_partial=False,
    )
    hub_entities.extend(hub_children)

    shell_usage = {id(entity): 0 for entity in hub_entities}
    shell_anchor_entities = []
    shell_anchor_log = []
    shell_attach_log = []
    shell_attach_usage = {}
    zr12_entities = []
    shell_anchor_stall = None
    remaining_shell_zr12 = args.shell_zr12_count

    while remaining_shell_zr12 > 0:
        batch_size = min(args.shell_batch_size, remaining_shell_zr12)
        batch_anchors, batch_anchor_log, batch_stall = grow_dedicated_zr6_children(
            assembly,
            parent_entities=hub_entities,
            child_count=batch_size,
            attempts_per_step=args.attempts_per_step,
            usage_map=shell_usage,
            max_usage=args.max_hub_branch_usage,
            prefer_outer=True,
            prioritize_inner=False,
            stage_label="shell_anchor",
            allow_partial=True,
        )
        shell_anchor_entities.extend(batch_anchors)
        shell_anchor_log.extend(batch_anchor_log)
        if shell_anchor_stall is None and batch_stall is not None:
            shell_anchor_stall = batch_stall
        if not batch_anchors:
            break

        batch_zr12, batch_attach_log, batch_attach_usage, batch_attach_stall = attach_zr12_to_shell_anchors(
            assembly,
            shell_anchor_entities=batch_anchors,
            target_count=len(batch_anchors),
            attempts_per_step=args.attempts_per_step,
            max_zr12_per_anchor=1,
            attach_label="dedicated_batch",
            allow_partial=True,
        )
        zr12_entities.extend(batch_zr12)
        shell_attach_log.extend(batch_attach_log)
        shell_attach_usage.update(batch_attach_usage)
        remaining_shell_zr12 -= len(batch_zr12)

        if batch_stall is not None and not batch_anchors:
            break

    dedicated_anchor_fraction = len(shell_anchor_entities) / max(1, args.shell_zr12_count)
    if dedicated_anchor_fraction < args.min_dedicated_shell_anchor_fraction:
        raise RuntimeError(
            f"Dedicated shell-anchor fraction {dedicated_anchor_fraction:.3f} fell below "
            f"requested {args.min_dedicated_shell_anchor_fraction:.3f}."
        )

    dedicated_shell_anchor_count = sum(1 for value in shell_attach_usage.values() if value >= 1)

    if remaining_shell_zr12 > 0:
        extra_usage_map = {
            id(entity): max(shell_attach_usage.get(id(entity), 0), 1)
            for entity in shell_anchor_entities
        }
        extra_zr12, extra_attach_log, shell_attach_usage, _ = attach_zr12_to_shell_anchors(
            assembly,
            shell_anchor_entities=shell_anchor_entities,
            target_count=remaining_shell_zr12,
            attempts_per_step=args.attempts_per_step,
            max_zr12_per_anchor=args.max_zr12_per_shell_anchor,
            initial_usage_map=extra_usage_map,
            attach_label="fallback_reuse",
            allow_partial=False,
        )
        zr12_entities.extend(extra_zr12)
        shell_attach_log.extend(extra_attach_log)

    current_count = len(assembly.entities)
    if current_count < args.target_total_entities:
        decoration_log = add_dangling_bdc_decorations(
            assembly,
            hub_entities=hub_entities,
            shell_anchor_entities=shell_anchor_entities,
            decoration_count=args.target_total_entities - current_count,
            attempts_per_step=args.attempts_per_step,
            allow_shell_anchor_decorations=not args.disable_shell_anchor_decorations,
            max_hub_usage=args.max_hub_decoration_usage,
            max_shell_anchor_usage=args.max_shell_anchor_decoration_usage,
        )
    else:
        decoration_log = []

    counts = core_builder.entity_counts(assembly)
    if counts["total_entities"] < args.min_total_entities:
        raise RuntimeError(
            f"Spoked seed reached only {counts['total_entities']} entities; requested at least {args.min_total_entities}."
        )

    cluster_rows = core_builder.cluster_summary(assembly)
    row_map = {id(row["entity_ref"]): row for row in cluster_rows}
    zr12_rows = [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in cluster_rows
        if row["kind"] == "Zr12_AA"
    ]
    zr6_rows = [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in cluster_rows
        if row["kind"] == "Zr6_AA"
    ]
    hub_rows = [
        {key: value for key, value in row_map[id(entity)].items() if key != "entity_ref"}
        for entity in hub_entities
    ]
    shell_anchor_rows = [
        {key: value for key, value in row_map[id(entity)].items() if key != "entity_ref"}
        for entity in shell_anchor_entities
    ]

    ratio = counts["Zr12_AA"] / max(1, counts["Zr6_AA"])
    zr12_stats = radial_stats(zr12_rows)
    zr6_stats = radial_stats(zr6_rows)
    shell_anchor_stats = radial_stats(shell_anchor_rows)
    hub_stats = radial_stats(hub_rows)

    if counts["Zr12_AA"] != args.shell_zr12_count:
        raise RuntimeError(
            f"Expected {args.shell_zr12_count} shell Zr12 clusters, found {counts['Zr12_AA']}."
        )

    if ratio < args.min_zr12_to_zr6_ratio or ratio > args.max_zr12_to_zr6_ratio:
        raise RuntimeError(
            f"Zr12:Zr6 ratio {ratio:.3f} fell outside "
            f"[{args.min_zr12_to_zr6_ratio:.3f}, {args.max_zr12_to_zr6_ratio:.3f}]."
        )
    if zr12_stats["min"] < args.min_zr12_radial_min:
        raise RuntimeError(
            f"Zr12 radial min {zr12_stats['min']:.3f} was below requested {args.min_zr12_radial_min:.3f}."
        )
    if zr12_stats["median"] < args.min_zr12_radial_median:
        raise RuntimeError(
            f"Zr12 radial median {zr12_stats['median']:.3f} was below requested {args.min_zr12_radial_median:.3f}."
        )
    if zr12_stats["mean"] < args.min_zr12_radial_mean:
        raise RuntimeError(
            f"Zr12 radial mean {zr12_stats['mean']:.3f} was below requested {args.min_zr12_radial_mean:.3f}."
        )
    if any(row["coordination"] > args.shell_zr12_max_coordination for row in zr12_rows):
        raise RuntimeError(
            f"At least one shell Zr12 exceeded coordination {args.shell_zr12_max_coordination}."
        )

    dedicated_shell_zr12_count = sum(
        1 for row in shell_attach_log if row["attach_label"] == "dedicated_batch"
    )
    fallback_reused_shell_zr12_count = sum(
        1 for row in shell_attach_log if row["attach_label"] == "fallback_reuse"
    )

    return assembly, {
        "seed_rng_seed": args.seed_rng_seed,
        "hub_zr6_count": args.hub_zr6_count,
        "shell_zr12_count": args.shell_zr12_count,
        "target_total_entities": args.target_total_entities,
        "min_total_entities": args.min_total_entities,
        "attempts_per_step": args.attempts_per_step,
        "counts": counts,
        "cluster_shape": core_builder.cluster_shape_metrics(assembly),
        "zr12_to_zr6_ratio": ratio,
        "zr12_radial_stats": zr12_stats,
        "zr6_radial_stats": zr6_stats,
        "hub_radial_stats": hub_stats,
        "shell_anchor_radial_stats": shell_anchor_stats,
        "delta_mean_radial_zr12_minus_zr6": zr12_stats["mean"] - zr6_stats["mean"],
        "hub_build_log": hub_log,
        "hub_build_stall": hub_stall,
        "shell_anchor_build_log": shell_anchor_log,
        "shell_anchor_build_stall": shell_anchor_stall,
        "shell_anchor_built_count": len(shell_anchor_entities),
        "dedicated_shell_anchor_fraction": dedicated_anchor_fraction,
        "dedicated_shell_anchor_count": dedicated_shell_anchor_count,
        "direct_shell_zr12_count": dedicated_shell_zr12_count,
        "dedicated_shell_zr12_count": dedicated_shell_zr12_count,
        "fallback_reused_shell_zr12_count": fallback_reused_shell_zr12_count,
        "multi_loaded_shell_anchor_count": sum(1 for value in shell_attach_usage.values() if value > 1),
        "shell_attach_log": shell_attach_log,
        "decoration_log": decoration_log,
        "zr12_rows": zr12_rows,
        "zr6_rows": zr6_rows,
        "hub_rows": hub_rows,
        "shell_anchor_rows": shell_anchor_rows,
        "spoked_seed_screen": {
            "min_zr12_to_zr6_ratio": args.min_zr12_to_zr6_ratio,
            "max_zr12_to_zr6_ratio": args.max_zr12_to_zr6_ratio,
            "min_zr12_radial_min": args.min_zr12_radial_min,
            "min_zr12_radial_median": args.min_zr12_radial_median,
            "min_zr12_radial_mean": args.min_zr12_radial_mean,
            "shell_zr12_max_coordination": args.shell_zr12_max_coordination,
        },
    }


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"spoked_sacrificial_zr12_zr6only_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        seed_assembly, seed_summary = build_spoked_seed(args)

    seed_stem = f"{run_name}__seed"
    seed_pkl_path, seed_mol2_path = outer_stage.save_assembly_outputs(seed_assembly, run_dir, seed_stem)
    seed_json_path = run_dir / f"{seed_stem}.json"
    seed_json_path.write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")

    candidate = outer_stage.candidate_from_args(args)
    followup_replicates = args.followup_replicates if args.followup_replicates is not None else args.replicates
    stage_summaries = [
        outer_stage.run_growth_stage(
            seed_path=seed_pkl_path,
            candidate=candidate,
            run_dir=run_dir,
            run_name=run_name,
            stage_index=1,
            replicates=args.replicates,
            base_rng_seed=args.base_rng_seed,
            total_steps=args.growth_total_steps,
            max_entities_delta=args.growth_max_entities_delta,
            dissolution_update_interval_steps=args.dissolution_update_interval_steps,
            bumping_threshold=args.bumping_threshold,
            entropy_correction_coefficient=args.entropy_correction_coefficient,
        )
    ]

    termination_reason = "single_stage_only"
    current_best_run = stage_summaries[0]["best_run_by_zr12_loss"]
    for followup_index in range(args.followup_stages):
        if current_best_run is None:
            termination_reason = "no_stage1_runs"
            break
        if current_best_run["end_zr12"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_index = followup_index + 2
        next_seed_path = Path(current_best_run["pkl_path"])
        next_seed_counts = outer_stage.probe.seed_counts(next_seed_path)
        if next_seed_counts["Zr12_AA"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_summary = outer_stage.run_growth_stage(
            seed_path=next_seed_path,
            candidate=candidate,
            run_dir=run_dir,
            run_name=run_name,
            stage_index=stage_index,
            replicates=followup_replicates,
            base_rng_seed=args.base_rng_seed + followup_index * 1000 + 1000,
            total_steps=args.growth_total_steps,
            max_entities_delta=args.growth_max_entities_delta,
            dissolution_update_interval_steps=args.dissolution_update_interval_steps,
            bumping_threshold=args.bumping_threshold,
            entropy_correction_coefficient=args.entropy_correction_coefficient,
        )
        stage_summaries.append(stage_summary)

        current_best_run = stage_summary["best_run_by_zr12_loss"]
        if current_best_run is None:
            termination_reason = f"stage{stage_index:02d}_produced_no_runs"
            break
        if current_best_run["end_zr12"] >= next_seed_counts["Zr12_AA"]:
            termination_reason = f"stage{stage_index:02d}_no_further_zr12_loss"
            break
    else:
        if args.followup_stages > 0:
            termination_reason = "requested_followup_stages_completed"

    all_run_payloads = [run_payload for stage in stage_summaries for run_payload in stage["runs"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "seed_outputs": {
            "pkl": seed_pkl_path.as_posix(),
            "mol2": seed_mol2_path.as_posix(),
            "json": seed_json_path.as_posix(),
        },
        "seed_summary": seed_summary,
        "growth_candidate": candidate,
        "growth_config": {
            "replicates": args.replicates,
            "followup_stages": args.followup_stages,
            "followup_replicates": followup_replicates,
            "base_rng_seed": args.base_rng_seed,
            "growth_total_steps": args.growth_total_steps,
            "growth_max_entities_delta": args.growth_max_entities_delta,
            "entropy_correction_coefficient": args.entropy_correction_coefficient,
            "h2o_dmf_ratio": args.h2o_dmf_ratio,
            "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
            "bumping_threshold": args.bumping_threshold,
        },
        "termination_reason": termination_reason,
        "stages": stage_summaries,
        "runs": all_run_payloads,
        "best_run_by_zr12_loss": outer_stage.summarize_best_run(all_run_payloads),
    }

    summary_path = run_dir / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "seed_zr12_count": seed_summary["counts"]["Zr12_AA"],
                "seed_zr6_count": seed_summary["counts"]["Zr6_AA"],
                "seed_total_entities": seed_summary["counts"]["total_entities"],
                "seed_ratio_zr12_to_zr6": seed_summary["zr12_to_zr6_ratio"],
                "termination_reason": termination_reason,
                "best_run": summary["best_run_by_zr12_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
