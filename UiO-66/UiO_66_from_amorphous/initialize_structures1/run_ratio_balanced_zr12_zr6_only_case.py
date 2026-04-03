import argparse
import contextlib
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import build_ratio_controlled_mixed_seed as ratio_builder
import run_outer_zr12_zr6_only_case as stage_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a near-1:1 mixed Zr6/Zr12 seed with outward-biased Zr12 placement, "
            "then run long Zr6-only secondary growth."
        )
    )
    parser.add_argument("--seed-rng-seed", type=int, default=635)
    parser.add_argument("--seed-target-entities", type=int, default=200)
    parser.add_argument("--seed-attempts-per-step", type=int, default=1000)
    parser.add_argument("--seed-max-growth-steps", type=int, default=30000)
    parser.add_argument("--seed-cluster-add-probability", type=float, default=0.65)
    parser.add_argument("--seed-internal-link-probability", type=float, default=0.20)
    parser.add_argument("--seed-target-zr12-fraction", type=float, default=0.50)
    parser.add_argument("--seed-initial-internal-zr12-count", type=int, default=5)
    parser.add_argument("--seed-zr6-branches", type=int, default=4)
    parser.add_argument("--seed-initial-zr6-branches-per-zr12", type=int, default=1)
    parser.add_argument("--seed-inner-zr6-fill-count", type=int, default=2)
    parser.add_argument("--seed-inner-zr6-radial-min", type=float, default=0.08)
    parser.add_argument("--seed-inner-zr6-radial-max", type=float, default=0.58)
    parser.add_argument("--seed-inner-zr6-target-radial", type=float, default=0.32)
    parser.add_argument("--seed-max-zr12-coordination", type=int, default=8)
    parser.add_argument(
        "--seed-pick-preference",
        default="sparse_outer",
        choices=["random", "outermost", "innermost", "sparse_outer"],
    )
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument(
        "--followup-stages",
        type=int,
        default=0,
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
    parser.add_argument("--base-rng-seed", type=int, default=43000)
    parser.add_argument("--growth-total-steps", type=int, default=20000)
    parser.add_argument("--growth-max-entities-delta", type=int, default=360)
    parser.add_argument("--growth-exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--growth-zr-conc", type=float, default=5000.0)
    parser.add_argument("--growth-linker-conc", type=float, default=69.1596872253079)
    parser.add_argument("--growth-capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--growth-equilibrium-constant-coefficient", type=float, default=6.0)
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument(
        "--h2o-dmf-ratio",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["H2O_DMF_RATIO"],
    )
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=stage_runner.probe.RUN_DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    parser.add_argument(
        "--bumping-threshold",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["BUMPING_THRESHOLD"],
    )
    parser.add_argument("--output-dir", default="output/mixed_nuclei/ratio_balanced_zr12_zr6only")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def build_ratio_balanced_seed(args):
    random.seed(args.seed_rng_seed)
    np.random.seed(args.seed_rng_seed)

    assembly = stage_runner.Assembly(
        stage_runner.Zr6_AA(),
        ZR6_PERCENTAGE=1.0,
        ENTROPY_GAIN=30.9,
        BUMPING_THRESHOLD=args.bumping_threshold,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        core_info = core_builder.build_low_coordination_internal_zr12_core(
            assembly,
            attempts_per_step=args.seed_attempts_per_step,
            internal_zr12_count=args.seed_initial_internal_zr12_count,
            seed_zr6_branches=args.seed_zr6_branches,
            zr6_branches_per_zr12=args.seed_initial_zr6_branches_per_zr12,
        )
        growth_summary = ratio_builder.ratio_controlled_growth(
            assembly,
            target_entities=args.seed_target_entities,
            max_growth_steps=args.seed_max_growth_steps,
            attempts_per_step=args.seed_attempts_per_step,
            cluster_add_probability=args.seed_cluster_add_probability,
            internal_link_probability=args.seed_internal_link_probability,
            target_zr12_fraction=args.seed_target_zr12_fraction,
            max_zr12_coordination=args.seed_max_zr12_coordination,
            pick_preference=args.seed_pick_preference,
        )
        inner_fill_summary = None
        if args.seed_inner_zr6_fill_count > 0:
            inner_fill_summary = ratio_builder.fill_inner_zr6_clusters(
                assembly,
                fill_count=args.seed_inner_zr6_fill_count,
                attempts_per_step=args.seed_attempts_per_step,
                max_zr12_coordination=args.seed_max_zr12_coordination,
                radial_min=args.seed_inner_zr6_radial_min,
                radial_max=args.seed_inner_zr6_radial_max,
                target_radial=args.seed_inner_zr6_target_radial,
            )

    counts, coordination, cluster_shape = ratio_builder.validate_final_seed(
        assembly,
        args.seed_max_zr12_coordination,
        None,
    )

    zr12_rows = [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in coordination
        if row["kind"] == "Zr12_AA"
    ]
    zr6_rows = [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in coordination
        if row["kind"] == "Zr6_AA"
    ]

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

    zr12_stats = radial_stats(zr12_rows)
    zr6_stats = radial_stats(zr6_rows)

    return assembly, {
        "seed_rng_seed": args.seed_rng_seed,
        "seed_target_entities": args.seed_target_entities,
        "seed_attempts_per_step": args.seed_attempts_per_step,
        "seed_max_growth_steps": args.seed_max_growth_steps,
        "seed_cluster_add_probability": args.seed_cluster_add_probability,
        "seed_internal_link_probability": args.seed_internal_link_probability,
        "seed_target_zr12_fraction": args.seed_target_zr12_fraction,
        "seed_initial_internal_zr12_count": args.seed_initial_internal_zr12_count,
        "seed_zr6_branches": args.seed_zr6_branches,
        "seed_initial_zr6_branches_per_zr12": args.seed_initial_zr6_branches_per_zr12,
        "seed_inner_zr6_fill_count": args.seed_inner_zr6_fill_count,
        "seed_inner_zr6_radial_min": args.seed_inner_zr6_radial_min,
        "seed_inner_zr6_radial_max": args.seed_inner_zr6_radial_max,
        "seed_inner_zr6_target_radial": args.seed_inner_zr6_target_radial,
        "seed_max_zr12_coordination": args.seed_max_zr12_coordination,
        "seed_pick_preference": args.seed_pick_preference,
        "counts": counts,
        "cluster_shape": cluster_shape,
        "zr12_to_zr6_ratio": counts["Zr12_AA"] / max(1, counts["Zr6_AA"]),
        "zr12_radial_stats": zr12_stats,
        "zr6_radial_stats": zr6_stats,
        "delta_mean_radial_zr12_minus_zr6": zr12_stats["mean"] - zr6_stats["mean"],
        "cluster_coordination": [
            {key: value for key, value in row.items() if key != "entity_ref"} for row in coordination
        ],
        "core_build_log": core_info["build_log"],
        "growth_summary": growth_summary,
        "inner_fill_summary": inner_fill_summary,
        "zr12_rows": zr12_rows,
        "zr6_rows": zr6_rows,
    }


def main():
    args = parse_args()
    output_root = stage_runner.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"ratio_balanced_zr12_zr6only_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_assembly, seed_summary = build_ratio_balanced_seed(args)
    seed_stem = f"{run_name}__seed"
    seed_pkl_path, seed_mol2_path = stage_runner.save_assembly_outputs(seed_assembly, run_dir, seed_stem)
    seed_json_path = run_dir / f"{seed_stem}.json"
    seed_json_path.write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")

    candidate = stage_runner.candidate_from_args(args)
    followup_replicates = args.followup_replicates if args.followup_replicates is not None else args.replicates
    stage_summaries = [
        stage_runner.run_growth_stage(
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
        next_seed_counts = stage_runner.probe.seed_counts(next_seed_path)
        if next_seed_counts["Zr12_AA"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_summary = stage_runner.run_growth_stage(
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
        "best_run_by_zr12_loss": stage_runner.summarize_best_run(all_run_payloads),
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
