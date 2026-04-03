import argparse
import contextlib
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import build_mixed_aa_nucleus as mixed_builder
import probe_zr6_only_growth as probe
import run_outer_zr12_zr6_only_case as outer_stage


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-phase UiO-66 mixed-nuclei simulation: "
            "first build a random mixed Zr6/Zr12 amorphous seed under supersaturation, "
            "then evolve it under an equal make/break single-bond regime with Zr6-only addition."
        )
    )
    parser.add_argument("--phase1-rng-seed", type=int, default=7)
    parser.add_argument("--phase1-cluster-sequence", default="zr6,zr12,zr6,zr12,zr6")
    parser.add_argument("--attempts-per-step", type=int, default=2000)
    parser.add_argument("--phase1-target-entities", type=int, default=200)
    parser.add_argument("--phase1-max-growth-steps", type=int, default=200000)
    parser.add_argument("--phase1-cluster-add-probability", type=float, default=0.25)
    parser.add_argument("--phase1-internal-link-probability", type=float, default=0.20)
    parser.add_argument("--phase1-zr6-growth-percentage", type=float, default=0.35)
    parser.add_argument("--phase1-stall-link-burst", type=int, default=20)
    parser.add_argument("--phase1-link-burst-size", type=int, default=30)
    parser.add_argument("--phase1-progress-every", type=int, default=0)

    parser.add_argument("--phase2-replicates", type=int, default=4)
    parser.add_argument(
        "--phase2-followup-stages",
        type=int,
        default=3,
        help=(
            "Additional equal-make/break follow-up stages after the first evolution stage. "
            "Each follow-up starts from the previous stage's best Zr12-loss replicate."
        ),
    )
    parser.add_argument("--phase2-followup-replicates", type=int, default=None)
    parser.add_argument("--phase2-base-rng-seed", type=int, default=89000)
    parser.add_argument("--phase2-total-steps", type=int, default=30000)
    parser.add_argument("--phase2-max-entities-delta", type=int, default=300)
    parser.add_argument("--phase2-exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--phase2-zr-conc", type=float, default=1200.0)
    parser.add_argument("--phase2-linker-conc", type=float, default=69.1596872253079)
    parser.add_argument("--phase2-capping-agent-conc", type=float, default=180.0)
    parser.add_argument("--phase2-equilibrium-constant-coefficient", type=float, default=1.3)
    parser.add_argument(
        "--phase2-entropy-correction-coefficient",
        type=float,
        default=0.0,
        help=(
            "Set to 0 to make the internal-link weight per ready pair approach 1 at large entity count, "
            "matching a formate/benzoate ratio near 1."
        ),
    )
    parser.add_argument("--phase2-h2o-dmf-ratio", type=float, default=probe.RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument(
        "--phase2-dissolution-update-interval-steps",
        type=int,
        default=None,
        help="Defaults to a fixed t=0 make/break ratio for the whole phase-2 run.",
    )
    parser.add_argument("--bumping-threshold", type=float, default=probe.RUN_DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument(
        "--output-dir",
        default="output/mixed_nuclei/two_phase_amorphous_equilibrate_zr6only",
    )
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def build_phase1_seed(args):
    random.seed(args.phase1_rng_seed)
    np.random.seed(args.phase1_rng_seed)

    cluster_sequence = mixed_builder.normalize_cluster_sequence(args.phase1_cluster_sequence)
    assembly, build_log, linked_pairs = mixed_builder.build_target_sequence(
        cluster_sequence,
        args.attempts_per_step,
    )
    growth_summary = mixed_builder.stochastic_amorphous_growth(
        assembly=assembly,
        target_entities=args.phase1_target_entities,
        max_growth_steps=args.phase1_max_growth_steps,
        attempts_per_step=args.attempts_per_step,
        cluster_add_probability=args.phase1_cluster_add_probability,
        internal_link_probability=args.phase1_internal_link_probability,
        zr6_growth_percentage=args.phase1_zr6_growth_percentage,
        stall_link_burst=args.phase1_stall_link_burst,
        link_burst_size=args.phase1_link_burst_size,
        progress_every=args.phase1_progress_every,
    )

    counts = mixed_builder.entity_counts(assembly)
    counts["connected"] = mixed_builder.is_connected(assembly)
    counts["internal_links_created"] = linked_pairs

    coordination_summary = mixed_builder.cluster_coordination_summary(assembly)
    cluster_shape = core_builder.cluster_shape_metrics(assembly)
    cluster_rows = core_builder.cluster_summary(assembly)
    zr12_rows = [{key: value for key, value in row.items() if key != "entity_ref"} for row in cluster_rows if row["kind"] == "Zr12_AA"]
    zr6_rows = [{key: value for key, value in row.items() if key != "entity_ref"} for row in cluster_rows if row["kind"] == "Zr6_AA"]

    return assembly, {
        "phase1_rng_seed": args.phase1_rng_seed,
        "cluster_sequence": cluster_sequence,
        "attempts_per_step": args.attempts_per_step,
        "growth_settings": {
            "target_entities": args.phase1_target_entities,
            "max_growth_steps": args.phase1_max_growth_steps,
            "cluster_add_probability": args.phase1_cluster_add_probability,
            "internal_link_probability": args.phase1_internal_link_probability,
            "zr6_growth_percentage": args.phase1_zr6_growth_percentage,
            "stall_link_burst": args.phase1_stall_link_burst,
            "link_burst_size": args.phase1_link_burst_size,
            "progress_every": args.phase1_progress_every,
        },
        "counts": counts,
        "cluster_shape": cluster_shape,
        "cluster_coordination": coordination_summary,
        "zr12_rows": zr12_rows,
        "zr6_rows": zr6_rows,
        "build_log": build_log,
        "growth_summary": growth_summary,
    }


def build_phase2_candidate(args):
    _, formate_benzoate_ratio = probe.dissolution_probability(
        0.0,
        args.phase2_equilibrium_constant_coefficient,
        args.phase2_h2o_dmf_ratio,
        args.phase2_capping_agent_conc,
        args.phase2_linker_conc,
    )
    cluster_add_probability = probe.zr6_cluster_add_probability(
        args.phase2_zr_conc,
        args.phase2_linker_conc,
        zr6_percentage=1.0,
    )
    return {
        "label": "equal_make_break_zr6_only",
        "exchange_rxn_time_seconds": args.phase2_exchange_rxn_time_seconds,
        "zr_conc": args.phase2_zr_conc,
        "linker_conc": args.phase2_linker_conc,
        "capping_agent_conc": args.phase2_capping_agent_conc,
        "equilibrium_constant_coefficient": args.phase2_equilibrium_constant_coefficient,
        "cluster_add_probability": cluster_add_probability,
        "formate_benzoate_ratio_t0": formate_benzoate_ratio,
    }


def phase2_weight_summary(args, phase1_entity_count, candidate):
    return {
        "internal_link_weight_per_ready_pair_at_seed_size": probe.entropy_assembly(
            phase1_entity_count,
            args.phase2_entropy_correction_coefficient,
        ),
        "removal_weight_per_linked_pair_t0": candidate["formate_benzoate_ratio_t0"],
        "weight_ratio_removal_over_internal_link": (
            candidate["formate_benzoate_ratio_t0"]
            / probe.entropy_assembly(phase1_entity_count, args.phase2_entropy_correction_coefficient)
        ),
        "cluster_add_probability_zr6_only": candidate["cluster_add_probability"],
    }


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"two_phase_amorphous_equilibrate_zr6only_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        phase1_assembly, phase1_summary = build_phase1_seed(args)

    seed_stem = f"{run_name}__phase1_seed"
    seed_pkl_path, seed_mol2_path = outer_stage.save_assembly_outputs(phase1_assembly, run_dir, seed_stem)
    seed_json_path = run_dir / f"{seed_stem}.json"
    seed_json_path.write_text(json.dumps(phase1_summary, indent=2), encoding="utf-8")

    phase2_candidate = build_phase2_candidate(args)
    phase2_replicates = args.phase2_followup_replicates if args.phase2_followup_replicates is not None else args.phase2_replicates

    stage_summaries = [
        outer_stage.run_growth_stage(
            seed_path=seed_pkl_path,
            candidate=phase2_candidate,
            run_dir=run_dir,
            run_name=run_name,
            stage_index=1,
            replicates=args.phase2_replicates,
            base_rng_seed=args.phase2_base_rng_seed,
            total_steps=args.phase2_total_steps,
            max_entities_delta=args.phase2_max_entities_delta,
            dissolution_update_interval_steps=args.phase2_dissolution_update_interval_steps,
            bumping_threshold=args.bumping_threshold,
            entropy_correction_coefficient=args.phase2_entropy_correction_coefficient,
        )
    ]

    termination_reason = "single_stage_only"
    current_best_run = stage_summaries[0]["best_run_by_zr12_loss"]
    for followup_index in range(args.phase2_followup_stages):
        if current_best_run is None:
            termination_reason = "phase2_stage01_no_runs"
            break

        stage_index = followup_index + 2
        next_seed_path = Path(current_best_run["pkl_path"])
        next_seed_counts = outer_stage.probe.seed_counts(next_seed_path)
        if next_seed_counts["Zr12_AA"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_summary = outer_stage.run_growth_stage(
            seed_path=next_seed_path,
            candidate=phase2_candidate,
            run_dir=run_dir,
            run_name=run_name,
            stage_index=stage_index,
            replicates=phase2_replicates,
            base_rng_seed=args.phase2_base_rng_seed + followup_index * 1000 + 1000,
            total_steps=args.phase2_total_steps,
            max_entities_delta=args.phase2_max_entities_delta,
            dissolution_update_interval_steps=args.phase2_dissolution_update_interval_steps,
            bumping_threshold=args.bumping_threshold,
            entropy_correction_coefficient=args.phase2_entropy_correction_coefficient,
        )
        stage_summaries.append(stage_summary)

        previous_best_end_zr12 = current_best_run["end_zr12"]
        current_best_run = stage_summary["best_run_by_zr12_loss"]
        if current_best_run is None:
            termination_reason = f"phase2_stage{stage_index:02d}_no_runs"
            break
        if current_best_run["end_zr12"] >= previous_best_end_zr12:
            termination_reason = f"phase2_stage{stage_index:02d}_no_further_zr12_loss"
            break
    else:
        if args.phase2_followup_stages > 0:
            termination_reason = "requested_phase2_followup_stages_completed"

    all_run_payloads = [run_payload for stage in stage_summaries for run_payload in stage["runs"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "phase1_seed_outputs": {
            "pkl": seed_pkl_path.as_posix(),
            "mol2": seed_mol2_path.as_posix(),
            "json": seed_json_path.as_posix(),
        },
        "phase1_seed_summary": phase1_summary,
        "phase2_candidate": phase2_candidate,
        "phase2_weight_summary": phase2_weight_summary(
            args,
            phase1_summary["counts"]["total_entities"],
            phase2_candidate,
        ),
        "phase2_config": {
            "replicates": args.phase2_replicates,
            "followup_stages": args.phase2_followup_stages,
            "followup_replicates": phase2_replicates,
            "base_rng_seed": args.phase2_base_rng_seed,
            "total_steps": args.phase2_total_steps,
            "max_entities_delta": args.phase2_max_entities_delta,
            "entropy_correction_coefficient": args.phase2_entropy_correction_coefficient,
            "dissolution_update_interval_steps": args.phase2_dissolution_update_interval_steps,
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
                "phase1_seed_counts": phase1_summary["counts"],
                "phase2_weight_summary": summary["phase2_weight_summary"],
                "termination_reason": termination_reason,
                "best_run": summary["best_run_by_zr12_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
