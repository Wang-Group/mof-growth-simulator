import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import run_outer_zr12_zr6_only_case as outer_stage


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a near-1:1 layered mixed seed with a Zr6-rich core and a Zr12-rich outer shell, "
            "then run long Zr6-only secondary growth."
        )
    )
    parser.add_argument("--seed-rng-seed", type=int, default=2402)
    parser.add_argument("--core-target-entities", type=int, default=110)
    parser.add_argument("--shell-zr12-count", type=int, default=34)
    parser.add_argument("--attempts-per-step", type=int, default=1200)
    parser.add_argument("--seed-internal-link-probability", type=float, default=0.18)
    parser.add_argument("--shell-anchor-min-radial-fraction", type=float, default=0.45)
    parser.add_argument("--shell-zr12-max-coordination", type=int, default=4)
    parser.add_argument(
        "--shell-anchor-max-coordination",
        type=int,
        default=None,
        help="Only use Zr6 shell anchors at or below this coordination. Default keeps all anchors.",
    )
    parser.add_argument(
        "--shell-anchor-max-usage",
        type=int,
        default=None,
        help="Maximum number of Zr12 shell placements allowed per Zr6 anchor. Default reuses anchors freely.",
    )
    parser.add_argument(
        "--shell-anchor-sort-mode",
        choices=("radial_first", "terminal_first", "sacrificial_first"),
        default="radial_first",
        help=(
            "How to rank candidate Zr6 shell anchors. "
            "'terminal_first' and 'sacrificial_first' bias toward weaker-connected anchors."
        ),
    )
    parser.add_argument(
        "--shell-zr12-min-final-radial-fraction",
        type=float,
        default=0.0,
        help="Loose per-cluster lower bound; layered-shell validation below is the main screen.",
    )
    parser.add_argument("--min-zr12-to-zr6-ratio", type=float, default=0.90)
    parser.add_argument("--max-zr12-to-zr6-ratio", type=float, default=1.15)
    parser.add_argument("--min-zr12-radial-min", type=float, default=0.20)
    parser.add_argument("--min-zr12-radial-median", type=float, default=0.70)
    parser.add_argument("--min-zr12-radial-mean", type=float, default=0.65)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument(
        "--disable-strip-stage",
        action="store_true",
        help="Skip the initial stripping chemistry stage and start directly with the growth chemistry.",
    )
    parser.add_argument(
        "--strip-replicates",
        type=int,
        default=None,
        help="Replicates in the stripping stage. Defaults to --replicates.",
    )
    parser.add_argument("--strip-total-steps", type=int, default=20000)
    parser.add_argument("--strip-max-entities-delta", type=int, default=120)
    parser.add_argument("--strip-exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--strip-zr-conc", type=float, default=5000.0)
    parser.add_argument("--strip-linker-conc", type=float, default=100.0)
    parser.add_argument("--strip-capping-agent-conc", type=float, default=500.0)
    parser.add_argument("--strip-equilibrium-constant-coefficient", type=float, default=4.0)
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
    parser.add_argument("--base-rng-seed", type=int, default=47000)
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
    parser.add_argument("--output-dir", default="output/mixed_nuclei/layered_ratio_zr12_zr6only")
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


def build_layered_seed(args):
    assembly, seed_summary = outer_stage.build_outer_zr12_seed(args)
    cluster_rows = core_builder.cluster_summary(assembly)
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

    zr12_stats = radial_stats(zr12_rows)
    zr6_stats = radial_stats(zr6_rows)
    ratio = seed_summary["counts"]["Zr12_AA"] / max(1, seed_summary["counts"]["Zr6_AA"])

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

    return assembly, {
        **seed_summary,
        "zr12_to_zr6_ratio": ratio,
        "zr12_radial_stats": zr12_stats,
        "zr6_radial_stats": zr6_stats,
        "delta_mean_radial_zr12_minus_zr6": zr12_stats["mean"] - zr6_stats["mean"],
        "zr12_rows": zr12_rows,
        "zr6_rows": zr6_rows,
        "layered_seed_screen": {
            "min_zr12_to_zr6_ratio": args.min_zr12_to_zr6_ratio,
            "max_zr12_to_zr6_ratio": args.max_zr12_to_zr6_ratio,
            "min_zr12_radial_min": args.min_zr12_radial_min,
            "min_zr12_radial_median": args.min_zr12_radial_median,
            "min_zr12_radial_mean": args.min_zr12_radial_mean,
        },
    }


def candidate_from_values(
    *,
    label,
    exchange_rxn_time_seconds,
    zr_conc,
    linker_conc,
    capping_agent_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
):
    _, formate_ratio = outer_stage.probe.dissolution_probability(
        0.0,
        equilibrium_constant_coefficient,
        h2o_dmf_ratio,
        capping_agent_conc,
        linker_conc,
    )
    return {
        "label": label,
        "exchange_rxn_time_seconds": exchange_rxn_time_seconds,
        "zr_conc": zr_conc,
        "linker_conc": linker_conc,
        "capping_agent_conc": capping_agent_conc,
        "equilibrium_constant_coefficient": equilibrium_constant_coefficient,
        "cluster_add_probability": outer_stage.probe.zr6_cluster_add_probability(
            zr_conc,
            linker_conc,
            zr6_percentage=1.0,
        ),
        "formate_benzoate_ratio_t0": formate_ratio,
    }


def annotate_stage_summary(stage_summary, *, stage_kind, candidate, total_steps, max_entities_delta):
    stage_summary["stage_kind"] = stage_kind
    stage_summary["stage_candidate"] = candidate
    stage_summary["stage_total_steps"] = total_steps
    stage_summary["stage_max_entities_delta"] = max_entities_delta
    return stage_summary


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"layered_ratio_zr12_zr6only_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_assembly, seed_summary = build_layered_seed(args)
    seed_stem = f"{run_name}__seed"
    seed_pkl_path, seed_mol2_path = outer_stage.save_assembly_outputs(seed_assembly, run_dir, seed_stem)
    seed_json_path = run_dir / f"{seed_stem}.json"
    seed_json_path.write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")

    strip_enabled = not args.disable_strip_stage
    strip_replicates = args.strip_replicates if args.strip_replicates is not None else args.replicates
    strip_candidate = candidate_from_values(
        label="strip_stage",
        exchange_rxn_time_seconds=args.strip_exchange_rxn_time_seconds,
        zr_conc=args.strip_zr_conc,
        linker_conc=args.strip_linker_conc,
        capping_agent_conc=args.strip_capping_agent_conc,
        equilibrium_constant_coefficient=args.strip_equilibrium_constant_coefficient,
        h2o_dmf_ratio=args.h2o_dmf_ratio,
    )
    growth_candidate = outer_stage.candidate_from_args(args)
    followup_replicates = args.followup_replicates if args.followup_replicates is not None else args.replicates

    stage_summaries = []
    stage_index = 1
    current_seed_path = seed_pkl_path

    if strip_enabled:
        strip_summary = annotate_stage_summary(
            outer_stage.run_growth_stage(
                seed_path=current_seed_path,
                candidate=strip_candidate,
                run_dir=run_dir,
                run_name=run_name,
                stage_index=stage_index,
                replicates=strip_replicates,
                base_rng_seed=args.base_rng_seed,
                total_steps=args.strip_total_steps,
                max_entities_delta=args.strip_max_entities_delta,
                dissolution_update_interval_steps=args.dissolution_update_interval_steps,
                bumping_threshold=args.bumping_threshold,
                entropy_correction_coefficient=args.entropy_correction_coefficient,
            ),
            stage_kind="strip",
            candidate=strip_candidate,
            total_steps=args.strip_total_steps,
            max_entities_delta=args.strip_max_entities_delta,
        )
        stage_summaries.append(strip_summary)
        current_best_run = strip_summary["best_run_by_zr12_loss"]
        if current_best_run is None:
            termination_reason = "strip_stage_produced_no_runs"
        else:
            current_seed_path = Path(current_best_run["pkl_path"])
            stage_index += 1
            termination_reason = "strip_stage_only"
    else:
        current_best_run = None
        termination_reason = "single_stage_only"

    growth_stage_summary = annotate_stage_summary(
        outer_stage.run_growth_stage(
            seed_path=current_seed_path,
            candidate=growth_candidate,
            run_dir=run_dir,
            run_name=run_name,
            stage_index=stage_index,
            replicates=args.replicates,
            base_rng_seed=args.base_rng_seed + (1000 if strip_enabled else 0),
            total_steps=args.growth_total_steps,
            max_entities_delta=args.growth_max_entities_delta,
            dissolution_update_interval_steps=args.dissolution_update_interval_steps,
            bumping_threshold=args.bumping_threshold,
            entropy_correction_coefficient=args.entropy_correction_coefficient,
        ),
        stage_kind="growth",
        candidate=growth_candidate,
        total_steps=args.growth_total_steps,
        max_entities_delta=args.growth_max_entities_delta,
    )
    stage_summaries.append(growth_stage_summary)

    termination_reason = "single_growth_stage_only" if not strip_enabled else "strip_plus_growth_completed"
    current_best_run = growth_stage_summary["best_run_by_zr12_loss"]
    for followup_index in range(args.followup_stages):
        if current_best_run is None:
            termination_reason = f"stage{stage_index:02d}_produced_no_runs"
            break
        if current_best_run["end_zr12"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_index = len(stage_summaries) + 1
        next_seed_path = Path(current_best_run["pkl_path"])
        next_seed_counts = outer_stage.probe.seed_counts(next_seed_path)
        if next_seed_counts["Zr12_AA"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_summary = annotate_stage_summary(
            outer_stage.run_growth_stage(
                seed_path=next_seed_path,
                candidate=growth_candidate,
                run_dir=run_dir,
                run_name=run_name,
                stage_index=stage_index,
                replicates=followup_replicates,
                base_rng_seed=args.base_rng_seed + followup_index * 1000 + (2000 if strip_enabled else 1000),
                total_steps=args.growth_total_steps,
                max_entities_delta=args.growth_max_entities_delta,
                dissolution_update_interval_steps=args.dissolution_update_interval_steps,
                bumping_threshold=args.bumping_threshold,
                entropy_correction_coefficient=args.entropy_correction_coefficient,
            ),
            stage_kind="growth_followup",
            candidate=growth_candidate,
            total_steps=args.growth_total_steps,
            max_entities_delta=args.growth_max_entities_delta,
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
        "strip_enabled": strip_enabled,
        "strip_candidate": strip_candidate if strip_enabled else None,
        "growth_candidate": growth_candidate,
        "growth_config": {
            "replicates": args.replicates,
            "strip_replicates": strip_replicates if strip_enabled else None,
            "strip_total_steps": args.strip_total_steps if strip_enabled else None,
            "strip_max_entities_delta": args.strip_max_entities_delta if strip_enabled else None,
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
