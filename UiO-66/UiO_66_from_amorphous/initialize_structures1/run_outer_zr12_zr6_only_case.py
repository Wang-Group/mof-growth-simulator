import argparse
import contextlib
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import probe_zr6_only_growth as probe
from UiO66_Assembly_Large_Correction_conc import Assembly, Zr6_AA, safe_pickle_save


DEFAULT_GROWTH_CHEMISTRY = {
    "exchange_rxn_time_seconds": 0.1,
    "zr_conc": 5000.0,
    "linker_conc": 69.1596872253079,
    "capping_agent_conc": 300.0,
    "equilibrium_constant_coefficient": 6.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an outer-Zr12 mixed seed, then run long Zr6-only secondary growth "
            "to test whether the exposed Zr12 clusters detach while Zr6 continues to grow."
        )
    )
    parser.add_argument("--seed-rng-seed", type=int, default=2402)
    parser.add_argument("--core-target-entities", type=int, default=160)
    parser.add_argument("--shell-zr12-count", type=int, default=6)
    parser.add_argument("--attempts-per-step", type=int, default=1000)
    parser.add_argument("--seed-internal-link-probability", type=float, default=0.18)
    parser.add_argument(
        "--shell-anchor-min-radial-fraction",
        type=float,
        default=0.72,
        help="Only use outer-shell Zr6 anchors at or above this radial fraction.",
    )
    parser.add_argument(
        "--shell-zr12-max-coordination",
        type=int,
        default=4,
        help="Fail if any shell Zr12 is born above this coordination.",
    )
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
        default=0.70,
        help="Fail unless every Zr12 cluster remains on the outside of the mixed seed.",
    )
    parser.add_argument("--replicates", type=int, default=3)
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
    parser.add_argument("--base-rng-seed", type=int, default=18000)
    parser.add_argument("--growth-total-steps", type=int, default=20000)
    parser.add_argument("--growth-max-entities-delta", type=int, default=360)
    parser.add_argument(
        "--growth-exchange-rxn-time-seconds",
        type=float,
        default=DEFAULT_GROWTH_CHEMISTRY["exchange_rxn_time_seconds"],
    )
    parser.add_argument(
        "--growth-zr-conc",
        type=float,
        default=DEFAULT_GROWTH_CHEMISTRY["zr_conc"],
    )
    parser.add_argument(
        "--growth-linker-conc",
        type=float,
        default=DEFAULT_GROWTH_CHEMISTRY["linker_conc"],
    )
    parser.add_argument(
        "--growth-capping-agent-conc",
        type=float,
        default=DEFAULT_GROWTH_CHEMISTRY["capping_agent_conc"],
    )
    parser.add_argument(
        "--growth-equilibrium-constant-coefficient",
        type=float,
        default=DEFAULT_GROWTH_CHEMISTRY["equilibrium_constant_coefficient"],
    )
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=probe.RUN_DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=probe.RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=probe.RUN_DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    parser.add_argument("--bumping-threshold", type=float, default=probe.RUN_DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument("--output-dir", default="output/mixed_nuclei/outer_zr12_zr6only")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def resolve_output_dir(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def quiet_pickle_save(assembly, filepath):
    with contextlib.redirect_stdout(io.StringIO()):
        success = safe_pickle_save(
            assembly,
            filepath.as_posix(),
            clean_connected_entities=True,
            rebuild_after_save=False,
        )
    if not success:
        raise RuntimeError(f"Failed to save pickle to {filepath}")


def save_assembly_outputs(assembly, run_dir, stem):
    pkl_path = run_dir / f"{stem}.pkl"
    mol2_path = run_dir / f"{stem}.mol2"

    quiet_pickle_save(assembly, pkl_path)
    assembly.get_mol2_file(mol2_path.as_posix())
    return pkl_path, mol2_path


def zr12_rows_from_assembly(assembly):
    cluster_summary = core_builder.cluster_summary(assembly)
    return [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in cluster_summary
        if row["kind"] == "Zr12_AA"
    ]


def outer_anchor_rows(
    assembly,
    min_radial_fraction,
    anchor_usage,
    *,
    max_anchor_coordination=None,
    max_anchor_usage=None,
    sort_mode="radial_first",
):
    octant_center, octant_counts = core_builder.cluster_octant_populations(assembly)
    rows = []

    for row in core_builder.cluster_summary(assembly):
        if row["kind"] != "Zr6_AA":
            continue
        if row["radial_fraction"] < min_radial_fraction:
            continue
        usage = anchor_usage.get(id(row["entity_ref"]), 0)
        if max_anchor_coordination is not None and row["coordination"] > max_anchor_coordination:
            continue
        if max_anchor_usage is not None and usage >= max_anchor_usage:
            continue
        entity = row["entity_ref"]
        free_formates = core_builder.free_sites(
            assembly,
            carboxylate_type="formate",
            owner_entity=entity,
        )
        if not free_formates:
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
                "anchor_usage": usage,
                "octant_occupancy": occupancy,
            }
        )

    if sort_mode == "terminal_first":
        rows.sort(
            key=lambda row: (
                row["anchor_usage"],
                row["coordination"],
                row["octant_occupancy"],
                -row["radius_from_cluster_centroid"],
                random.random(),
            )
        )
    elif sort_mode == "sacrificial_first":
        rows.sort(
            key=lambda row: (
                row["anchor_usage"],
                row["coordination"],
                -row["radius_from_cluster_centroid"],
                row["octant_occupancy"],
                random.random(),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                row["anchor_usage"],
                row["octant_occupancy"],
                -row["radius_from_cluster_centroid"],
                row["coordination"],
                random.random(),
            )
        )
    return rows


def grow_outer_shell_zr12(
    assembly,
    shell_zr12_count,
    attempts_per_step,
    min_anchor_radial_fraction,
    *,
    max_anchor_coordination=None,
    max_anchor_usage=None,
    anchor_sort_mode="radial_first",
):
    shell_log = []
    anchor_usage = {}

    for shell_index in range(shell_zr12_count):
        candidate_rows = outer_anchor_rows(
            assembly,
            min_radial_fraction=min_anchor_radial_fraction,
            anchor_usage=anchor_usage,
            max_anchor_coordination=max_anchor_coordination,
            max_anchor_usage=max_anchor_usage,
            sort_mode=anchor_sort_mode,
        )
        if not candidate_rows:
            raise RuntimeError(
                "No outer-shell Zr6 anchors with free formate sites were available for Zr12 placement."
            )

        selected_row = None
        last_error = None
        for row in candidate_rows[:40]:
            try:
                linker, zr12 = core_builder.grow_cluster_from_anchor(
                    assembly,
                    row["entity_ref"],
                    "zr12",
                    attempts_per_step,
                )
                selected_row = row
                anchor_usage[id(row["entity_ref"])] = anchor_usage.get(id(row["entity_ref"]), 0) + 1
                shell_log.append(
                    {
                        "shell_index": shell_index,
                        "anchor_radial_fraction": row["radial_fraction"],
                        "anchor_radius": row["radius_from_cluster_centroid"],
                        "anchor_coordination_before": row["coordination"],
                        "anchor_octant_occupancy": row["octant_occupancy"],
                        "linker_center": np.asarray(linker.center, dtype=float).round(4).tolist(),
                        "zr12_center": np.asarray(zr12.center, dtype=float).round(4).tolist(),
                    }
                )
                break
            except Exception as exc:
                last_error = exc

        if selected_row is None:
            raise RuntimeError(
                f"Failed to attach shell Zr12 cluster {shell_index}. Last error: {last_error}"
            )

    return shell_log


def validate_outer_zr12_seed(
    assembly,
    *,
    expected_zr12_count,
    max_zr12_coordination,
    min_zr12_radial_fraction,
):
    counts = core_builder.entity_counts(assembly)
    summary = core_builder.cluster_summary(assembly)
    shape = core_builder.cluster_shape_metrics(assembly)
    zr12_rows = [row for row in summary if row["kind"] == "Zr12_AA"]

    if len(zr12_rows) != expected_zr12_count:
        raise RuntimeError(
            f"Expected {expected_zr12_count} shell Zr12 clusters, found {len(zr12_rows)}."
        )
    if any(row["coordination"] > max_zr12_coordination for row in zr12_rows):
        raise RuntimeError(
            f"At least one shell Zr12 cluster exceeded coordination {max_zr12_coordination}. "
            f"Rows: {zr12_rows}"
        )
    if any(row["radial_fraction"] < min_zr12_radial_fraction for row in zr12_rows):
        raise RuntimeError(
            f"At least one shell Zr12 cluster fell below radial_fraction {min_zr12_radial_fraction}. "
            f"Rows: {zr12_rows}"
        )

    return counts, summary, shape


def build_outer_zr12_seed(args):
    random.seed(args.seed_rng_seed)
    np.random.seed(args.seed_rng_seed)

    assembly = Assembly(Zr6_AA(), ZR6_PERCENTAGE=1.0, ENTROPY_GAIN=30.9, BUMPING_THRESHOLD=args.bumping_threshold)
    with contextlib.redirect_stdout(io.StringIO()):
        core_growth_snapshots = core_builder.grow_outer_shell_zr6_only(
            assembly=assembly,
            target_entities=args.core_target_entities,
            attempts_per_step=args.attempts_per_step,
            internal_link_probability=args.seed_internal_link_probability,
        )
    shell_log = grow_outer_shell_zr12(
        assembly,
        shell_zr12_count=args.shell_zr12_count,
        attempts_per_step=args.attempts_per_step,
        min_anchor_radial_fraction=args.shell_anchor_min_radial_fraction,
        max_anchor_coordination=args.shell_anchor_max_coordination,
        max_anchor_usage=args.shell_anchor_max_usage,
        anchor_sort_mode=args.shell_anchor_sort_mode,
    )
    counts, coordination, cluster_shape = validate_outer_zr12_seed(
        assembly,
        expected_zr12_count=args.shell_zr12_count,
        max_zr12_coordination=args.shell_zr12_max_coordination,
        min_zr12_radial_fraction=args.shell_zr12_min_final_radial_fraction,
    )

    zr12_rows = [
        {key: value for key, value in row.items() if key != "entity_ref"}
        for row in coordination
        if row["kind"] == "Zr12_AA"
    ]
    return assembly, {
        "seed_rng_seed": args.seed_rng_seed,
        "core_target_entities": args.core_target_entities,
        "shell_zr12_count": args.shell_zr12_count,
        "attempts_per_step": args.attempts_per_step,
        "seed_internal_link_probability": args.seed_internal_link_probability,
        "shell_anchor_min_radial_fraction": args.shell_anchor_min_radial_fraction,
        "shell_zr12_max_coordination": args.shell_zr12_max_coordination,
        "shell_anchor_max_coordination": args.shell_anchor_max_coordination,
        "shell_anchor_max_usage": args.shell_anchor_max_usage,
        "shell_anchor_sort_mode": args.shell_anchor_sort_mode,
        "shell_zr12_min_final_radial_fraction": args.shell_zr12_min_final_radial_fraction,
        "counts": counts,
        "cluster_shape": cluster_shape,
        "core_growth_snapshots": core_growth_snapshots,
        "shell_build_log": shell_log,
        "zr12_rows": zr12_rows,
    }


def candidate_from_args(args):
    _, formate_ratio = probe.dissolution_probability(
        0.0,
        args.growth_equilibrium_constant_coefficient,
        args.h2o_dmf_ratio,
        args.growth_capping_agent_conc,
        args.growth_linker_conc,
    )
    return {
        "label": "outer_shell_zr12_case",
        "exchange_rxn_time_seconds": args.growth_exchange_rxn_time_seconds,
        "zr_conc": args.growth_zr_conc,
        "linker_conc": args.growth_linker_conc,
        "capping_agent_conc": args.growth_capping_agent_conc,
        "equilibrium_constant_coefficient": args.growth_equilibrium_constant_coefficient,
        "cluster_add_probability": probe.zr6_cluster_add_probability(
            args.growth_zr_conc,
            args.growth_linker_conc,
            zr6_percentage=1.0,
        ),
        "formate_benzoate_ratio_t0": formate_ratio,
    }


def summarize_best_run(run_payloads):
    if not run_payloads:
        return None
    ordered = sorted(
        run_payloads,
        key=lambda row: (
            row["end_counts"]["Zr12_AA"],
            row["delta_zr12"],
            -row["delta_zr6"],
            -row["delta_bdc"],
        ),
    )
    best = ordered[0]
    return {
        "seed_name": best["seed_name"],
        "rng_seed": best["rng_seed"],
        "delta_zr6": best["delta_zr6"],
        "delta_zr12": best["delta_zr12"],
        "delta_bdc": best["delta_bdc"],
        "start_zr12": best["start_counts"]["Zr12_AA"],
        "end_zr12": best["end_counts"]["Zr12_AA"],
        "zr12_remaining_fraction": best["zr12_remaining_fraction"],
        "pkl_path": best["pkl_path"],
        "mol2_path": best["mol2_path"],
        "json_path": best["json_path"],
    }


def run_growth_stage(
    *,
    seed_path,
    candidate,
    run_dir,
    run_name,
    stage_index,
    replicates,
    base_rng_seed,
    total_steps,
    max_entities_delta,
    dissolution_update_interval_steps,
    bumping_threshold,
    entropy_correction_coefficient,
):
    seed_path = Path(seed_path)
    stage_label = f"stage{stage_index:02d}"
    stage_seed_counts = probe.seed_counts(seed_path)
    entropy_table = probe.build_entropy_table(
        stage_seed_counts["total_entities"] + max_entities_delta + 50,
        entropy_correction_coefficient,
    )

    run_payloads = []
    for replicate_index in range(replicates):
        rng_seed = base_rng_seed + replicate_index
        assembly, run_result = probe.simulate_growth_case(
            seed_path=seed_path,
            candidate=candidate,
            entropy_table=entropy_table,
            total_steps=total_steps,
            max_entities_delta=max_entities_delta,
            dissolution_update_interval_steps=dissolution_update_interval_steps,
            bumping_threshold=bumping_threshold,
            rng_seed=rng_seed,
        )

        run_stem = f"{run_name}__{stage_label}__rep{replicate_index + 1:02d}__seed{rng_seed}"
        pkl_path, mol2_path = save_assembly_outputs(assembly, run_dir, run_stem)
        json_path = run_dir / f"{run_stem}.json"

        run_payload = {
            **run_result,
            "stage_index": stage_index,
            "stage_label": stage_label,
            "seed_path": seed_path.as_posix(),
            "candidate": candidate,
            "pkl_path": pkl_path.as_posix(),
            "mol2_path": mol2_path.as_posix(),
            "json_path": json_path.as_posix(),
            "shape_metrics": core_builder.cluster_shape_metrics(assembly),
            "zr12_rows": zr12_rows_from_assembly(assembly),
        }
        json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        run_payloads.append(run_payload)

        print(
            json.dumps(
                {
                    "stage": stage_label,
                    "replicate": replicate_index + 1,
                    "rng_seed": rng_seed,
                    "seed_zr12": stage_seed_counts["Zr12_AA"],
                    "end_zr12": run_result["end_counts"]["Zr12_AA"],
                    "delta_zr6": run_result["delta_zr6"],
                    "delta_zr12": run_result["delta_zr12"],
                    "delta_bdc": run_result["delta_bdc"],
                    "pkl_path": pkl_path.as_posix(),
                }
            )
        )

    return {
        "stage_index": stage_index,
        "stage_label": stage_label,
        "stage_seed_path": seed_path.as_posix(),
        "stage_seed_counts": stage_seed_counts,
        "replicates": replicates,
        "base_rng_seed": base_rng_seed,
        "total_steps": total_steps,
        "max_entities_delta": max_entities_delta,
        "runs": run_payloads,
        "best_run_by_zr12_loss": summarize_best_run(run_payloads),
    }


def main():
    args = parse_args()
    output_root = resolve_output_dir(args.output_dir)
    run_name = args.basename or f"outer_zr12_zr6only_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_assembly, seed_summary = build_outer_zr12_seed(args)
    seed_stem = f"{run_name}__seed"
    seed_pkl_path, seed_mol2_path = save_assembly_outputs(seed_assembly, run_dir, seed_stem)
    seed_json_path = run_dir / f"{seed_stem}.json"
    seed_json_path.write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")

    candidate = candidate_from_args(args)

    followup_replicates = args.followup_replicates if args.followup_replicates is not None else args.replicates
    stage_summaries = [
        run_growth_stage(
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
        next_seed_counts = probe.seed_counts(next_seed_path)
        if next_seed_counts["Zr12_AA"] <= 0:
            termination_reason = "zr12_fully_removed"
            break

        stage_summary = run_growth_stage(
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
        "best_run_by_zr12_loss": summarize_best_run(all_run_payloads),
    }

    summary_path = run_dir / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "seed_zr12_count": seed_summary["counts"]["Zr12_AA"],
                "seed_zr6_count": seed_summary["counts"]["Zr6_AA"],
                "termination_reason": termination_reason,
                "best_run": summary["best_run_by_zr12_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
