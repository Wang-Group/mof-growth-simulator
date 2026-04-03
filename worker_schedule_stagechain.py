import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy.core.numeric as numpy_core_numeric

import run_outer_zr12_zr6_only_case as stage_runner


# Older seeds in this workspace were pickled against numpy's private module path.
sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a temporary staged schedule starting from an existing seed pickle. "
            "The plan is a JSON array of stage specs."
        )
    )
    parser.add_argument("--start-seed-pkl", required=True, help="Seed pickle to start from.")
    parser.add_argument("--plan", help="JSON array describing the stage schedule.")
    parser.add_argument("--plan-file", help="Path to a JSON file with the stage schedule.")
    parser.add_argument("--output-dir", default="output/mixed_nuclei/worker_schedule")
    parser.add_argument("--basename", default=None)
    parser.add_argument(
        "--default-replicates",
        type=int,
        default=4,
        help="Fallback replicate count for stages that do not specify one.",
    )
    parser.add_argument(
        "--default-steps",
        type=int,
        default=20000,
        help="Fallback total steps for stages that do not specify one.",
    )
    parser.add_argument(
        "--default-max-entities-delta",
        type=int,
        default=360,
        help="Fallback max_entities_delta for stages that do not specify one.",
    )
    parser.add_argument(
        "--default-bumping-threshold",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["BUMPING_THRESHOLD"],
    )
    parser.add_argument(
        "--default-entropy-correction-coefficient",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument(
        "--default-h2o-dmf-ratio",
        type=float,
        default=stage_runner.probe.RUN_DEFAULTS["H2O_DMF_RATIO"],
    )
    parser.add_argument(
        "--default-dissolution-update-interval-steps",
        type=int,
        default=stage_runner.probe.RUN_DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    return parser.parse_args()


def resolve_output_dir(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_plan(args):
    if args.plan and args.plan_file:
        raise ValueError("Use either --plan or --plan-file, not both.")
    if not args.plan and not args.plan_file:
        raise ValueError("A stage plan is required.")

    if args.plan_file:
        plan_path = Path(args.plan_file)
        if not plan_path.is_absolute():
            plan_path = (Path(__file__).resolve().parent / plan_path).resolve()
        return json.loads(plan_path.read_text(encoding="utf-8"))
    return json.loads(args.plan)


def build_candidate(spec, *, default_h2o_dmf_ratio):
    label = spec.get("label") or spec["kind"]
    exchange_rxn_time_seconds = spec.get("exchange_rxn_time_seconds", 0.1)
    zr_conc = spec.get("zr_conc", stage_runner.DEFAULT_GROWTH_CHEMISTRY["zr_conc"])
    linker_conc = spec.get("linker_conc", stage_runner.DEFAULT_GROWTH_CHEMISTRY["linker_conc"])
    capping_agent_conc = spec.get(
        "capping_agent_conc",
        stage_runner.DEFAULT_GROWTH_CHEMISTRY["capping_agent_conc"],
    )
    equilibrium_constant_coefficient = spec.get(
        "equilibrium_constant_coefficient",
        stage_runner.DEFAULT_GROWTH_CHEMISTRY["equilibrium_constant_coefficient"],
    )
    h2o_dmf_ratio = spec.get("h2o_dmf_ratio", default_h2o_dmf_ratio)
    _, formate_ratio = stage_runner.probe.dissolution_probability(
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
        "cluster_add_probability": stage_runner.probe.zr6_cluster_add_probability(
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
    plan = load_plan(args)

    output_root = resolve_output_dir(args.output_dir)
    run_name = args.basename or f"worker_schedule_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    current_seed_path = Path(args.start_seed_pkl)
    if not current_seed_path.is_absolute():
        current_seed_path = (Path(__file__).resolve().parent / current_seed_path).resolve()

    stage_summaries = []
    for stage_index, spec in enumerate(plan, start=1):
        stage_kind = spec["kind"]
        candidate = build_candidate(spec, default_h2o_dmf_ratio=args.default_h2o_dmf_ratio)
        stage_summary = annotate_stage_summary(
            stage_runner.run_growth_stage(
                seed_path=current_seed_path,
                candidate=candidate,
                run_dir=run_dir,
                run_name=run_name,
                stage_index=stage_index,
                replicates=spec.get("replicates", args.default_replicates),
                base_rng_seed=spec.get("base_rng_seed", 47000 + (stage_index - 1) * 1000),
                total_steps=spec.get("total_steps", args.default_steps),
                max_entities_delta=spec.get("max_entities_delta", args.default_max_entities_delta),
                dissolution_update_interval_steps=spec.get(
                    "dissolution_update_interval_steps",
                    args.default_dissolution_update_interval_steps,
                ),
                bumping_threshold=spec.get("bumping_threshold", args.default_bumping_threshold),
                entropy_correction_coefficient=spec.get(
                    "entropy_correction_coefficient",
                    args.default_entropy_correction_coefficient,
                ),
            ),
            stage_kind=stage_kind,
            candidate=candidate,
            total_steps=spec.get("total_steps", args.default_steps),
            max_entities_delta=spec.get("max_entities_delta", args.default_max_entities_delta),
        )
        stage_summaries.append(stage_summary)

        best_run = stage_summary["best_run_by_zr12_loss"]
        if best_run is None:
            break
        current_seed_path = Path(best_run["pkl_path"])
        if best_run["end_zr12"] <= 0:
            break

    all_run_payloads = [run_payload for stage in stage_summaries for run_payload in stage["runs"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "start_seed_pkl": current_seed_path.as_posix(),
        "plan": plan,
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
                "best_run": summary["best_run_by_zr12_loss"],
                "stage_end_zr12": [
                    {
                        "stage": stage["stage_label"],
                        "kind": stage["stage_kind"],
                        "best_end_zr12": stage["best_run_by_zr12_loss"]["end_zr12"]
                        if stage["best_run_by_zr12_loss"]
                        else None,
                    }
                    for stage in stage_summaries
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
