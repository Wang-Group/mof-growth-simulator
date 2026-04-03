import json
import math
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
INIT_DIR = CASE_DIR.parent
if INIT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, INIT_DIR.as_posix())

import build_internal_zr12_seed as core_builder
import fragment_cleanup
import probe_zr6_only_growth as probe
import run_outer_zr12_zr6_only_case as outer_stage


DEFAULT_SEED_PKL = (
    INIT_DIR
    / "output"
    / "mixed_nuclei"
    / "two_phase_amorphous_equilibrate_zr6only"
    / "two_phase_eqbond_zr6only_default"
    / "two_phase_eqbond_zr6only_default__phase1_seed.pkl"
)
DEFAULT_OUTPUT_DIR = (
    INIT_DIR
    / "output"
    / "mixed_nuclei"
    / "two_phase_size_stabilized_zr6only"
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Run a size-stabilized Zr6-only follow-up chain from a mixed phase-1 seed. "
            "External Zr6/BDC addition is still allowed, but unlink probability is dynamically "
            "adjusted to keep total size near a target."
        )
    )
    parser.add_argument("--seed-pkl", default=DEFAULT_SEED_PKL.as_posix())
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--base-rng-seed", type=int, default=105000)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--snapshot-interval", type=int, default=5000)
    parser.add_argument("--control-update-interval", type=int, default=10)
    parser.add_argument("--target-total-entities", type=int, default=210)
    parser.add_argument("--size-feedback-gain", type=float, default=20.0)
    parser.add_argument("--size-deadband-fraction", type=float, default=0.0)
    parser.add_argument("--min-formate-ratio", type=float, default=0.05)
    parser.add_argument("--max-formate-ratio", type=float, default=200.0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--zr-conc", type=float, default=1200.0)
    parser.add_argument("--linker-conc", type=float, default=69.1596872253079)
    parser.add_argument("--capping-agent-conc", type=float, default=180.0)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=1.3)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.0)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=probe.RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
    )
    parser.add_argument("--basename", default="size_stabilized_phase1seed_target210_pruned")
    return parser.parse_args()


def dynamic_formate_ratio(
    *,
    current_total_entities,
    target_total_entities,
    base_formate_ratio,
    size_feedback_gain,
    size_deadband_fraction,
    min_formate_ratio,
    max_formate_ratio,
):
    error_fraction = (current_total_entities - target_total_entities) / max(target_total_entities, 1)
    if abs(error_fraction) <= size_deadband_fraction:
        return base_formate_ratio, error_fraction
    multiplier = math.exp(size_feedback_gain * error_fraction)
    controlled_ratio = base_formate_ratio * multiplier
    controlled_ratio = min(max(controlled_ratio, min_formate_ratio), max_formate_ratio)
    return controlled_ratio, error_fraction


def summarize_state(assembly):
    counts = core_builder.entity_counts(assembly)
    zr12_summary = probe.summarize_zr12_coordination(assembly)
    return {
        **counts,
        "zr12_coordination": zr12_summary,
        "shape_metrics": core_builder.cluster_shape_metrics(assembly),
    }


def safe_delta(end_value, start_value):
    if end_value is None or start_value is None:
        return None
    return end_value - start_value


def safe_mean(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def simulate_size_stabilized_chain(
    *,
    assembly,
    candidate,
    total_steps,
    control_update_interval,
    snapshot_interval,
    target_total_entities,
    size_feedback_gain,
    size_deadband_fraction,
    min_formate_ratio,
    max_formate_ratio,
    entropy_table,
    rng_seed,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    start_state = summarize_state(assembly)
    timing = 0.0
    step = 0
    event_counts = {
        "link": 0,
        "grow_attempt": 0,
        "grow_success": 0,
        "grow_fail": 0,
        "remove": 0,
        "zr6_add": 0,
        "bdc_add": 0,
        "fragment_prune": 0,
        "fragment_pruned_entities": 0,
        "fragment_pruned_components": 0,
    }
    snapshots = []

    current_formate_ratio = candidate["base_formate_benzoate_ratio_t0"]
    current_size_error_fraction = 0.0
    total_entity_trace = []

    for cycle in range(total_steps + 1):
        current_total_entities = len(assembly.entities)
        total_entity_trace.append(current_total_entities)
        sim_time_seconds = timing * candidate["exchange_rxn_time_seconds"]

        if step == 0 or (control_update_interval > 0 and step % control_update_interval == 0):
            current_formate_ratio, current_size_error_fraction = dynamic_formate_ratio(
                current_total_entities=current_total_entities,
                target_total_entities=target_total_entities,
                base_formate_ratio=candidate["base_formate_benzoate_ratio_t0"],
                size_feedback_gain=size_feedback_gain,
                size_deadband_fraction=size_deadband_fraction,
                min_formate_ratio=min_formate_ratio,
                max_formate_ratio=max_formate_ratio,
            )

        if snapshot_interval > 0 and step % snapshot_interval == 0:
            snapshots.append(
                {
                    "step": step,
                    "sim_time_seconds": sim_time_seconds,
                    "controlled_formate_ratio": current_formate_ratio,
                    "size_error_fraction": current_size_error_fraction,
                    **summarize_state(assembly),
                }
            )

        step += 1
        flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(
            current_formate_ratio,
            entropy_table,
            candidate["cluster_add_probability"],
        )

        if flag == 0:
            assembly.link_internal_carboxylate(selected_pair)
            event_counts["link"] += 1
        elif flag == 1:
            before_entities = len(assembly.entities)
            event_counts["grow_attempt"] += 1
            carboxylate_type = selected_carboxylate.carboxylate_type
            assembly.grow_one_step(selected_carboxylate)
            after_entities = len(assembly.entities)
            if after_entities == before_entities + 1:
                event_counts["grow_success"] += 1
                if carboxylate_type == "benzoate":
                    event_counts["zr6_add"] += 1
                else:
                    event_counts["bdc_add"] += 1
            else:
                event_counts["grow_fail"] += 1
        elif flag == -1:
            assembly.remove_linkage(selected_pair)
            event_counts["remove"] += 1
            prune_stats = fragment_cleanup.prune_disconnected_fragments(assembly)
            if prune_stats["removed_entity_count"] > 0:
                event_counts["fragment_prune"] += 1
                event_counts["fragment_pruned_entities"] += prune_stats["removed_entity_count"]
                event_counts["fragment_pruned_components"] += prune_stats["removed_component_count"]

        random_draw = max(random.random(), 1e-12)
        timing -= math.log(random_draw) / max(total_growth_rate, 1e-12)

    end_state = summarize_state(assembly)
    if not snapshots or snapshots[-1]["step"] != step:
        snapshots.append(
            {
                "step": step,
                "sim_time_seconds": timing * candidate["exchange_rxn_time_seconds"],
                "controlled_formate_ratio": current_formate_ratio,
                "size_error_fraction": current_size_error_fraction,
                **end_state,
            }
        )

    size_trace = np.array(total_entity_trace, dtype=float)
    result = {
        "rng_seed": rng_seed,
        "steps_completed": step,
        "sim_time_seconds": timing * candidate["exchange_rxn_time_seconds"],
        "start_state": start_state,
        "end_state": end_state,
        "delta_zr6": end_state["Zr6_AA"] - start_state["Zr6_AA"],
        "delta_zr12": end_state["Zr12_AA"] - start_state["Zr12_AA"],
        "delta_bdc": end_state["BDC"] - start_state["BDC"],
        "delta_total_entities": end_state["total_entities"] - start_state["total_entities"],
        "delta_linked_pairs": end_state["linked_pairs"] - start_state["linked_pairs"],
        "delta_ready_pairs": end_state["ready_pairs"] - start_state["ready_pairs"],
        "delta_zr12_coord_mean": safe_delta(
            end_state["zr12_coordination"]["coord_mean"],
            start_state["zr12_coordination"]["coord_mean"],
        ),
        "event_counts": event_counts,
        "controlled_formate_ratio_final": current_formate_ratio,
        "size_error_fraction_final": current_size_error_fraction,
        "size_trace_summary": {
            "target_total_entities": target_total_entities,
            "min_total_entities": int(size_trace.min()),
            "mean_total_entities": float(size_trace.mean()),
            "max_total_entities": int(size_trace.max()),
            "std_total_entities": float(size_trace.std()),
        },
        "snapshots": snapshots,
    }
    return assembly, result


def aggregate_runs(run_payloads):
    avg_mean_size = float(np.mean([row["size_trace_summary"]["mean_total_entities"] for row in run_payloads]))
    representative = sorted(
        run_payloads,
        key=lambda row: (
            abs(row["size_trace_summary"]["mean_total_entities"] - avg_mean_size),
            abs(row["delta_total_entities"]),
            row["end_state"]["Zr12_AA"],
            -row["delta_zr6"],
        ),
    )[0]
    aggregate = {
        "mean_total_entities_mean": avg_mean_size,
        "mean_total_entities_std": float(
            np.std([row["size_trace_summary"]["mean_total_entities"] for row in run_payloads])
        ),
        "final_total_entities_mean": float(np.mean([row["end_state"]["total_entities"] for row in run_payloads])),
        "final_zr12_mean": float(np.mean([row["end_state"]["Zr12_AA"] for row in run_payloads])),
        "final_zr6_mean": float(np.mean([row["end_state"]["Zr6_AA"] for row in run_payloads])),
        "final_bdc_mean": float(np.mean([row["end_state"]["BDC"] for row in run_payloads])),
        "final_zr12_coord_mean_mean": safe_mean(
            [row["end_state"]["zr12_coordination"]["coord_mean"] for row in run_payloads]
        ),
        "delta_total_entities_mean": float(np.mean([row["delta_total_entities"] for row in run_payloads])),
        "delta_total_entities_std": float(np.std([row["delta_total_entities"] for row in run_payloads])),
        "delta_zr12_mean": float(np.mean([row["delta_zr12"] for row in run_payloads])),
        "delta_zr6_mean": float(np.mean([row["delta_zr6"] for row in run_payloads])),
        "delta_bdc_mean": float(np.mean([row["delta_bdc"] for row in run_payloads])),
    }
    return aggregate, representative


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"size_stabilized_phase1seed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(args.seed_pkl)
    if not seed_path.is_absolute():
        seed_path = (INIT_DIR / seed_path).resolve()

    base_formate_ratio_t0 = probe.dissolution_probability(
        0.0,
        args.equilibrium_constant_coefficient,
        args.h2o_dmf_ratio,
        args.capping_agent_conc,
        args.linker_conc,
    )[1]
    candidate = {
        "label": "size_stabilized_zr6_only",
        "exchange_rxn_time_seconds": args.exchange_rxn_time_seconds,
        "zr_conc": args.zr_conc,
        "linker_conc": args.linker_conc,
        "capping_agent_conc": args.capping_agent_conc,
        "equilibrium_constant_coefficient": args.equilibrium_constant_coefficient,
        "cluster_add_probability": probe.zr6_cluster_add_probability(
            args.zr_conc,
            args.linker_conc,
            zr6_percentage=1.0,
        ),
        "base_formate_benzoate_ratio_t0": base_formate_ratio_t0,
    }

    initial_assembly = probe.load_seed_assembly(
        seed_path,
        zr6_percentage=1.0,
        bumping_threshold=probe.RUN_DEFAULTS["BUMPING_THRESHOLD"],
    )
    start_total_entities = core_builder.entity_counts(initial_assembly)["total_entities"]
    entropy_table = probe.build_entropy_table(
        start_total_entities + 2000,
        args.entropy_correction_coefficient,
    )

    run_payloads = []
    for replicate_index in range(args.replicates):
        rng_seed = args.base_rng_seed + replicate_index
        assembly = probe.load_seed_assembly(
            seed_path,
            zr6_percentage=1.0,
            bumping_threshold=probe.RUN_DEFAULTS["BUMPING_THRESHOLD"],
        )
        assembly, result = simulate_size_stabilized_chain(
            assembly=assembly,
            candidate=candidate,
            total_steps=args.total_steps,
            control_update_interval=args.control_update_interval,
            snapshot_interval=args.snapshot_interval,
            target_total_entities=args.target_total_entities,
            size_feedback_gain=args.size_feedback_gain,
            size_deadband_fraction=args.size_deadband_fraction,
            min_formate_ratio=args.min_formate_ratio,
            max_formate_ratio=args.max_formate_ratio,
            entropy_table=entropy_table,
            rng_seed=rng_seed,
        )

        run_stem = f"{run_name}__rep{replicate_index + 1:02d}__seed{rng_seed}"
        pkl_path = run_dir / f"{run_stem}.pkl"
        json_path = run_dir / f"{run_stem}.json"
        outer_stage.quiet_pickle_save(assembly, pkl_path)

        payload = {
            **result,
            "candidate": candidate,
            "seed_path": seed_path.as_posix(),
            "pkl_path": pkl_path.as_posix(),
            "json_path": json_path.as_posix(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_payloads.append(payload)

        print(
            json.dumps(
                {
                    "replicate": replicate_index + 1,
                    "rng_seed": rng_seed,
                    "mean_total_entities": payload["size_trace_summary"]["mean_total_entities"],
                    "final_total_entities": payload["end_state"]["total_entities"],
                    "final_zr12": payload["end_state"]["Zr12_AA"],
                    "delta_zr6": payload["delta_zr6"],
                    "delta_zr12": payload["delta_zr12"],
                    "json_path": payload["json_path"],
                }
            )
        )

    aggregate, representative = aggregate_runs(run_payloads)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "seed_path": seed_path.as_posix(),
        "candidate": candidate,
        "control": {
            "target_total_entities": args.target_total_entities,
            "control_update_interval": args.control_update_interval,
            "size_feedback_gain": args.size_feedback_gain,
            "size_deadband_fraction": args.size_deadband_fraction,
            "min_formate_ratio": args.min_formate_ratio,
            "max_formate_ratio": args.max_formate_ratio,
        },
        "config": {
            "replicates": args.replicates,
            "base_rng_seed": args.base_rng_seed,
            "total_steps": args.total_steps,
            "snapshot_interval": args.snapshot_interval,
        },
        "runs": run_payloads,
        "aggregate": aggregate,
        "representative_run": {
            "rng_seed": representative["rng_seed"],
            "json_path": representative["json_path"],
            "pkl_path": representative["pkl_path"],
        },
    }
    summary_path = run_dir / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": summary_path.as_posix(),
                "aggregate": aggregate,
                "representative_run": summary["representative_run"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
