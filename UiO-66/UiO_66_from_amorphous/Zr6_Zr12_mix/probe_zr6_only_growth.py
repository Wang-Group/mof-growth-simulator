import argparse
import contextlib
import io
import json
import math
import random
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import build_internal_zr12_seed as core_builder
import fragment_cleanup
from UiO66_Assembly_Large_Correction_conc import Assembly, safe_pickle_load


RUN_DEFAULTS = {
    "ZR6_PERCENTAGE": 1.0,
    "entropy_correction_coefficient": 0.789387907185137,
    "equilibrium_constant_coefficient": 1.32975172557788,
    "H2O_DMF_RATIO": 3e-10,
    "Capping_agent_conc": 1473.06341756944,
    "Linker_conc": 69.1596872253079,
    "Zr_conc": 22.9154705859894,
    "BUMPING_THRESHOLD": 2.0,
    "EXCHANGE_RXN_TIME_SECONDS": 0.1,
    "DISSOLUTION_UPDATE_INTERVAL_STEPS": None,
}

CORRECTION_TERM_FOR_DEPROTONATION = 10 ** (3.51 - 4.74)
EQUILIBRIUM_CONSTANT = 1.64
H2O_PURE = 55500.0
DMF_PURE = 12900.0
H2O_FORMATE_COEFFICIENT = 0.01
DMF_FORMATE_COEFFICIENT = 0.01
NUM_CARBOXYLATE_ON_LINKER = 2
END_DMF_DECOMPOSITION_CONC = 560.0
EXPERIMENT_TIME_HOURS = 3.0
DMF_DECOMPOSITION_RATE = END_DMF_DECOMPOSITION_CONC / (EXPERIMENT_TIME_HOURS * 3.6 * 10 ** 3)
ENTROPY_GAIN = 30.9


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_seed_dir = script_dir / "output" / "mixed_nuclei"
    parser = argparse.ArgumentParser(
        description=(
            "Probe Zr6-only growth conditions for mixed Zr6/Zr12 seeds. "
            "The script uses the repository's Assembly event logic but replaces the "
            "SciPy root solve with a monotonic binary-search solver."
        )
    )
    parser.add_argument(
        "--seed-pkls",
        nargs="+",
        default=[
            str(default_seed_dir / "ratio_controlled_mixed_seed_154_seed612_frac055_fill4.pkl"),
            str(default_seed_dir / "ratio_controlled_mixed_seed_160_seed612_frac055_fill6.pkl"),
        ],
        help="Seed PKL files to probe.",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--base-rng-seed", type=int, default=9100)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument(
        "--max-entities-delta",
        type=int,
        default=20,
        help="Stop a trajectory once seed_size + max_entities_delta is exceeded.",
    )
    parser.add_argument(
        "--exchange-rxn-time-seconds",
        default="0.1,1.0",
        help="Comma-separated exchange times to test.",
    )
    parser.add_argument(
        "--zr-conc-values",
        default="500,1000,2000,5000",
        help="Comma-separated Zr concentrations (mM).",
    )
    parser.add_argument(
        "--linker-conc-values",
        default="69.1596872253079,100,150",
        help="Comma-separated linker concentrations (mM).",
    )
    parser.add_argument(
        "--capping-agent-conc-values",
        default="50,100,200,400",
        help="Comma-separated capping-agent concentrations (mM).",
    )
    parser.add_argument(
        "--equilibrium-constant-coefficient-values",
        default="6,10,20",
        help="Comma-separated equilibrium-coefficient values.",
    )
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=RUN_DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument("--bumping-threshold", type=float, default=RUN_DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=RUN_DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    parser.add_argument(
        "--min-cluster-add-probability",
        type=float,
        default=0.12,
        help="Analytic prefilter lower bound.",
    )
    parser.add_argument(
        "--min-formate-benzoate-ratio",
        type=float,
        default=0.02,
        help="Analytic prefilter lower bound.",
    )
    parser.add_argument(
        "--max-formate-benzoate-ratio",
        type=float,
        default=0.35,
        help="Analytic prefilter upper bound.",
    )
    parser.add_argument(
        "--target-formate-benzoate-ratio",
        type=float,
        default=0.10,
        help="Candidates near this ratio are prioritized after filtering.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum number of analytic candidates to simulate, excluding the baseline control.",
    )
    parser.add_argument(
        "--include-baseline-control",
        action="store_true",
        help="Also simulate the default seeded-growth chemistry as a control.",
    )
    parser.add_argument("--output-dir", default="output/mixed_nuclei/probes")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def logistic_from_log_ratio(log_ratio):
    if log_ratio >= 0:
        exp_term = math.exp(-log_ratio)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(log_ratio)
    return exp_term / (1.0 + exp_term)


def zr6_cluster_add_probability(zr_conc, linker_conc, zr6_percentage=1.0):
    zr6_conc = zr_conc * zr6_percentage / 6.0
    zr6_conc_reference = linker_conc * NUM_CARBOXYLATE_ON_LINKER / 12.0
    if zr6_conc <= 0 or zr6_conc_reference <= 0:
        return 0.0
    return logistic_from_log_ratio(-3.6 + math.log(zr6_conc / zr6_conc_reference))


def effective_equilibrium_constant(
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
):
    dmf_conc = DMF_PURE / (1.0 + h2o_dmf_ratio)
    h2o_conc = H2O_PURE * h2o_dmf_ratio / (1.0 + h2o_dmf_ratio)
    h2o_power = h2o_conc / capping_agent_conc * H2O_FORMATE_COEFFICIENT
    dmf_power = dmf_conc / capping_agent_conc * DMF_FORMATE_COEFFICIENT
    return equilibrium_constant_coefficient * EQUILIBRIUM_CONSTANT / (1.0 + h2o_power + dmf_power)


def deprotonation_balance(y_value, capping_agent_conc, linker_conc):
    left_formate = (
        y_value
        * CORRECTION_TERM_FOR_DEPROTONATION
        / (1.0 + y_value * CORRECTION_TERM_FOR_DEPROTONATION)
        * capping_agent_conc
    )
    left_linker = y_value / (1.0 + y_value) * linker_conc * NUM_CARBOXYLATE_ON_LINKER
    return left_formate + left_linker


def solve_linker_carboxylate_to_acid_ratio(dimethylamine_conc, capping_agent_conc, linker_conc):
    if dimethylamine_conc <= 0:
        return 0.0

    high = 1.0
    while deprotonation_balance(high, capping_agent_conc, linker_conc) < dimethylamine_conc:
        high *= 2.0
        if high > 1e16:
            break

    low = 0.0
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        if deprotonation_balance(midpoint, capping_agent_conc, linker_conc) < dimethylamine_conc:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def dissolution_probability(
    time_passed_seconds,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    linker_conc,
):
    dimethylamine_conc = min(time_passed_seconds * DMF_DECOMPOSITION_RATE, END_DMF_DECOMPOSITION_CONC)
    linker_ratio = solve_linker_carboxylate_to_acid_ratio(
        dimethylamine_conc,
        capping_agent_conc,
        linker_conc,
    )
    formate_ratio = linker_ratio * CORRECTION_TERM_FOR_DEPROTONATION
    linker_carboxylic_acid_conc = linker_conc * (1.0 / (1.0 + linker_ratio)) * NUM_CARBOXYLATE_ON_LINKER
    formic_acid_conc = capping_agent_conc * (1.0 / (1.0 + formate_ratio))
    effective_equilibrium = effective_equilibrium_constant(
        equilibrium_constant_coefficient,
        h2o_dmf_ratio,
        capping_agent_conc,
    )
    formate_benzoate_ratio = formic_acid_conc / linker_carboxylic_acid_conc / effective_equilibrium
    probability = formate_benzoate_ratio / (1.0 + formate_benzoate_ratio)
    return probability, formate_benzoate_ratio


def entropy_assembly(num_entity, entropy_correction_coefficient, limit=150):
    corrected_entropy_gain = ENTROPY_GAIN * entropy_correction_coefficient
    entity_extra_gain = 0.35
    if num_entity >= limit:
        return math.exp(corrected_entropy_gain)
    return math.exp(
        corrected_entropy_gain
        + entity_extra_gain * (1.0 - math.log(num_entity + 1.0) / math.log(limit))
    )


def build_entropy_table(max_entities, entropy_correction_coefficient):
    return [
        entropy_assembly(num_entity, entropy_correction_coefficient)
        for num_entity in range(max_entities + 5)
    ]


def resolve_seed_paths(seed_pkls):
    script_dir = Path(__file__).resolve().parent
    resolved = []
    for raw_path in seed_pkls:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (script_dir / path).resolve()
        resolved.append(path)
    return resolved


def seed_counts(seed_path):
    with contextlib.redirect_stdout(io.StringIO()):
        assembly_loaded = safe_pickle_load(seed_path.as_posix(), rebuild_references=True)
    return core_builder.entity_counts(assembly_loaded)


def load_seed_assembly(seed_path, zr6_percentage, bumping_threshold):
    with contextlib.redirect_stdout(io.StringIO()):
        assembly_loaded = safe_pickle_load(seed_path.as_posix(), rebuild_references=True)
    return Assembly(assembly_loaded, zr6_percentage, ENTROPY_GAIN, bumping_threshold)


def analytic_candidate_rows(args):
    rows = []
    for exchange_time, zr_conc, linker_conc, capping_agent_conc, equilibrium_coeff in product(
        parse_csv_floats(args.exchange_rxn_time_seconds),
        parse_csv_floats(args.zr_conc_values),
        parse_csv_floats(args.linker_conc_values),
        parse_csv_floats(args.capping_agent_conc_values),
        parse_csv_floats(args.equilibrium_constant_coefficient_values),
    ):
        cluster_probability = zr6_cluster_add_probability(zr_conc, linker_conc, zr6_percentage=1.0)
        _, formate_ratio = dissolution_probability(
            0.0,
            equilibrium_coeff,
            args.h2o_dmf_ratio,
            capping_agent_conc,
            linker_conc,
        )
        if cluster_probability < args.min_cluster_add_probability:
            continue
        if formate_ratio < args.min_formate_benzoate_ratio:
            continue
        if formate_ratio > args.max_formate_benzoate_ratio:
            continue
        rows.append(
            {
                "exchange_rxn_time_seconds": exchange_time,
                "zr_conc": zr_conc,
                "linker_conc": linker_conc,
                "capping_agent_conc": capping_agent_conc,
                "equilibrium_constant_coefficient": equilibrium_coeff,
                "cluster_add_probability": cluster_probability,
                "formate_benzoate_ratio_t0": formate_ratio,
            }
        )

    rows.sort(
        key=lambda row: (
            abs(row["formate_benzoate_ratio_t0"] - args.target_formate_benzoate_ratio),
            -row["cluster_add_probability"],
            row["zr_conc"],
            row["linker_conc"],
            row["capping_agent_conc"],
        )
    )
    selected = rows[: args.max_candidates]

    if args.include_baseline_control:
        _, baseline_ratio = dissolution_probability(
            0.0,
            RUN_DEFAULTS["equilibrium_constant_coefficient"],
            RUN_DEFAULTS["H2O_DMF_RATIO"],
            RUN_DEFAULTS["Capping_agent_conc"],
            RUN_DEFAULTS["Linker_conc"],
        )
        selected = [
            {
                "label": "baseline_control",
                "exchange_rxn_time_seconds": RUN_DEFAULTS["EXCHANGE_RXN_TIME_SECONDS"],
                "zr_conc": RUN_DEFAULTS["Zr_conc"],
                "linker_conc": RUN_DEFAULTS["Linker_conc"],
                "capping_agent_conc": RUN_DEFAULTS["Capping_agent_conc"],
                "equilibrium_constant_coefficient": RUN_DEFAULTS["equilibrium_constant_coefficient"],
                "cluster_add_probability": zr6_cluster_add_probability(
                    RUN_DEFAULTS["Zr_conc"],
                    RUN_DEFAULTS["Linker_conc"],
                    zr6_percentage=1.0,
                ),
                "formate_benzoate_ratio_t0": baseline_ratio,
            }
        ] + selected

    for index, row in enumerate(selected):
        row.setdefault("label", f"candidate_{index:02d}")
    return selected


def summarize_zr12_coordination(assembly):
    zr12_rows = [row for row in core_builder.cluster_summary(assembly) if row["kind"] == "Zr12_AA"]
    if not zr12_rows:
        return {
            "count": 0,
            "coord_min": None,
            "coord_mean": None,
            "coord_max": None,
        }
    coords = [row["coordination"] for row in zr12_rows]
    return {
        "count": len(zr12_rows),
        "coord_min": int(min(coords)),
        "coord_mean": float(np.mean(coords)),
        "coord_max": int(max(coords)),
    }


def simulate_growth_on_assembly(
    assembly,
    candidate,
    total_steps,
    max_entities_delta,
    dissolution_update_interval_steps,
    entropy_table,
    rng_seed,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    start_counts = core_builder.entity_counts(assembly)
    start_total_entities = start_counts["total_entities"]
    max_entities = start_total_entities + max_entities_delta

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

    for cycle in range(total_steps + 1):
        sim_time_seconds = timing * candidate["exchange_rxn_time_seconds"]
        if len(assembly.entities) > max_entities:
            break

        if step == 0 or (
            dissolution_update_interval_steps is not None
            and dissolution_update_interval_steps > 0
            and step % dissolution_update_interval_steps == 0
        ):
            _, formate_benzoate_ratio = dissolution_probability(
                sim_time_seconds,
                candidate["equilibrium_constant_coefficient"],
                RUN_DEFAULTS["H2O_DMF_RATIO"],
                candidate["capping_agent_conc"],
                candidate["linker_conc"],
            )

        step += 1
        flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(
            formate_benzoate_ratio,
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

    end_counts = core_builder.entity_counts(assembly)
    end_zr12_summary = summarize_zr12_coordination(assembly)
    result = {
        "rng_seed": rng_seed,
        "steps_completed": step,
        "sim_time_seconds": timing * candidate["exchange_rxn_time_seconds"],
        "start_counts": start_counts,
        "end_counts": end_counts,
        "delta_zr6": end_counts["Zr6_AA"] - start_counts["Zr6_AA"],
        "delta_zr12": end_counts["Zr12_AA"] - start_counts["Zr12_AA"],
        "delta_bdc": end_counts["BDC"] - start_counts["BDC"],
        "delta_total_entities": end_counts["total_entities"] - start_counts["total_entities"],
        "zr12_loss_count": start_counts["Zr12_AA"] - end_counts["Zr12_AA"],
        "zr12_remaining_fraction": (
            end_counts["Zr12_AA"] / start_counts["Zr12_AA"] if start_counts["Zr12_AA"] else None
        ),
        "event_counts": event_counts,
        "end_zr12_summary": end_zr12_summary,
    }
    return assembly, result


def simulate_growth_case(
    seed_path,
    candidate,
    entropy_table,
    total_steps,
    max_entities_delta,
    dissolution_update_interval_steps,
    bumping_threshold,
    rng_seed,
):
    assembly = load_seed_assembly(seed_path, zr6_percentage=1.0, bumping_threshold=bumping_threshold)
    assembly, result = simulate_growth_on_assembly(
        assembly=assembly,
        candidate=candidate,
        total_steps=total_steps,
        max_entities_delta=max_entities_delta,
        dissolution_update_interval_steps=dissolution_update_interval_steps,
        entropy_table=entropy_table,
        rng_seed=rng_seed,
    )
    result = {
        "seed_name": seed_path.stem,
        **result,
    }
    return assembly, result


def run_single_probe(
    seed_path,
    candidate,
    entropy_table,
    total_steps,
    max_entities_delta,
    dissolution_update_interval_steps,
    bumping_threshold,
    rng_seed,
):
    _, result = simulate_growth_case(
        seed_path=seed_path,
        candidate=candidate,
        entropy_table=entropy_table,
        total_steps=total_steps,
        max_entities_delta=max_entities_delta,
        dissolution_update_interval_steps=dissolution_update_interval_steps,
        bumping_threshold=bumping_threshold,
        rng_seed=rng_seed,
    )
    return result


def aggregate_probe_runs(seed_path, candidate, run_rows):
    delta_zr6_values = [row["delta_zr6"] for row in run_rows]
    delta_zr12_values = [row["delta_zr12"] for row in run_rows]
    zr12_remaining_fraction_values = [row["zr12_remaining_fraction"] for row in run_rows]
    aggregated = {
        "seed_path": seed_path.as_posix(),
        "seed_name": seed_path.stem,
        "candidate_label": candidate["label"],
        "candidate": candidate,
        "replicates": len(run_rows),
        "avg_delta_zr6": float(np.mean(delta_zr6_values)),
        "avg_delta_zr12": float(np.mean(delta_zr12_values)),
        "avg_delta_bdc": float(np.mean([row["delta_bdc"] for row in run_rows])),
        "avg_delta_total_entities": float(np.mean([row["delta_total_entities"] for row in run_rows])),
        "avg_zr12_remaining_fraction": float(np.mean(zr12_remaining_fraction_values)),
        "runs_with_net_zr6_gain": int(sum(1 for value in delta_zr6_values if value > 0)),
        "runs_with_any_zr12_loss": int(sum(1 for value in delta_zr12_values if value < 0)),
        "runs_with_both_zr6_gain_and_zr12_loss": int(
            sum(1 for row in run_rows if row["delta_zr6"] > 0 and row["delta_zr12"] < 0)
        ),
        "avg_zr6_add_events": float(np.mean([row["event_counts"]["zr6_add"] for row in run_rows])),
        "avg_bdc_add_events": float(np.mean([row["event_counts"]["bdc_add"] for row in run_rows])),
        "avg_remove_events": float(np.mean([row["event_counts"]["remove"] for row in run_rows])),
        "avg_link_events": float(np.mean([row["event_counts"]["link"] for row in run_rows])),
        "avg_steps_completed": float(np.mean([row["steps_completed"] for row in run_rows])),
        "avg_sim_time_seconds": float(np.mean([row["sim_time_seconds"] for row in run_rows])),
        "end_zr12_coord_mean_avg": float(
            np.mean(
                [
                    row["end_zr12_summary"]["coord_mean"]
                    for row in run_rows
                    if row["end_zr12_summary"]["coord_mean"] is not None
                ]
            )
        )
        if any(row["end_zr12_summary"]["coord_mean"] is not None for row in run_rows)
        else None,
        "per_run": run_rows,
    }
    return aggregated


def main():
    args = parse_args()
    seed_paths = resolve_seed_paths(args.seed_pkls)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).resolve().parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_metadata = {seed_path.stem: seed_counts(seed_path) for seed_path in seed_paths}
    max_seed_entities = max(meta["total_entities"] for meta in seed_metadata.values())
    entropy_table = build_entropy_table(
        max_seed_entities + args.max_entities_delta + 50,
        args.entropy_correction_coefficient,
    )
    candidate_rows = analytic_candidate_rows(args)

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "seed_pkls": [path.as_posix() for path in seed_paths],
            "replicates": args.replicates,
            "base_rng_seed": args.base_rng_seed,
            "total_steps": args.total_steps,
            "max_entities_delta": args.max_entities_delta,
            "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
            "entropy_correction_coefficient": args.entropy_correction_coefficient,
            "h2o_dmf_ratio": args.h2o_dmf_ratio,
            "bumping_threshold": args.bumping_threshold,
        },
        "seed_metadata": seed_metadata,
        "candidate_rows": candidate_rows,
        "aggregated_results": [],
    }

    for candidate_index, candidate in enumerate(candidate_rows):
        for seed_index, seed_path in enumerate(seed_paths):
            run_rows = []
            for replicate_index in range(args.replicates):
                rng_seed = (
                    args.base_rng_seed
                    + candidate_index * 1000
                    + seed_index * 100
                    + replicate_index
                )
                run_rows.append(
                    run_single_probe(
                        seed_path=seed_path,
                        candidate=candidate,
                        entropy_table=entropy_table,
                        total_steps=args.total_steps,
                        max_entities_delta=args.max_entities_delta,
                        dissolution_update_interval_steps=args.dissolution_update_interval_steps,
                        bumping_threshold=args.bumping_threshold,
                        rng_seed=rng_seed,
                    )
                )
            results["aggregated_results"].append(
                aggregate_probe_runs(
                    seed_path=seed_path,
                    candidate=candidate,
                    run_rows=run_rows,
                )
            )

    sortable_rows = sorted(
        results["aggregated_results"],
        key=lambda row: (
            -row["avg_delta_zr6"],
            row["avg_delta_zr12"],
            row["candidate"]["formate_benzoate_ratio_t0"],
        ),
    )
    results["top_by_zr6_growth"] = sortable_rows[:10]
    results["top_with_any_zr12_loss"] = [
        row for row in sortable_rows if row["runs_with_any_zr12_loss"] > 0
    ][:10]

    basename = args.basename or f"probe_zr6_only_growth_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    json_path = output_dir / f"{basename}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote probe summary to {json_path}")
    for row in results["top_by_zr6_growth"][:6]:
        print(
            json.dumps(
                {
                    "seed": row["seed_name"],
                    "candidate": row["candidate_label"],
                    "avg_delta_zr6": row["avg_delta_zr6"],
                    "avg_delta_zr12": row["avg_delta_zr12"],
                    "cluster_add_probability": row["candidate"]["cluster_add_probability"],
                    "formate_ratio_t0": row["candidate"]["formate_benzoate_ratio_t0"],
                    "zr_conc": row["candidate"]["zr_conc"],
                    "linker_conc": row["candidate"]["linker_conc"],
                    "capping_agent_conc": row["candidate"]["capping_agent_conc"],
                    "equilibrium_constant_coefficient": row["candidate"]["equilibrium_constant_coefficient"],
                    "exchange_rxn_time_seconds": row["candidate"]["exchange_rxn_time_seconds"],
                    "runs_with_both_zr6_gain_and_zr12_loss": row["runs_with_both_zr6_gain_and_zr12_loss"],
                }
            )
        )


if __name__ == "__main__":
    main()
