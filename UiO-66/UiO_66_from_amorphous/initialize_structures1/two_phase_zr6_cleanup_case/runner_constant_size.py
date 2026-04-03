import argparse
import contextlib
import io
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
INIT_DIR = CASE_DIR.parent
if INIT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, INIT_DIR.as_posix())

import build_internal_zr12_seed as core_builder
import probe_zr6_only_growth as probe
import run_outer_zr12_zr6_only_case as outer_stage

OUTPUT_ROOT = INIT_DIR / "output" / "mixed_nuclei" / "two_phase_constant_size_closed_anneal"
DEFAULT_SEED_PKL = (
    INIT_DIR
    / "output"
    / "mixed_nuclei"
    / "two_phase_amorphous_equilibrate_zr6only"
    / "two_phase_eqbond_zr6only_default"
    / "two_phase_eqbond_zr6only_default__phase1_seed.pkl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a constant-size closed anneal from the mixed phase-1 seed. "
            "External addition is disabled; only existing internal link/unlink events are allowed."
        )
    )
    parser.add_argument("--seed-pkl", default=DEFAULT_SEED_PKL.as_posix())
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--base-rng-seed", type=int, default=104000)
    parser.add_argument("--total-steps", type=int, default=150000)
    parser.add_argument("--snapshot-interval", type=int, default=10000)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--linker-conc", type=float, default=69.1596872253079)
    parser.add_argument("--capping-agent-conc", type=float, default=180.0)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=1.3)
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=0.0,
        help="0.0 reproduces the near-unity internal-link weight used in the growth chain.",
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=probe.RUN_DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=None,
        help="Defaults to fixed t=0 unlink weight for the whole run.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_ROOT.as_posix(),
    )
    parser.add_argument("--basename", default="constant_size_closed_phase1seed")
    return parser.parse_args()


def connected_component_metrics(assembly):
    entities = list(assembly.entities)
    if not entities:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "isolated_entity_count": 0,
            "component_sizes_desc": [],
        }

    visited = set()
    component_sizes = []
    for entity in entities:
        if entity in visited:
            continue
        stack = [entity]
        visited.add(entity)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = getattr(current, "connected_entities", None) or []
            for neighbor in set(neighbors):
                if neighbor is None or neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        component_sizes.append(size)

    component_sizes.sort(reverse=True)
    return {
        "component_count": len(component_sizes),
        "largest_component_size": component_sizes[0],
        "isolated_entity_count": sum(1 for size in component_sizes if size == 1),
        "component_sizes_desc": component_sizes,
    }


def summarize_state(assembly):
    counts = core_builder.entity_counts(assembly)
    return {
        **counts,
        "zr12_coordination": probe.summarize_zr12_coordination(assembly),
        "components": connected_component_metrics(assembly),
    }


def unlink_pair_preserve_size(assembly, selected_pair):
    pair = selected_pair
    if pair not in assembly.linked_carboxylate_pairs:
        reverse_pair = (selected_pair[1], selected_pair[0])
        if reverse_pair not in assembly.linked_carboxylate_pairs:
            return False
        pair = reverse_pair

    assembly.linked_carboxylate_pairs.remove(pair)
    assembly.ready_to_connect_carboxylate_pairs.add(pair)
    for carboxylate in pair:
        assembly.pair_index[carboxylate] = pair
        other_entity = pair[0].belonging_entity if carboxylate is pair[1] else pair[1].belonging_entity
        if carboxylate.belonging_entity is None or other_entity is None:
            return False
        try:
            carboxylate.belonging_entity.connected_entities.remove(other_entity)
        except ValueError:
            return False
    return True


def simulate_constant_size_closed_anneal(
    assembly,
    *,
    total_steps,
    exchange_rxn_time_seconds,
    equilibrium_constant_coefficient,
    capping_agent_conc,
    linker_conc,
    h2o_dmf_ratio,
    dissolution_update_interval_steps,
    entropy_table,
    rng_seed,
    snapshot_interval,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    start_state = summarize_state(assembly)
    timing = 0.0
    step = 0
    event_counts = {
        "link": 0,
        "unlink": 0,
        "idle": 0,
    }
    snapshots = []

    formate_benzoate_ratio = None
    termination_reason = "requested_steps_completed"
    for cycle in range(total_steps + 1):
        sim_time_seconds = timing * exchange_rxn_time_seconds
        if step == 0 or (
            dissolution_update_interval_steps is not None
            and dissolution_update_interval_steps > 0
            and step % dissolution_update_interval_steps == 0
        ):
            _, formate_benzoate_ratio = probe.dissolution_probability(
                sim_time_seconds,
                equilibrium_constant_coefficient,
                h2o_dmf_ratio,
                capping_agent_conc,
                linker_conc,
            )

        if snapshot_interval and step % snapshot_interval == 0:
            snapshots.append(
                {
                    "step": step,
                    "sim_time_seconds": sim_time_seconds,
                    **summarize_state(assembly),
                }
            )

        num_entities = len(assembly.entities)
        ready_pairs = len(assembly.ready_to_connect_carboxylate_pairs)
        linked_pairs = len(assembly.linked_carboxylate_pairs)
        link_weight = ready_pairs * entropy_table[num_entities]
        unlink_weight = linked_pairs * formate_benzoate_ratio
        total_rate = link_weight + unlink_weight
        if total_rate <= 0:
            termination_reason = "no_internal_events_available"
            break

        step += 1
        random_select = random.random() * total_rate
        if random_select < link_weight and ready_pairs > 0:
            selected_pair = assembly.ready_to_connect_carboxylate_pairs.get_random()
            assembly.link_internal_carboxylate(selected_pair)
            event_counts["link"] += 1
        elif linked_pairs > 0:
            selected_pair = assembly.linked_carboxylate_pairs.get_random()
            if unlink_pair_preserve_size(assembly, selected_pair):
                event_counts["unlink"] += 1
            else:
                event_counts["idle"] += 1
        else:
            event_counts["idle"] += 1

        random_draw = max(random.random(), 1e-12)
        timing -= math.log(random_draw) / max(total_rate, 1e-12)

    end_state = summarize_state(assembly)
    if not snapshots or snapshots[-1]["step"] != step:
        snapshots.append(
            {
                "step": step,
                "sim_time_seconds": timing * exchange_rxn_time_seconds,
                **end_state,
            }
        )

    result = {
        "rng_seed": rng_seed,
        "steps_completed": step,
        "sim_time_seconds": timing * exchange_rxn_time_seconds,
        "termination_reason": termination_reason,
        "start_state": start_state,
        "end_state": end_state,
        "delta_linked_pairs": end_state["linked_pairs"] - start_state["linked_pairs"],
        "delta_ready_pairs": end_state["ready_pairs"] - start_state["ready_pairs"],
        "delta_component_count": (
            end_state["components"]["component_count"] - start_state["components"]["component_count"]
        ),
        "delta_largest_component_size": (
            end_state["components"]["largest_component_size"]
            - start_state["components"]["largest_component_size"]
        ),
        "delta_zr12_coord_mean": (
            end_state["zr12_coordination"]["coord_mean"] - start_state["zr12_coordination"]["coord_mean"]
        ),
        "event_counts": event_counts,
        "snapshots": snapshots,
    }
    return assembly, result


def aggregate_runs(run_payloads):
    metrics = {
        "final_linked_pairs_mean": float(np.mean([row["end_state"]["linked_pairs"] for row in run_payloads])),
        "final_ready_pairs_mean": float(np.mean([row["end_state"]["ready_pairs"] for row in run_payloads])),
        "final_component_count_mean": float(
            np.mean([row["end_state"]["components"]["component_count"] for row in run_payloads])
        ),
        "final_largest_component_size_mean": float(
            np.mean([row["end_state"]["components"]["largest_component_size"] for row in run_payloads])
        ),
        "final_zr12_coord_mean_mean": float(
            np.mean([row["end_state"]["zr12_coordination"]["coord_mean"] for row in run_payloads])
        ),
        "final_zr12_coord_mean_min": float(
            np.min([row["end_state"]["zr12_coordination"]["coord_mean"] for row in run_payloads])
        ),
        "final_zr12_coord_mean_max": float(
            np.max([row["end_state"]["zr12_coordination"]["coord_mean"] for row in run_payloads])
        ),
    }
    representative = sorted(
        run_payloads,
        key=lambda row: (
            abs(row["end_state"]["zr12_coordination"]["coord_mean"] - metrics["final_zr12_coord_mean_mean"]),
            row["end_state"]["components"]["component_count"],
            -row["end_state"]["components"]["largest_component_size"],
        ),
    )[0]
    return metrics, {
        "rng_seed": representative["rng_seed"],
        "json_path": representative["json_path"],
        "pkl_path": representative["pkl_path"],
    }


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_name = args.basename or f"constant_size_closed_anneal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(args.seed_pkl)
    if not seed_path.is_absolute():
        seed_path = (INIT_DIR / seed_path).resolve()

    entropy_table = probe.build_entropy_table(
        core_builder.entity_counts(probe.load_seed_assembly(seed_path, 1.0, probe.RUN_DEFAULTS["BUMPING_THRESHOLD"]))[
            "total_entities"
        ]
        + 10,
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
        assembly, result = simulate_constant_size_closed_anneal(
            assembly,
            total_steps=args.total_steps,
            exchange_rxn_time_seconds=args.exchange_rxn_time_seconds,
            equilibrium_constant_coefficient=args.equilibrium_constant_coefficient,
            capping_agent_conc=args.capping_agent_conc,
            linker_conc=args.linker_conc,
            h2o_dmf_ratio=args.h2o_dmf_ratio,
            dissolution_update_interval_steps=args.dissolution_update_interval_steps,
            entropy_table=entropy_table,
            rng_seed=rng_seed,
            snapshot_interval=args.snapshot_interval,
        )

        run_stem = f"{run_name}__rep{replicate_index + 1:02d}__seed{rng_seed}"
        pkl_path = run_dir / f"{run_stem}.pkl"
        json_path = run_dir / f"{run_stem}.json"
        outer_stage.quiet_pickle_save(assembly, pkl_path)
        payload = {
            **result,
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
                    "final_linked_pairs": payload["end_state"]["linked_pairs"],
                    "final_ready_pairs": payload["end_state"]["ready_pairs"],
                    "final_component_count": payload["end_state"]["components"]["component_count"],
                    "final_largest_component_size": payload["end_state"]["components"]["largest_component_size"],
                    "final_zr12_coord_mean": payload["end_state"]["zr12_coordination"]["coord_mean"],
                    "json_path": payload["json_path"],
                }
            )
        )

    aggregate, representative_run = aggregate_runs(run_payloads)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "seed_path": seed_path.as_posix(),
        "protocol": {
            "mode": "constant_size_closed_anneal",
            "external_addition_enabled": False,
            "entity_removal_enabled": False,
            "internal_linking_enabled": True,
            "internal_unlinking_enabled": True,
            "exchange_rxn_time_seconds": args.exchange_rxn_time_seconds,
            "equilibrium_constant_coefficient": args.equilibrium_constant_coefficient,
            "capping_agent_conc": args.capping_agent_conc,
            "linker_conc": args.linker_conc,
            "entropy_correction_coefficient": args.entropy_correction_coefficient,
            "formate_benzoate_ratio_t0": probe.dissolution_probability(
                0.0,
                args.equilibrium_constant_coefficient,
                args.h2o_dmf_ratio,
                args.capping_agent_conc,
                args.linker_conc,
            )[1],
        },
        "config": {
            "replicates": args.replicates,
            "base_rng_seed": args.base_rng_seed,
            "total_steps": args.total_steps,
            "snapshot_interval": args.snapshot_interval,
            "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
        },
        "runs": run_payloads,
        "aggregate": aggregate,
        "representative_run": representative_run,
    }
    summary_path = run_dir / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": summary_path.as_posix(),
                "aggregate": aggregate,
                "representative_run": representative_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
