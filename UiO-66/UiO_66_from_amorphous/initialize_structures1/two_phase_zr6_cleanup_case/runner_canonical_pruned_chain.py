import contextlib
import io
import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
INIT_DIR = CASE_DIR.parent
if INIT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, INIT_DIR.as_posix())

import build_internal_zr12_seed as core_builder
import fragment_cleanup
import probe_zr6_only_growth as probe
import run_outer_zr12_zr6_only_case as outer_stage


DEFAULT_OUTPUT_DIR = (
    INIT_DIR
    / "output"
    / "mixed_nuclei"
    / "two_phase_amorphous_equilibrate_zr6only_pruned"
)
DEFAULT_PHASE1_SEED = (
    INIT_DIR
    / "output"
    / "mixed_nuclei"
    / "two_phase_amorphous_equilibrate_zr6only"
    / "two_phase_eqbond_zr6only_default"
    / "two_phase_eqbond_zr6only_default__phase1_seed.pkl"
)
CANONICAL_CANDIDATE = {
    "label": "equal_make_break_zr6_only",
    "exchange_rxn_time_seconds": 0.1,
    "zr_conc": 1200.0,
    "linker_conc": 69.1596872253079,
    "capping_agent_conc": 180.0,
    "equilibrium_constant_coefficient": 1.3,
    "cluster_add_probability": 0.32161903475593895,
    "formate_benzoate_ratio_t0": 1.0478238106145703,
}
COMMON_CONFIG = {
    "dissolution_update_interval_steps": None,
    "entropy_correction_coefficient": 0.0,
    "bumping_threshold": probe.RUN_DEFAULTS["BUMPING_THRESHOLD"],
}
STAGE_PLAN = [
    {
        "id": "default_stage01",
        "run_name": "two_phase_eqbond_zr6only_default",
        "stage_label": "stage01",
        "source": "phase1_seed",
        "replicates": 4,
        "base_rng_seed": 89000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "default_stage02",
        "run_name": "two_phase_eqbond_zr6only_default",
        "stage_label": "stage02",
        "source": "default_stage01",
        "replicates": 4,
        "base_rng_seed": 90000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "default_stage03",
        "run_name": "two_phase_eqbond_zr6only_default",
        "stage_label": "stage03",
        "source": "default_stage02",
        "replicates": 4,
        "base_rng_seed": 91000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "default_stage04",
        "run_name": "two_phase_eqbond_zr6only_default",
        "stage_label": "stage04",
        "source": "default_stage03",
        "replicates": 4,
        "base_rng_seed": 92000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "continue10_stage01",
        "run_name": "two_phase_eqbond_zr6only_continue_from10",
        "stage_label": "stage01",
        "source": "default_stage04",
        "replicates": 6,
        "base_rng_seed": 93000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "continue10_stage02",
        "run_name": "two_phase_eqbond_zr6only_continue_from10",
        "stage_label": "stage02",
        "source": "continue10_stage01",
        "replicates": 6,
        "base_rng_seed": 94000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "from9_recheck_stage01",
        "run_name": "two_phase_eqbond_zr6only_from9_recheck",
        "stage_label": "stage01",
        "source": "continue10_stage01",
        "replicates": 12,
        "base_rng_seed": 95000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "continue8_stage01",
        "run_name": "two_phase_eqbond_zr6only_continue_from8",
        "stage_label": "stage01",
        "source": "from9_recheck_stage01",
        "replicates": 8,
        "base_rng_seed": 96000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "continue8_stage02",
        "run_name": "two_phase_eqbond_zr6only_continue_from8",
        "stage_label": "stage02",
        "source": "continue8_stage01",
        "replicates": 8,
        "base_rng_seed": 97000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "continue8_stage03",
        "run_name": "two_phase_eqbond_zr6only_continue_from8",
        "stage_label": "stage03",
        "source": "continue8_stage02",
        "replicates": 8,
        "base_rng_seed": 98000,
        "total_steps": 30000,
        "max_entities_delta": 300,
    },
    {
        "id": "long6_stage01",
        "run_name": "two_phase_eqbond_zr6only_long_from6",
        "stage_label": "stage01",
        "source": "continue8_stage02",
        "replicates": 8,
        "base_rng_seed": 99000,
        "total_steps": 150000,
        "max_entities_delta": 1500,
    },
    {
        "id": "long6_stage02",
        "run_name": "two_phase_eqbond_zr6only_long_from6",
        "stage_label": "stage02",
        "source": "long6_stage01",
        "replicates": 8,
        "base_rng_seed": 100000,
        "total_steps": 150000,
        "max_entities_delta": 1500,
    },
    {
        "id": "long6_stage03",
        "run_name": "two_phase_eqbond_zr6only_long_from6",
        "stage_label": "stage03",
        "source": "long6_stage02",
        "replicates": 8,
        "base_rng_seed": 101000,
        "total_steps": 150000,
        "max_entities_delta": 1500,
    },
    {
        "id": "long2_stage01",
        "run_name": "two_phase_eqbond_zr6only_long_from2",
        "stage_label": "stage01",
        "source": "long6_stage03",
        "replicates": 8,
        "base_rng_seed": 102000,
        "total_steps": 150000,
        "max_entities_delta": 1500,
    },
    {
        "id": "long2_stage02",
        "run_name": "two_phase_eqbond_zr6only_long_from2",
        "stage_label": "stage02",
        "source": "long2_stage01",
        "replicates": 8,
        "base_rng_seed": 103000,
        "total_steps": 150000,
        "max_entities_delta": 1500,
    },
]


def parse_args():
    parser = ArgumentParser(
        description=(
            "Replay the retained canonical Zr12-cleanup chain with detached-fragment pruning "
            "enabled after every unlink event."
        )
    )
    parser.add_argument("--phase1-seed-pkl", default=DEFAULT_PHASE1_SEED.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--basename", default="canonical_pruned_chain")
    parser.add_argument("--stop-after-stage", default=None)
    return parser.parse_args()


def component_summary(assembly):
    components = fragment_cleanup.connected_components(assembly)
    if not components:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
        }
    largest = len(components[0])
    total = len(assembly.entities)
    return {
        "component_count": len(components),
        "largest_component_size": largest,
        "largest_component_fraction": largest / max(total, 1),
    }


def run_stage(stage_spec, seed_path, run_dir):
    seed_counts = probe.seed_counts(seed_path)
    entropy_table = probe.build_entropy_table(
        seed_counts["total_entities"] + stage_spec["max_entities_delta"] + 50,
        COMMON_CONFIG["entropy_correction_coefficient"],
    )

    run_payloads = []
    for replicate_index in range(stage_spec["replicates"]):
        rng_seed = stage_spec["base_rng_seed"] + replicate_index
        assembly, run_result = probe.simulate_growth_case(
            seed_path=seed_path,
            candidate=CANONICAL_CANDIDATE,
            entropy_table=entropy_table,
            total_steps=stage_spec["total_steps"],
            max_entities_delta=stage_spec["max_entities_delta"],
            dissolution_update_interval_steps=COMMON_CONFIG["dissolution_update_interval_steps"],
            bumping_threshold=COMMON_CONFIG["bumping_threshold"],
            rng_seed=rng_seed,
        )
        end_components = component_summary(assembly)

        stem = (
            f"{stage_spec['run_name']}__{stage_spec['stage_label']}__rep{replicate_index + 1:02d}"
            f"__seed{rng_seed}"
        )
        pkl_path, mol2_path = outer_stage.save_assembly_outputs(assembly, run_dir, stem)
        json_path = run_dir / f"{stem}.json"
        payload = {
            **run_result,
            "stage_id": stage_spec["id"],
            "stage_label": stage_spec["stage_label"],
            "stage_seed_path": seed_path.as_posix(),
            "candidate": CANONICAL_CANDIDATE,
            "pkl_path": pkl_path.as_posix(),
            "mol2_path": mol2_path.as_posix(),
            "json_path": json_path.as_posix(),
            "shape_metrics": core_builder.cluster_shape_metrics(assembly),
            "end_components": end_components,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_payloads.append(payload)

        print(
            json.dumps(
                {
                    "stage_id": stage_spec["id"],
                    "replicate": replicate_index + 1,
                    "rng_seed": rng_seed,
                    "end_zr12": payload["end_counts"]["Zr12_AA"],
                    "end_total_entities": payload["end_counts"]["total_entities"],
                    "component_count": end_components["component_count"],
                    "largest_component_fraction": end_components["largest_component_fraction"],
                    "fragment_pruned_entities": payload["event_counts"]["fragment_pruned_entities"],
                }
            )
        )

    best_run = outer_stage.summarize_best_run(run_payloads)
    if best_run is None:
        raise RuntimeError(f"No runs were produced for stage {stage_spec['id']}.")

    summary = {
        "stage_id": stage_spec["id"],
        "run_name": stage_spec["run_name"],
        "stage_label": stage_spec["stage_label"],
        "stage_seed_path": seed_path.as_posix(),
        "stage_seed_counts": seed_counts,
        "replicates": stage_spec["replicates"],
        "base_rng_seed": stage_spec["base_rng_seed"],
        "total_steps": stage_spec["total_steps"],
        "max_entities_delta": stage_spec["max_entities_delta"],
        "runs": run_payloads,
        "best_run_by_zr12_loss": best_run,
        "aggregate": {
            "final_total_entities_mean": sum(row["end_counts"]["total_entities"] for row in run_payloads)
            / len(run_payloads),
            "final_zr12_mean": sum(row["end_counts"]["Zr12_AA"] for row in run_payloads) / len(run_payloads),
            "component_count_mean": sum(row["end_components"]["component_count"] for row in run_payloads)
            / len(run_payloads),
            "largest_component_fraction_mean": sum(
                row["end_components"]["largest_component_fraction"] for row in run_payloads
            )
            / len(run_payloads),
            "fragment_pruned_entities_mean": sum(
                row["event_counts"]["fragment_pruned_entities"] for row in run_payloads
            )
            / len(run_payloads),
        },
    }
    return summary


def main():
    args = parse_args()
    output_root = outer_stage.resolve_output_dir(args.output_dir)
    run_dir = output_root / args.basename
    run_dir.mkdir(parents=True, exist_ok=True)

    phase1_seed = Path(args.phase1_seed_pkl)
    if not phase1_seed.is_absolute():
        phase1_seed = (INIT_DIR / phase1_seed).resolve()

    best_run_by_stage = {}
    stage_summaries = []

    for stage_spec in STAGE_PLAN:
        source = stage_spec["source"]
        if source == "phase1_seed":
            seed_path = phase1_seed
        else:
            seed_path = Path(best_run_by_stage[source]["pkl_path"])

        stage_summary = run_stage(stage_spec, seed_path, run_dir)
        stage_summaries.append(stage_summary)
        best_run_by_stage[stage_spec["id"]] = stage_summary["best_run_by_zr12_loss"]

        if args.stop_after_stage and stage_spec["id"] == args.stop_after_stage:
            break

    all_run_payloads = [row for stage in stage_summaries for row in stage["runs"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "phase1_seed_pkl": phase1_seed.as_posix(),
        "candidate": CANONICAL_CANDIDATE,
        "common_config": COMMON_CONFIG,
        "stage_plan": STAGE_PLAN,
        "stages": stage_summaries,
        "best_run_by_zr12_loss": outer_stage.summarize_best_run(all_run_payloads),
    }
    summary_path = run_dir / f"{args.basename}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": summary_path.as_posix(),
                "best_run": summary["best_run_by_zr12_loss"],
                "completed_stage_count": len(stage_summaries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with contextlib.redirect_stderr(sys.stderr), contextlib.redirect_stdout(sys.stdout):
        main()
