import argparse
import contextlib
import io
import json
from datetime import datetime
from pathlib import Path

import build_internal_zr12_seed as core_builder
import probe_zr6_only_growth as probe
from UiO66_Assembly_Large_Correction_conc import safe_pickle_save


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_seed_dir = script_dir / "output" / "mixed_nuclei"
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed Zr6-only growth condition on one or more mixed seeds and "
            "save the final assemblies."
        )
    )
    parser.add_argument(
        "--seed-pkls",
        nargs="+",
        default=[
            str(default_seed_dir / "ratio_controlled_mixed_seed_154_seed612_frac055_fill4.pkl"),
            str(default_seed_dir / "ratio_controlled_mixed_seed_160_seed612_frac055_fill6.pkl"),
        ],
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--base-rng-seed", type=int, default=15000)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--max-entities-delta", type=int, default=80)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--zr-conc", type=float, default=5000.0)
    parser.add_argument("--linker-conc", type=float, default=69.1596872253079)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=6.0)
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
    parser.add_argument("--output-dir", default="output/mixed_nuclei/zr6_only_growth_runs")
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def candidate_from_args(args):
    _, formate_ratio = probe.dissolution_probability(
        0.0,
        args.equilibrium_constant_coefficient,
        args.h2o_dmf_ratio,
        args.capping_agent_conc,
        args.linker_conc,
    )
    return {
        "label": "fixed_run",
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
        "formate_benzoate_ratio_t0": formate_ratio,
    }


def resolve_output_dir(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
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


def main():
    args = parse_args()
    seed_paths = probe.resolve_seed_paths(args.seed_pkls)
    output_dir = resolve_output_dir(args.output_dir)
    candidate = candidate_from_args(args)

    seed_metadata = {seed_path.stem: probe.seed_counts(seed_path) for seed_path in seed_paths}
    max_seed_entities = max(meta["total_entities"] for meta in seed_metadata.values())
    entropy_table = probe.build_entropy_table(
        max_seed_entities + args.max_entities_delta + 50,
        args.entropy_correction_coefficient,
    )

    run_group_name = args.basename or f"zr6_only_growth_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_group_dir = output_dir / run_group_name
    run_group_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": run_group_dir.as_posix(),
        "config": {
            "seed_pkls": [path.as_posix() for path in seed_paths],
            "replicates": args.replicates,
            "base_rng_seed": args.base_rng_seed,
            "total_steps": args.total_steps,
            "max_entities_delta": args.max_entities_delta,
            "entropy_correction_coefficient": args.entropy_correction_coefficient,
            "h2o_dmf_ratio": args.h2o_dmf_ratio,
            "dissolution_update_interval_steps": args.dissolution_update_interval_steps,
            "bumping_threshold": args.bumping_threshold,
        },
        "candidate": candidate,
        "seed_metadata": seed_metadata,
        "runs": [],
    }

    for seed_index, seed_path in enumerate(seed_paths):
        for replicate_index in range(args.replicates):
            rng_seed = args.base_rng_seed + seed_index * 100 + replicate_index
            assembly, run_result = probe.simulate_growth_case(
                seed_path=seed_path,
                candidate=candidate,
                entropy_table=entropy_table,
                total_steps=args.total_steps,
                max_entities_delta=args.max_entities_delta,
                dissolution_update_interval_steps=args.dissolution_update_interval_steps,
                bumping_threshold=args.bumping_threshold,
                rng_seed=rng_seed,
            )

            run_stem = (
                f"{seed_path.stem}__zr6only_opt__rep{replicate_index + 1:02d}"
                f"__seed{rng_seed}"
            )
            pkl_path = run_group_dir / f"{run_stem}.pkl"
            mol2_path = run_group_dir / f"{run_stem}.mol2"
            json_path = run_group_dir / f"{run_stem}.json"

            quiet_pickle_save(assembly, pkl_path)
            assembly.get_mol2_file(mol2_path.as_posix())

            cluster_summary = core_builder.cluster_summary(assembly)
            zr12_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "entity_ref"
                }
                for row in cluster_summary
                if row["kind"] == "Zr12_AA"
            ]
            run_payload = {
                **run_result,
                "seed_path": seed_path.as_posix(),
                "candidate": candidate,
                "pkl_path": pkl_path.as_posix(),
                "mol2_path": mol2_path.as_posix(),
                "shape_metrics": core_builder.cluster_shape_metrics(assembly),
                "zr12_rows": zr12_rows,
            }
            json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
            summary_data["runs"].append({**run_payload, "json_path": json_path.as_posix()})
            print(
                json.dumps(
                    {
                        "seed": seed_path.stem,
                        "replicate": replicate_index + 1,
                        "rng_seed": rng_seed,
                        "delta_zr6": run_result["delta_zr6"],
                        "delta_zr12": run_result["delta_zr12"],
                        "delta_bdc": run_result["delta_bdc"],
                        "pkl_path": pkl_path.as_posix(),
                    }
                )
            )

    summary_path = run_group_dir / f"{run_group_name}.summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
