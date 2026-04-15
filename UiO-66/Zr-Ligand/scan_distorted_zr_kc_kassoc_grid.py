import argparse
import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from scan_distorted_time_to_target import run_case


DEFAULT_ZR_VALUES = [24, 32, 40, 48]
DEFAULT_KC_VALUES = [1.0, 1.32975172557788, 2.0]
DEFAULT_KASSOC_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05]


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan distorted/prebound time-to-target across Zr, KC, and an independent "
            "distorted-ligand association constant."
        )
    )
    parser.add_argument(
        "--zr-values",
        default=",".join(str(value) for value in DEFAULT_ZR_VALUES),
        help="Comma-separated Zr concentrations to scan.",
    )
    parser.add_argument(
        "--kc-values",
        default=",".join(str(value) for value in DEFAULT_KC_VALUES),
        help="Comma-separated KC values to scan.",
    )
    parser.add_argument(
        "--kassoc-values",
        default=",".join(str(value) for value in DEFAULT_KASSOC_VALUES),
        help="Comma-separated direct distorted-ligand association constants.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-entities", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10_000_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Rewrite the per-run and summary CSVs after this many finished jobs.",
    )
    parser.add_argument("--output-root", default="output/distorted_zr_kc_kassoc_grid")
    parser.add_argument("--zr6-percentage", type=float, default=1.0)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.789387907185137)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=0.0)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--linker-conc", type=float, default=4.0)
    parser.add_argument("--bumping-threshold", type=float, default=2.0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--dissolution-update-interval-steps", type=int, default=1000000)
    parser.add_argument(
        "--distorted-second-step-equivalents",
        type=float,
        default=0.0,
        help="Legacy sink parameter. Keep at 0 for the current single-step model.",
    )
    return parser.parse_args()


def mean_or_none(values):
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return statistics.mean(valid_values)


def stdev_or_none(values):
    valid_values = [value for value in values if value is not None]
    if len(valid_values) < 2:
        return None
    return statistics.stdev(valid_values)


def write_per_run_csv(rows, output_path):
    fieldnames = [
        "zr_conc",
        "kc",
        "kassoc",
        "repeat_index",
        "seed",
        "reached_target",
        "time_to_target_seconds",
        "final_entities",
        "steps_executed",
        "simulated_time_seconds",
        "prebound_fraction",
        "effective_zr6_conc",
        "effective_linker_conc",
        "external_addition_activity",
        "prebound_growth_attempts",
        "prebound_growth_successes",
        "prebound_growth_failures",
        "wall_seconds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(rows, output_path):
    fieldnames = [
        "zr_conc",
        "kc",
        "kassoc",
        "runs",
        "reach_fraction_target",
        "mean_time_to_target_seconds",
        "std_time_to_target_seconds",
        "median_time_to_target_seconds",
        "mean_final_entities",
        "mean_prebound_fraction",
        "mean_effective_zr6_conc",
        "mean_effective_linker_conc",
        "mean_prebound_growth_failures",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(rows, zr_values, kc_values, kassoc_values):
    summary_rows = []
    for zr_value in zr_values:
        for kc_value in kc_values:
            for kassoc_value in kassoc_values:
                subset = [
                    row for row in rows
                    if row["zr_conc"] == float(zr_value)
                    and row["kc"] == float(kc_value)
                    and row["kassoc"] == float(kassoc_value)
                ]
                if not subset:
                    continue
                target_times = [row["time_to_target_seconds"] for row in subset if row["time_to_target_seconds"] is not None]
                summary_rows.append(
                    {
                        "zr_conc": float(zr_value),
                        "kc": float(kc_value),
                        "kassoc": float(kassoc_value),
                        "runs": len(subset),
                        "reach_fraction_target": sum(1 for row in subset if row["reached_target"]) / len(subset),
                        "mean_time_to_target_seconds": mean_or_none(target_times),
                        "std_time_to_target_seconds": stdev_or_none(target_times),
                        "median_time_to_target_seconds": statistics.median(target_times) if target_times else None,
                        "mean_final_entities": statistics.mean(row["final_entities"] for row in subset),
                        "mean_prebound_fraction": statistics.mean(row["prebound_fraction"] for row in subset),
                        "mean_effective_zr6_conc": statistics.mean(row["effective_zr6_conc"] for row in subset),
                        "mean_effective_linker_conc": statistics.mean(row["effective_linker_conc"] for row in subset),
                        "mean_prebound_growth_failures": statistics.mean(row["prebound_growth_failures"] for row in subset),
                    }
                )

    summary_rows.sort(
        key=lambda item: item["median_time_to_target_seconds"] if item["median_time_to_target_seconds"] is not None else -1,
        reverse=True,
    )
    return summary_rows


def build_jobs(args, zr_values, kc_values, kassoc_values):
    jobs = []
    for zr_index, zr_value in enumerate(zr_values):
        for kc_index, kc_value in enumerate(kc_values):
            for kassoc_index, kassoc_value in enumerate(kassoc_values):
                for repeat_index in range(args.repeats):
                    jobs.append(
                        {
                            "zr_conc": float(zr_value),
                            "kc": float(kc_value),
                            "kassoc": float(kassoc_value),
                            "repeat_index": int(repeat_index),
                            "seed": 61000 + zr_index * 10000 + kc_index * 1000 + kassoc_index * 100 + repeat_index,
                            "target_entities": int(args.target_entities),
                            "max_steps": int(args.max_steps),
                            "zr6_percentage": float(args.zr6_percentage),
                            "entropy_correction_coefficient": float(args.entropy_correction_coefficient),
                            "equilibrium_constant_coefficient": float(kc_value),
                            "h2o_dmf_ratio": float(args.h2o_dmf_ratio),
                            "capping_agent_conc": float(args.capping_agent_conc),
                            "linker_conc": float(args.linker_conc),
                            "bumping_threshold": float(args.bumping_threshold),
                            "exchange_rxn_time_seconds": float(args.exchange_rxn_time_seconds),
                            "dissolution_update_interval_steps": int(args.dissolution_update_interval_steps),
                            "distorted_ligand_association_constant": float(kassoc_value),
                            "distorted_second_step_equivalents": float(args.distorted_second_step_equivalents),
                        }
                    )
    return jobs


def main():
    args = parse_args()
    zr_values = parse_csv_floats(args.zr_values)
    kc_values = parse_csv_floats(args.kc_values)
    kassoc_values = parse_csv_floats(args.kassoc_values)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    output_root = output_root / f"scan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    per_run_path = output_root / "zr_kc_kassoc_time_to_target_per_run.csv"
    summary_path = output_root / "zr_kc_kassoc_time_to_target_summary.csv"
    summary_json_path = output_root / "zr_kc_kassoc_time_to_target_summary.json"

    jobs = build_jobs(args, zr_values, kc_values, kassoc_values)
    (output_root / "scan_config.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "zr_values": zr_values,
                "kc_values": kc_values,
                "kassoc_values": kassoc_values,
                "repeats": args.repeats,
                "target_entities": args.target_entities,
                "max_steps": args.max_steps,
                "workers": args.workers,
                "output_root": output_root.as_posix(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_job = {executor.submit(run_case, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            job = future_to_job[future]
            result["kc"] = job["kc"]
            result["kassoc"] = job["kassoc"]
            rows.append(result)
            print(
                f"finished zr={result['zr_conc']:g} kc={result['kc']:g} "
                f"kassoc={result['kassoc']:g} rep={result['repeat_index']} "
                f"time={result['time_to_target_seconds']}"
            )
            if len(rows) % max(1, int(args.checkpoint_every)) == 0:
                rows.sort(key=lambda item: (item["zr_conc"], item["kc"], item["kassoc"], item["repeat_index"]))
                write_per_run_csv(rows, per_run_path)
                summary_rows = build_summary_rows(rows, zr_values, kc_values, kassoc_values)
                write_summary_csv(summary_rows, summary_path)
                summary_json_path.write_text(
                    json.dumps(
                        {
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "output_root": output_root.as_posix(),
                            "per_run_csv": per_run_path.as_posix(),
                            "summary_csv": summary_path.as_posix(),
                            "completed_jobs": len(rows),
                            "total_jobs": len(jobs),
                            "top_slowest_conditions": summary_rows[:15],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    rows.sort(key=lambda item: (item["zr_conc"], item["kc"], item["kassoc"], item["repeat_index"]))
    write_per_run_csv(rows, per_run_path)
    summary_rows = build_summary_rows(rows, zr_values, kc_values, kassoc_values)
    write_summary_csv(summary_rows, summary_path)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": output_root.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "summary_csv": summary_path.as_posix(),
        "top_slowest_conditions": summary_rows[:15],
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
