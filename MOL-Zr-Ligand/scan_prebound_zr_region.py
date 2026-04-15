import argparse
import csv
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_ZR_VALUES = [16, 24, 32, 40, 48, 56, 64, 80, 96, 128]
DEFAULT_TARGETS = [6, 8, 10]


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_csv_ints(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a replicated Zr scan for the updated prebound Zr-BTB model and "
            "summarize where growth slows down as Zr concentration increases."
        )
    )
    parser.add_argument(
        "--zr-values",
        default=",".join(str(value) for value in DEFAULT_ZR_VALUES),
        help="Comma-separated Zr concentrations to scan.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(str(value) for value in DEFAULT_TARGETS),
        help="Comma-separated entity targets for time-to-target analysis.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--max-entities", type=int, default=30)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--output-root", default="output/mol_prebound_zr_region_scan")
    parser.add_argument("--output-inter", type=int, default=0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--dissolution-update-interval-steps", type=int, default=1000000)
    parser.add_argument("--zr6-percentage", type=float, default=1.0)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.789387907185137)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=1.32975172557788)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=0.0)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--linker-conc", type=float, default=4.0)
    parser.add_argument("--bumping-threshold", type=float, default=1.8)
    parser.add_argument(
        "--distorted-chemistry-model",
        default="cluster_one_to_one",
        help=(
            "Prebound chemistry model. "
            "Use 'cluster_one_to_one' or 'multisite_first_binding_only'."
        ),
    )
    parser.add_argument(
        "--distorted-ligand-association-constant",
        type=float,
        default=None,
        help="Optional direct override for the effective prebound association constant.",
    )
    parser.add_argument(
        "--distorted-site-equilibrium-constant",
        type=float,
        default=None,
        help="Optional direct override for the multisite site-level exchange constant.",
    )
    parser.add_argument(
        "--distorted-second-step-equivalents",
        type=float,
        default=0.0,
        help="Legacy extra sink term. Keep at 0 for the current single-step model.",
    )
    parser.add_argument("--distorted-num-sites-on-cluster", type=int, default=12)
    parser.add_argument("--distorted-num-sites-on-linker", type=int, default=3)
    parser.add_argument(
        "--script",
        default="run_mol_zr_ligand_case.py",
        help="Path to the single-case runner.",
    )
    return parser.parse_args()


def time_to_target(trace_times, trace_entities, target_entities):
    for time_value, entity_count in zip(trace_times, trace_entities):
        if entity_count >= target_entities:
            return float(time_value)
    return None


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


def load_metrics(run_dir, targets):
    launcher = json.loads((run_dir / "launcher_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    chemistry_path = run_dir / "chemistry_summary.json"
    chemistry_summary = None
    if chemistry_path.exists():
        chemistry_summary = json.loads(chemistry_path.read_text(encoding="utf-8"))

    with open(run_dir / "entities_number.pkl", "rb") as handle:
        entities_number = pickle.load(handle)

    exchange_time = launcher["config"]["EXCHANGE_RXN_TIME_SECONDS"]
    trace_times = [row[0] * exchange_time for row in entities_number]
    trace_entities = [int(row[1]) for row in entities_number]

    metrics = {
        "run_dir": run_dir.as_posix(),
        "simulated_time_seconds": run_summary["simulated_time_seconds"],
        "final_entities": run_summary["final_entities"],
        "max_entities_seen": max(trace_entities) if trace_entities else 0,
        "prebound_zr_bdc_fraction": run_summary.get("prebound_zr_bdc_fraction", 0.0),
        "off_pathway_linker_fraction": run_summary.get("off_pathway_linker_fraction", 0.0),
        "effective_zr6_conc": run_summary.get("effective_zr6_conc", 0.0),
        "effective_linker_conc": run_summary.get("effective_linker_conc", 0.0),
        "external_addition_activity": run_summary.get("external_addition_activity", 1.0),
        "prebound_growth_attempts": run_summary.get("prebound_growth_attempts", 0),
        "prebound_growth_successes": run_summary.get("prebound_growth_successes", 0),
        "prebound_growth_failures": run_summary.get("prebound_growth_failures", 0),
        "prebound_entities_added": run_summary.get("prebound_entities_added", 0),
        "prebound_linkages_formed": run_summary.get("prebound_linkages_formed", 0),
        "prebound_free_growth_site_delta": run_summary.get("prebound_free_growth_site_delta", 0),
        "prebound_ready_pair_delta": run_summary.get("prebound_ready_pair_delta", 0),
        "event_num_grow": run_summary.get("event_num_grow", 0),
        "event_num_grow_success": run_summary.get("event_num_grow_success", 0),
        "event_num_grow_fail": run_summary.get("event_num_grow_fail", 0),
        "event_num_link": run_summary.get("event_num_link", 0),
        "event_num_remove": run_summary.get("event_num_remove", 0),
        "chemistry_summary": chemistry_summary,
    }
    for target_entities in targets:
        metrics[f"time_to_{target_entities}"] = time_to_target(trace_times, trace_entities, target_entities)
        metrics[f"reached_{target_entities}"] = metrics[f"time_to_{target_entities}"] is not None
    return metrics


def aggregate_mode_rows(rows, targets):
    aggregated = {
        "runs": len(rows),
        "mean_final_entities": statistics.mean(row["final_entities"] for row in rows),
        "mean_max_entities_seen": statistics.mean(row["max_entities_seen"] for row in rows),
        "std_max_entities_seen": stdev_or_none([row["max_entities_seen"] for row in rows]),
        "mean_simulated_time_seconds": statistics.mean(row["simulated_time_seconds"] for row in rows),
        "mean_prebound_fraction": statistics.mean(row["prebound_zr_bdc_fraction"] for row in rows),
        "mean_off_pathway_fraction": statistics.mean(row["off_pathway_linker_fraction"] for row in rows),
        "mean_effective_zr6_conc": statistics.mean(row["effective_zr6_conc"] for row in rows),
        "mean_effective_linker_conc": statistics.mean(row["effective_linker_conc"] for row in rows),
        "mean_external_addition_activity": statistics.mean(row["external_addition_activity"] for row in rows),
        "mean_prebound_growth_attempts": statistics.mean(row["prebound_growth_attempts"] for row in rows),
        "mean_prebound_growth_successes": statistics.mean(row["prebound_growth_successes"] for row in rows),
        "mean_prebound_growth_failures": statistics.mean(row["prebound_growth_failures"] for row in rows),
        "mean_prebound_entities_added": statistics.mean(row["prebound_entities_added"] for row in rows),
        "mean_prebound_linkages_formed": statistics.mean(row["prebound_linkages_formed"] for row in rows),
        "mean_prebound_free_growth_site_delta": statistics.mean(row["prebound_free_growth_site_delta"] for row in rows),
        "mean_prebound_ready_pair_delta": statistics.mean(row["prebound_ready_pair_delta"] for row in rows),
    }
    for target_entities in targets:
        target_times = [row[f"time_to_{target_entities}"] for row in rows]
        aggregated[f"reach_fraction_{target_entities}"] = (
            sum(1 for value in target_times if value is not None) / len(rows)
        )
        aggregated[f"mean_time_to_{target_entities}"] = mean_or_none(target_times)
        aggregated[f"std_time_to_{target_entities}"] = stdev_or_none(target_times)
    return aggregated


def build_jobs(args, output_root):
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (Path(__file__).resolve().parent / script_path).resolve()

    jobs = []
    for zr_value in parse_csv_floats(args.zr_values):
        zr_token = str(zr_value).replace(".", "p")
        for repeat_index in range(args.repeats):
            for mode in ("control", "distorted"):
                basename = f"{mode}_zr{zr_token}_r{repeat_index}"
                run_dir = output_root / basename
                command = [
                    sys.executable,
                    str(script_path),
                    "--zr6-percentage", str(args.zr6_percentage),
                    "--zr-conc", str(zr_value),
                    "--entropy-correction-coefficient", str(args.entropy_correction_coefficient),
                    "--equilibrium-constant-coefficient", str(args.equilibrium_constant_coefficient),
                    "--h2o-dmf-ratio", str(args.h2o_dmf_ratio),
                    "--capping-agent-conc", str(args.capping_agent_conc),
                    "--linker-conc", str(args.linker_conc),
                    "--total-steps", str(args.total_steps),
                    "--bumping-threshold", str(args.bumping_threshold),
                    "--max-entities", str(args.max_entities),
                    "--output-inter", str(args.output_inter),
                    "--exchange-rxn-time-seconds", str(args.exchange_rxn_time_seconds),
                    "--dissolution-update-interval-steps", str(args.dissolution_update_interval_steps),
                    "--output-root", str(output_root),
                    "--basename", basename,
                    "--index", str(repeat_index),
                    "--distorted-chemistry-model", str(args.distorted_chemistry_model),
                    "--distorted-second-step-equivalents", str(args.distorted_second_step_equivalents),
                    "--distorted-num-sites-on-cluster", str(args.distorted_num_sites_on_cluster),
                    "--distorted-num-sites-on-linker", str(args.distorted_num_sites_on_linker),
                ]
                if mode == "distorted":
                    command.append("--enable-distorted-linker")
                if args.distorted_ligand_association_constant is not None:
                    command.extend(
                        [
                            "--distorted-ligand-association-constant",
                            str(args.distorted_ligand_association_constant),
                        ]
                    )
                if args.distorted_site_equilibrium_constant is not None:
                    command.extend(
                        [
                            "--distorted-site-equilibrium-constant",
                            str(args.distorted_site_equilibrium_constant),
                        ]
                    )
                jobs.append(
                    {
                        "mode": mode,
                        "zr_conc": float(zr_value),
                        "repeat_index": repeat_index,
                        "basename": basename,
                        "run_dir": run_dir,
                        "command": command,
                        "cwd": script_path.parent,
                    }
                )
    return jobs


def run_job(job):
    completed = subprocess.run(
        job["command"],
        cwd=job["cwd"],
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = job["run_dir"] / "scan_launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "COMMAND:",
                " ".join(job["command"]),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {job['basename']} with code {completed.returncode}. See {log_path}"
        )
    return job


def write_per_run_csv(rows, targets, output_path):
    fieldnames = [
        "zr_conc",
        "mode",
        "repeat_index",
        "simulated_time_seconds",
        "final_entities",
        "max_entities_seen",
        "prebound_zr_bdc_fraction",
        "off_pathway_linker_fraction",
        "effective_zr6_conc",
        "effective_linker_conc",
        "external_addition_activity",
        "prebound_growth_attempts",
        "prebound_growth_successes",
        "prebound_growth_failures",
        "prebound_entities_added",
        "prebound_linkages_formed",
        "prebound_free_growth_site_delta",
        "prebound_ready_pair_delta",
        "event_num_grow",
        "event_num_grow_success",
        "event_num_grow_fail",
        "event_num_link",
        "event_num_remove",
        "run_dir",
    ]
    for target_entities in targets:
        fieldnames.append(f"time_to_{target_entities}")
        fieldnames.append(f"reached_{target_entities}")

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_csv(summary_rows, targets, output_path):
    fieldnames = [
        "zr_conc",
        "mode",
        "runs",
        "mean_final_entities",
        "mean_max_entities_seen",
        "std_max_entities_seen",
        "mean_simulated_time_seconds",
        "mean_prebound_fraction",
        "mean_off_pathway_fraction",
        "mean_effective_zr6_conc",
        "mean_effective_linker_conc",
        "mean_external_addition_activity",
        "mean_prebound_growth_attempts",
        "mean_prebound_growth_successes",
        "mean_prebound_growth_failures",
        "mean_prebound_entities_added",
        "mean_prebound_linkages_formed",
        "mean_prebound_free_growth_site_delta",
        "mean_prebound_ready_pair_delta",
    ]
    for target_entities in targets:
        fieldnames.append(f"reach_fraction_{target_entities}")
        fieldnames.append(f"mean_time_to_{target_entities}")
        fieldnames.append(f"std_time_to_{target_entities}")

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def safe_errorbar(ax, xs, ys, errors, label, color):
    if all(error is None or math.isnan(error) for error in errors):
        ax.plot(xs, ys, marker="o", label=label, color=color)
        return
    normalized_errors = [0.0 if error is None or math.isnan(error) else error for error in errors]
    ax.errorbar(xs, ys, yerr=normalized_errors, marker="o", label=label, color=color, capsize=3)


def build_plot(summary_lookup, zr_values, targets, output_path):
    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    colors = {"control": "#1f77b4", "distorted": "#d95f02"}

    max_entities_ax = axes[0]
    time_ax = axes[1]
    fraction_ax = axes[2]

    for mode in ("control", "distorted"):
        max_entities = [summary_lookup[(zr_value, mode)]["mean_max_entities_seen"] for zr_value in zr_values]
        max_entities_std = [summary_lookup[(zr_value, mode)]["std_max_entities_seen"] for zr_value in zr_values]
        safe_errorbar(
            max_entities_ax,
            zr_values,
            max_entities,
            max_entities_std,
            label=mode,
            color=colors[mode],
        )

        time_target = targets[min(1, len(targets) - 1)]
        mean_times = []
        std_times = []
        for zr_value in zr_values:
            row = summary_lookup[(zr_value, mode)]
            mean_times.append(row.get(f"mean_time_to_{time_target}"))
            std_times.append(row.get(f"std_time_to_{time_target}"))
        valid_x = [x for x, y in zip(zr_values, mean_times) if y is not None]
        valid_y = [y for y in mean_times if y is not None]
        valid_err = [e for y, e in zip(mean_times, std_times) if y is not None]
        safe_errorbar(
            time_ax,
            valid_x,
            valid_y,
            valid_err,
            label=f"{mode} time to {time_target}",
            color=colors[mode],
        )

    prebound_fractions = [
        summary_lookup[(zr_value, "distorted")]["mean_prebound_fraction"] for zr_value in zr_values
    ]
    reach_fraction_target = targets[min(1, len(targets) - 1)]
    reach_fractions = [
        summary_lookup[(zr_value, "distorted")].get(f"reach_fraction_{reach_fraction_target}", 0.0)
        for zr_value in zr_values
    ]
    fraction_ax.plot(zr_values, prebound_fractions, marker="o", color="#4c9f70", label="mean prebound fraction")
    fraction_ax.plot(zr_values, reach_fractions, marker="s", color="#7f3c8d", label=f"reach fraction to {reach_fraction_target}")

    max_entities_ax.set_ylabel("Mean max entities")
    max_entities_ax.set_title("Updated prebound Zr-BTB model: growth suppression vs Zr")
    max_entities_ax.legend()
    max_entities_ax.grid(alpha=0.25)

    time_ax.set_ylabel(f"Mean time to {time_target} entities (s)")
    time_ax.legend()
    time_ax.grid(alpha=0.25)

    fraction_ax.set_xlabel("Total Zr concentration (mM)")
    fraction_ax.set_ylabel("Fraction")
    fraction_ax.legend()
    fraction_ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def identify_decreasing_region(summary_lookup, zr_values, target_entities):
    region_rows = []
    previous_row = None
    for zr_value in zr_values:
        row = summary_lookup[(zr_value, "distorted")]
        if previous_row is not None:
            current_time = row.get(f"mean_time_to_{target_entities}")
            previous_time = previous_row.get(f"mean_time_to_{target_entities}")
            if current_time is None and previous_time is not None:
                trend = "no_longer_reaches_target"
            elif current_time is not None and previous_time is not None and current_time > previous_time:
                trend = "slower_time_to_target"
            elif row["mean_max_entities_seen"] < previous_row["mean_max_entities_seen"]:
                trend = "lower_max_entities"
            else:
                trend = None
            if trend is not None:
                region_rows.append(
                    {
                        "zr_conc": zr_value,
                        "previous_zr_conc": previous_row["zr_conc"],
                        "trend": trend,
                        "mean_max_entities_seen": row["mean_max_entities_seen"],
                        f"mean_time_to_{target_entities}": current_time,
                        "mean_prebound_fraction": row["mean_prebound_fraction"],
                    }
                )
        previous_row = row
    return region_rows


def main():
    args = parse_args()
    zr_values = parse_csv_floats(args.zr_values)
    targets = parse_csv_ints(args.targets)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = output_root / f"scan_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, output_root)
    workers = max(1, min(args.workers, len(jobs)))

    print(
        json.dumps(
            {
                "output_root": output_root.as_posix(),
                "workers": workers,
                "jobs": len(jobs),
                "zr_values": zr_values,
                "targets": targets,
                "repeats": args.repeats,
            },
            indent=2,
        )
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            completed_job = future.result()
            print(
                f"finished {completed_job['basename']}"
            )

    per_run_rows = []
    grouped_rows = {}
    for job in jobs:
        metrics = load_metrics(job["run_dir"], targets)
        row = {
            "zr_conc": job["zr_conc"],
            "mode": job["mode"],
            "repeat_index": job["repeat_index"],
            **metrics,
        }
        per_run_rows.append(row)
        grouped_rows.setdefault((job["zr_conc"], job["mode"]), []).append(row)

    summary_rows = []
    summary_lookup = {}
    for zr_value in zr_values:
        for mode in ("control", "distorted"):
            aggregated = aggregate_mode_rows(grouped_rows[(zr_value, mode)], targets)
            summary_row = {
                "zr_conc": zr_value,
                "mode": mode,
                **aggregated,
            }
            summary_rows.append(summary_row)
            summary_lookup[(zr_value, mode)] = summary_row

    per_run_path = output_root / "per_run_metrics.csv"
    summary_path = output_root / "summary_metrics.csv"
    write_per_run_csv(per_run_rows, targets, per_run_path)
    write_summary_csv(summary_rows, targets, summary_path)

    plot_path = output_root / "region_scan.svg"
    build_plot(summary_lookup, zr_values, targets, plot_path)

    target_for_region = targets[min(1, len(targets) - 1)]
    decreasing_region = identify_decreasing_region(summary_lookup, zr_values, target_for_region)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": output_root.as_posix(),
        "targets": targets,
        "zr_values": zr_values,
        "repeats": args.repeats,
        "summary_csv": summary_path.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "plot_path": plot_path.as_posix() if plot_path.exists() else None,
        "decreasing_region": decreasing_region,
        "summary_rows": summary_rows,
    }
    json_path = output_root / "region_scan_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
