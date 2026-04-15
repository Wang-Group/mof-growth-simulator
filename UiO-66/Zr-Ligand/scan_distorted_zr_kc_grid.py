import argparse
import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from scan_distorted_time_to_target import run_case


DEFAULT_ZR_VALUES = [24, 28, 32, 36, 40, 44, 48]
DEFAULT_KC_VALUES = [0.5, 1.0, 1.32975172557788, 2.0, 4.0, 8.0]


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Systematically scan distorted/prebound time-to-20 across a Zr x KC grid."
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
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--target-entities", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10_000_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-root", default="output/distorted_zr_kc_grid")
    parser.add_argument("--zr6-percentage", type=float, default=1.0)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.789387907185137)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=0.0)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--linker-conc", type=float, default=4.0)
    parser.add_argument("--bumping-threshold", type=float, default=2.0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--dissolution-update-interval-steps", type=int, default=1000000)
    parser.add_argument(
        "--distorted-ligand-association-constant",
        type=float,
        default=None,
        help="Optional direct override for the effective prebound association constant.",
    )
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


def heatmap_color(value, min_value, max_value):
    if value is None:
        return "#f2f2f2"
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    red = int(255 * ratio + 245 * (1.0 - ratio))
    green = int(120 * (1.0 - ratio) + 230 * ratio)
    blue = int(100 * (1.0 - ratio) + 80 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_heatmap(summary_lookup, zr_values, kc_values, output_path):
    width = 1080
    height = 760
    text_color = "#222222"
    title_y = 34

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="70" y="{title_y}" font-size="24" font-weight="bold" fill="{text_color}">Distorted/Prebound Zr x KC Grid</text>',
        f'<text x="70" y="56" font-size="12" fill="{text_color}">Each cell is 4 repeats, target = 20 entities, max steps = 1e10.</text>',
    ]

    panel_specs = [
        ("median_time_to_20_seconds", "Median Time To 20 (s)", 70, 100),
        ("mean_prebound_fraction", "Mean Prebound Fraction", 560, 100),
        ("std_time_to_20_seconds", "Std(Time To 20)", 70, 430),
        ("mean_prebound_growth_failures", "Mean Prebound Growth Failures", 560, 430),
    ]

    for metric_key, panel_title, panel_left, panel_top in panel_specs:
        cell_w = 70
        cell_h = 38
        x_label_y = panel_top - 20
        y_label_x = panel_left - 16
        lines.append(f'<text x="{panel_left}" y="{panel_top - 36}" font-size="16" font-weight="bold" fill="{text_color}">{svg_escape(panel_title)}</text>')
        values = [summary_lookup[(zr_value, kc_value)].get(metric_key) for zr_value in zr_values for kc_value in kc_values]
        numeric_values = [value for value in values if value is not None]
        min_value = min(numeric_values) if numeric_values else 0.0
        max_value = max(numeric_values) if numeric_values else 1.0

        lines.append(f'<text x="{panel_left + len(kc_values) * cell_w / 2:.2f}" y="{x_label_y}" text-anchor="middle" font-size="12" fill="{text_color}">KC</text>')
        lines.append(f'<text x="{panel_left - 42}" y="{panel_top + len(zr_values) * cell_h / 2:.2f}" transform="rotate(-90 {panel_left - 42} {panel_top + len(zr_values) * cell_h / 2:.2f})" text-anchor="middle" font-size="12" fill="{text_color}">Zr (mM)</text>')

        for col_index, kc_value in enumerate(kc_values):
            x = panel_left + col_index * cell_w
            lines.append(f'<text x="{x + cell_w / 2:.2f}" y="{panel_top - 6}" text-anchor="middle" font-size="11" fill="{text_color}">{kc_value:g}</text>')
        for row_index, zr_value in enumerate(zr_values):
            y = panel_top + row_index * cell_h
            lines.append(f'<text x="{y_label_x}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="end" font-size="11" fill="{text_color}">{zr_value:g}</text>')
            for col_index, kc_value in enumerate(kc_values):
                x = panel_left + col_index * cell_w
                row = summary_lookup[(zr_value, kc_value)]
                value = row.get(metric_key)
                fill = heatmap_color(value, min_value, max_value)
                lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white" stroke-width="1.5"/>')
                label = "NA" if value is None else (
                    f"{value:.0f}" if abs(value) >= 100 else f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
                )
                lines.append(f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="middle" font-size="10" fill="{text_color}">{label}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_run_csv(rows, output_path):
    fieldnames = [
        "zr_conc",
        "kc",
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
        "runs",
        "reach_fraction_20",
        "mean_time_to_20_seconds",
        "std_time_to_20_seconds",
        "median_time_to_20_seconds",
        "mean_final_entities",
        "mean_steps_executed",
        "mean_prebound_fraction",
        "mean_prebound_growth_attempts",
        "mean_prebound_growth_successes",
        "mean_prebound_growth_failures",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    zr_values = parse_csv_floats(args.zr_values)
    kc_values = parse_csv_floats(args.kc_values)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    output_root = output_root / f"scan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for zr_index, zr_value in enumerate(zr_values):
        for kc_index, kc_value in enumerate(kc_values):
            for repeat_index in range(args.repeats):
                jobs.append(
                    {
                        "zr_conc": float(zr_value),
                        "kc": float(kc_value),
                        "repeat_index": int(repeat_index),
                        "seed": 51000 + zr_index * 1000 + kc_index * 100 + repeat_index,
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
                        "distorted_ligand_association_constant": args.distorted_ligand_association_constant,
                        "distorted_second_step_equivalents": float(args.distorted_second_step_equivalents),
                    }
                )

    (output_root / "scan_config.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "zr_values": zr_values,
                "kc_values": kc_values,
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
            rows.append(result)
            print(
                f"finished zr={result['zr_conc']:g} kc={result['kc']:g} rep={result['repeat_index']} "
                f"time={result['time_to_target_seconds']}"
            )

    rows.sort(key=lambda item: (item["zr_conc"], item["kc"], item["repeat_index"]))
    per_run_path = output_root / "zr_kc_time20_per_run.csv"
    write_per_run_csv(rows, per_run_path)

    summary_rows = []
    summary_lookup = {}
    for zr_value in zr_values:
        for kc_value in kc_values:
            subset = [row for row in rows if row["zr_conc"] == float(zr_value) and row["kc"] == float(kc_value)]
            target_times = [row["time_to_target_seconds"] for row in subset if row["time_to_target_seconds"] is not None]
            summary_row = {
                "zr_conc": float(zr_value),
                "kc": float(kc_value),
                "runs": len(subset),
                "reach_fraction_20": sum(1 for row in subset if row["reached_target"]) / len(subset),
                "mean_time_to_20_seconds": mean_or_none(target_times),
                "std_time_to_20_seconds": stdev_or_none(target_times),
                "median_time_to_20_seconds": statistics.median(target_times) if target_times else None,
                "mean_final_entities": statistics.mean(row["final_entities"] for row in subset),
                "mean_steps_executed": statistics.mean(row["steps_executed"] for row in subset),
                "mean_prebound_fraction": statistics.mean(row["prebound_fraction"] for row in subset),
                "mean_prebound_growth_attempts": statistics.mean(row["prebound_growth_attempts"] for row in subset),
                "mean_prebound_growth_successes": statistics.mean(row["prebound_growth_successes"] for row in subset),
                "mean_prebound_growth_failures": statistics.mean(row["prebound_growth_failures"] for row in subset),
            }
            summary_rows.append(summary_row)
            summary_lookup[(float(zr_value), float(kc_value))] = summary_row

    summary_rows.sort(key=lambda item: (item["zr_conc"], item["kc"]))
    summary_path = output_root / "zr_kc_time20_summary.csv"
    write_summary_csv(summary_rows, summary_path)

    ranked_rows = sorted(
        summary_rows,
        key=lambda item: item["median_time_to_20_seconds"] if item["median_time_to_20_seconds"] is not None else -1,
        reverse=True,
    )
    ranked_path = output_root / "zr_kc_ranked_by_median_time20.csv"
    write_summary_csv(ranked_rows, ranked_path)

    heatmap_path = output_root / "zr_kc_time20_heatmap.svg"
    render_heatmap(summary_lookup, zr_values, kc_values, heatmap_path)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": output_root.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "summary_csv": summary_path.as_posix(),
        "ranked_csv": ranked_path.as_posix(),
        "heatmap_svg": heatmap_path.as_posix(),
        "top_slowest_conditions": ranked_rows[:10],
    }
    (output_root / "zr_kc_time20_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
