import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a pure-SVG distribution-focused time-to-target plot from the per-run scan CSV."
        )
    )
    parser.add_argument("--per-run-csv", required=True, help="Path to the per-run CSV.")
    parser.add_argument("--target-entities", type=int, required=True, help="Entity target used in the scan.")
    parser.add_argument("--output-svg", required=True, help="Output SVG path.")
    parser.add_argument(
        "--output-stats-csv",
        default=None,
        help="Optional output CSV for the per-Zr summary statistics.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("reached_target", "")).strip().lower() != "true":
                continue
            time_value = row.get("time_to_target_seconds")
            if not time_value:
                continue
            rows.append(
                {
                    "zr_conc": float(row["zr_conc"]),
                    "repeat_index": int(row["repeat_index"]),
                    "time_to_target_seconds": float(time_value),
                }
            )
    return rows


def build_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["zr_conc"]].append(row["time_to_target_seconds"])

    stats_rows = []
    for zr_value in sorted(grouped):
        values = np.array(sorted(grouped[zr_value]), dtype=float)
        q1 = float(np.percentile(values, 25))
        median = float(np.percentile(values, 50))
        q3 = float(np.percentile(values, 75))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats_rows.append(
            {
                "zr_conc": zr_value,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": median,
                "std": std,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values.tolist(),
            }
        )
    return stats_rows


def write_stats_csv(stats_rows, output_path):
    fieldnames = ["zr_conc", "n", "mean", "median", "std", "q1", "q3", "iqr", "min", "max"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow({key: row[key] for key in fieldnames})


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_tick(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def log_ticks(y_min, y_max):
    exponent_min = int(math.floor(math.log10(y_min)))
    exponent_max = int(math.ceil(math.log10(y_max)))
    ticks = []
    for exponent in range(exponent_min, exponent_max + 1):
        for multiplier in (1, 3):
            tick_value = multiplier * (10 ** exponent)
            if y_min <= tick_value <= y_max:
                ticks.append(float(tick_value))
    if not ticks:
        ticks = [y_min, y_max]
    return ticks


def map_y(value, y_min, y_max, top, height):
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)
    log_value = math.log10(max(value, y_min))
    fraction = (log_value - log_min) / max(log_max - log_min, 1e-12)
    return top + height - fraction * height


def map_x(index, count, left, width):
    if count <= 1:
        return left + width / 2.0
    return left + index / (count - 1) * width


def polyline_points(xs, ys):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def polygon_band(xs, lower, upper):
    points = [(x, y) for x, y in zip(xs, upper)]
    points.extend((x, y) for x, y in reversed(list(zip(xs, lower))))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_axes(lines, left, top, width, height, zr_values, y_min, y_max, panel_title, y_label):
    text_color = "#1f1f1f"
    grid_color = "#dddddd"
    axis_color = "#333333"

    lines.append(f'<text x="{left}" y="{top - 18}" font-size="18" font-weight="bold" fill="{text_color}">{svg_escape(panel_title)}</text>')

    for tick_value in log_ticks(y_min, y_max):
        y_coord = map_y(tick_value, y_min, y_max, top, height)
        lines.append(f'<line x1="{left}" y1="{y_coord:.2f}" x2="{left + width}" y2="{y_coord:.2f}" stroke="{grid_color}" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y_coord + 4:.2f}" text-anchor="end" font-size="12" fill="{text_color}">{format_tick(tick_value)}</text>')

    for index, zr_value in enumerate(zr_values):
        x_coord = map_x(index, len(zr_values), left, width)
        lines.append(f'<line x1="{x_coord:.2f}" y1="{top}" x2="{x_coord:.2f}" y2="{top + height}" stroke="{grid_color}" stroke-width="1"/>')
        lines.append(f'<text x="{x_coord:.2f}" y="{top + height + 24:.2f}" text-anchor="middle" font-size="12" fill="{text_color}">{zr_value:g}</text>')

    lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="{axis_color}" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="{axis_color}" stroke-width="1.5"/>')
    lines.append(f'<text x="{left + width / 2:.2f}" y="{top + height + 52:.2f}" text-anchor="middle" font-size="13" fill="{text_color}">Total Zr concentration (mM)</text>')
    y_center = top + height / 2.0
    lines.append(
        f'<text x="{left - 62}" y="{y_center:.2f}" transform="rotate(-90 {left - 62} {y_center:.2f})" text-anchor="middle" font-size="13" fill="{text_color}">{svg_escape(y_label)}</text>'
    )


def render_svg(stats_rows, output_svg, target_entities, title):
    width = 1180
    height = 920
    left = 110
    right = 60
    top = 90
    panel_width = width - left - right
    panel_height = 300
    panel_gap = 140
    bottom_top = top + panel_height + panel_gap

    zr_values = [row["zr_conc"] for row in stats_rows]
    y_min = min(row["min"] for row in stats_rows)
    y_max = max(row["max"] for row in stats_rows)
    y_min = max(y_min * 0.8, 1e-3)
    y_max = y_max * 1.25

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; }',
        '</style>',
        f'<text x="{left}" y="36" font-size="26" font-weight="bold" fill="#1f1f1f">{svg_escape(title or f"Time to {target_entities} Entities with Multisite Prebound Model")}</text>',
        f'<text x="{left}" y="60" font-size="13" fill="#555555">Raw replicate distributions above, median/IQR and mean/std summary below. Log-scaled y-axis because the kinetics are heavy-tailed.</text>',
    ]

    render_axes(
        lines,
        left,
        top,
        panel_width,
        panel_height,
        zr_values,
        y_min,
        y_max,
        f"Raw Time-to-{target_entities} Distributions",
        f"Time to {target_entities} entities (s, log scale)",
    )
    render_axes(
        lines,
        left,
        bottom_top,
        panel_width,
        panel_height,
        zr_values,
        y_min,
        y_max,
        f"Summary Trends",
        f"Summary time to {target_entities} (s, log scale)",
    )

    box_width = 54.0
    raw_color = "#d95f02"
    box_fill = "#f4c27a"
    box_edge = "#7f5539"
    mean_color = "#0077b6"
    median_color = "#222222"
    iqr_fill = "#f4c27a"
    std_fill = "#8ecae6"
    extrema_color = "#8d99ae"

    rng = np.random.default_rng(20260415)
    top_xs = []
    mean_ys = []
    median_ys = []
    q1_ys = []
    q3_ys = []
    lower_std_ys = []
    upper_std_ys = []
    min_ys = []
    max_ys = []

    for index, row in enumerate(stats_rows):
        x_coord = map_x(index, len(stats_rows), left, panel_width)
        top_xs.append(x_coord)
        mean_ys.append(map_y(row["mean"], y_min, y_max, bottom_top, panel_height))
        median_ys.append(map_y(row["median"], y_min, y_max, bottom_top, panel_height))
        q1_ys.append(map_y(row["q1"], y_min, y_max, bottom_top, panel_height))
        q3_ys.append(map_y(row["q3"], y_min, y_max, bottom_top, panel_height))
        lower_std_value = max(row["mean"] - row["std"], y_min)
        upper_std_value = row["mean"] + row["std"]
        lower_std_ys.append(map_y(lower_std_value, y_min, y_max, bottom_top, panel_height))
        upper_std_ys.append(map_y(upper_std_value, y_min, y_max, bottom_top, panel_height))
        min_ys.append(map_y(row["min"], y_min, y_max, bottom_top, panel_height))
        max_ys.append(map_y(row["max"], y_min, y_max, bottom_top, panel_height))

        min_y = map_y(row["min"], y_min, y_max, top, panel_height)
        max_y = map_y(row["max"], y_min, y_max, top, panel_height)
        q1_y = map_y(row["q1"], y_min, y_max, top, panel_height)
        median_y = map_y(row["median"], y_min, y_max, top, panel_height)
        q3_y = map_y(row["q3"], y_min, y_max, top, panel_height)
        mean_y = map_y(row["mean"], y_min, y_max, top, panel_height)

        lines.append(f'<line x1="{x_coord:.2f}" y1="{min_y:.2f}" x2="{x_coord:.2f}" y2="{max_y:.2f}" stroke="{box_edge}" stroke-width="1.6"/>')
        lines.append(f'<line x1="{x_coord - 13:.2f}" y1="{min_y:.2f}" x2="{x_coord + 13:.2f}" y2="{min_y:.2f}" stroke="{box_edge}" stroke-width="1.6"/>')
        lines.append(f'<line x1="{x_coord - 13:.2f}" y1="{max_y:.2f}" x2="{x_coord + 13:.2f}" y2="{max_y:.2f}" stroke="{box_edge}" stroke-width="1.6"/>')
        lines.append(
            f'<rect x="{x_coord - box_width / 2:.2f}" y="{q3_y:.2f}" width="{box_width:.2f}" height="{max(q1_y - q3_y, 1.2):.2f}" fill="{box_fill}" fill-opacity="0.75" stroke="{box_edge}" stroke-width="1.5"/>'
        )
        lines.append(f'<line x1="{x_coord - box_width / 2:.2f}" y1="{median_y:.2f}" x2="{x_coord + box_width / 2:.2f}" y2="{median_y:.2f}" stroke="{median_color}" stroke-width="2.3"/>')
        lines.append(f'<circle cx="{x_coord:.2f}" cy="{mean_y:.2f}" r="4.8" fill="{mean_color}"/>')

        for offset, value in zip(rng.uniform(-box_width * 0.32, box_width * 0.32, size=len(row["values"])), row["values"]):
            point_y = map_y(value, y_min, y_max, top, panel_height)
            lines.append(f'<circle cx="{x_coord + offset:.2f}" cy="{point_y:.2f}" r="3.3" fill="{raw_color}" fill-opacity="0.52"/>')

        n_y = max(max_y - 8.0, top + 14.0)
        lines.append(f'<text x="{x_coord:.2f}" y="{n_y:.2f}" text-anchor="middle" font-size="10.5" fill="#666666">n={row["n"]}</text>')

    lines.append(f'<polygon points="{polygon_band(top_xs, q1_ys, q3_ys)}" fill="{iqr_fill}" fill-opacity="0.35" stroke="none"/>')
    lines.append(f'<polygon points="{polygon_band(top_xs, lower_std_ys, upper_std_ys)}" fill="{std_fill}" fill-opacity="0.30" stroke="none"/>')
    lines.append(f'<polyline points="{polyline_points(top_xs, min_ys)}" fill="none" stroke="{extrema_color}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    lines.append(f'<polyline points="{polyline_points(top_xs, max_ys)}" fill="none" stroke="{extrema_color}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    lines.append(f'<polyline points="{polyline_points(top_xs, mean_ys)}" fill="none" stroke="{mean_color}" stroke-width="2.4"/>')
    lines.append(f'<polyline points="{polyline_points(top_xs, median_ys)}" fill="none" stroke="{median_color}" stroke-width="2.2"/>')

    for x_coord, mean_y, median_y in zip(top_xs, mean_ys, median_ys):
        lines.append(f'<circle cx="{x_coord:.2f}" cy="{mean_y:.2f}" r="4.8" fill="{mean_color}"/>')
        lines.append(f'<rect x="{x_coord - 4.5:.2f}" y="{median_y - 4.5:.2f}" width="9" height="9" fill="{median_color}" transform="rotate(45 {x_coord:.2f} {median_y:.2f})"/>')

    legend_x = left + panel_width - 230
    legend_y = top + 14
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="{box_fill}" fill-opacity="0.75" stroke="{box_edge}" stroke-width="1.2"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{legend_y + 11}" font-size="12" fill="#333333">IQR / box</text>')
    lines.append(f'<circle cx="{legend_x + 7}" cy="{legend_y + 33}" r="4.5" fill="{mean_color}"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{legend_y + 37}" font-size="12" fill="#333333">Mean</text>')
    lines.append(f'<rect x="{legend_x + 2.5}" y="{legend_y + 49.5}" width="9" height="9" fill="{median_color}" transform="rotate(45 {legend_x + 7:.2f} {legend_y + 54:.2f})"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{legend_y + 58}" font-size="12" fill="#333333">Median</text>')
    lines.append(f'<circle cx="{legend_x + 7}" cy="{legend_y + 75}" r="3.3" fill="{raw_color}" fill-opacity="0.52"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{legend_y + 79}" font-size="12" fill="#333333">Raw repeats</text>')
    lines.append(f'<rect x="{legend_x}" y="{bottom_top + 10}" width="14" height="14" fill="{std_fill}" fill-opacity="0.30" stroke="none"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{bottom_top + 22}" font-size="12" fill="#333333">Mean ± std</text>')
    lines.append(f'<rect x="{legend_x}" y="{bottom_top + 34}" width="14" height="14" fill="{iqr_fill}" fill-opacity="0.35" stroke="none"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{bottom_top + 46}" font-size="12" fill="#333333">Median IQR band</text>')
    lines.append(f'<line x1="{legend_x}" y1="{bottom_top + 62}" x2="{legend_x + 14}" y2="{bottom_top + 62}" stroke="{extrema_color}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    lines.append(f'<text x="{legend_x + 22}" y="{bottom_top + 66}" font-size="12" fill="#333333">Min / max</text>')

    lines.append("</svg>")
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_svg = Path(args.output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.per_run_csv)
    stats_rows = build_stats(rows)
    render_svg(
        stats_rows=stats_rows,
        output_svg=output_svg,
        target_entities=args.target_entities,
        title=args.title,
    )
    if args.output_stats_csv:
        write_stats_csv(stats_rows, args.output_stats_csv)


if __name__ == "__main__":
    main()
