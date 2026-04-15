import argparse
import csv
from pathlib import Path


CONTROL_COLOR = "#1f77b4"
DISTORTED_COLOR = "#d95f02"
PREBOUND_COLOR = "#2a9d8f"
GRID_COLOR = "#d9d9d9"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render SVG summary plots from a prebound-Zr region scan."
    )
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--per-run-csv", default=None)
    parser.add_argument("--output-summary-svg", required=True)
    parser.add_argument("--output-replicates-svg", default=None)
    return parser.parse_args()


def load_summary_rows(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    summary = {}
    zr_values = sorted({float(row["zr_conc"]) for row in rows})
    for row in rows:
        summary[(float(row["zr_conc"]), row["mode"])] = row
    return zr_values, summary


def load_per_run_rows(path):
    if path is None:
        return []
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def get_float(row, key):
    if row is None:
        return None
    value = row.get(key)
    if value in ("", None):
        return None
    return float(value)


def map_x(value, x_min, x_max, left, width):
    x_span = max(x_max - x_min, 1e-12)
    return left + (value - x_min) / x_span * width


def map_y(value, y_min, y_max, top, height):
    y_span = max(y_max - y_min, 1e-12)
    return top + height - (value - y_min) / y_span * height


def build_polyline(xs, ys, x_min, x_max, y_min, y_max, left, top, width, height):
    points = []
    for x_value, y_value in zip(xs, ys):
        if y_value is None:
            continue
        x_coord = map_x(x_value, x_min, x_max, left, width)
        y_coord = map_y(y_value, y_min, y_max, top, height)
        points.append(f"{x_coord:.2f},{y_coord:.2f}")
    return " ".join(points)


def add_panel_axes(lines, title, x_label, y_label, x_ticks, y_ticks, left, top, width, height, x_min, x_max, y_min, y_max):
    lines.append(f'<text x="{left}" y="{top - 16}" font-size="15" font-weight="bold" fill="{TEXT_COLOR}">{svg_escape(title)}</text>')
    for y_tick in y_ticks:
        y_coord = map_y(y_tick, y_min, y_max, top, height)
        lines.append(f'<line x1="{left}" y1="{y_coord:.2f}" x2="{left + width}" y2="{y_coord:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y_coord + 4:.2f}" text-anchor="end" font-size="11" fill="{TEXT_COLOR}">{svg_escape(f"{y_tick:g}")}</text>')
    for x_tick in x_ticks:
        x_coord = map_x(x_tick, x_min, x_max, left, width)
        lines.append(f'<line x1="{x_coord:.2f}" y1="{top}" x2="{x_coord:.2f}" y2="{top + height}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        lines.append(f'<text x="{x_coord:.2f}" y="{top + height + 18}" text-anchor="middle" font-size="11" fill="{TEXT_COLOR}">{svg_escape(f"{x_tick:g}")}</text>')
    lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="{AXIS_COLOR}" stroke-width="1.3"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="{AXIS_COLOR}" stroke-width="1.3"/>')
    lines.append(f'<text x="{left + width / 2:.2f}" y="{top + height + 40}" text-anchor="middle" font-size="12" fill="{TEXT_COLOR}">{svg_escape(x_label)}</text>')
    lines.append(
        f'<text x="{left - 50}" y="{top + height / 2:.2f}" transform="rotate(-90 {left - 50} {top + height / 2:.2f})" text-anchor="middle" font-size="12" fill="{TEXT_COLOR}">{svg_escape(y_label)}</text>'
    )


def render_summary_svg(zr_values, summary, output_path):
    width = 980
    panel_width = 380
    panel_height = 240
    left_margin = 90
    top_margin = 70
    col_gap = 70
    row_gap = 90
    total_height = top_margin + panel_height * 2 + row_gap + 80

    x_min = min(zr_values)
    x_max = max(zr_values)
    x_ticks = zr_values

    panels = [
        {
            "title": "Mean Max Entities Seen",
            "y_label": "Mean max entities",
            "series": [
                ("control", CONTROL_COLOR, "mean_max_entities_seen"),
                ("distorted", DISTORTED_COLOR, "mean_max_entities_seen"),
            ],
            "y_min": 0.0,
            "y_max": max(get_float(summary[(zr, mode)], "mean_max_entities_seen") or 0.0 for zr in zr_values for mode in ("control", "distorted")) + 2.0,
        },
        {
            "title": "Reach Fraction To 20 Entities",
            "y_label": "Reach fraction",
            "series": [
                ("control", CONTROL_COLOR, "reach_fraction_20"),
                ("distorted", DISTORTED_COLOR, "reach_fraction_20"),
            ],
            "y_min": 0.0,
            "y_max": 1.0,
        },
        {
            "title": "Mean Time To 10 Entities",
            "y_label": "Time to 10 (s)",
            "series": [
                ("control", CONTROL_COLOR, "mean_time_to_10"),
                ("distorted", DISTORTED_COLOR, "mean_time_to_10"),
            ],
            "y_min": 0.0,
            "y_max": max(get_float(summary[(zr, mode)], "mean_time_to_10") or 0.0 for zr in zr_values for mode in ("control", "distorted")) + 40.0,
        },
        {
            "title": "Prebound Fraction",
            "y_label": "Prebound fraction",
            "series": [
                ("distorted", PREBOUND_COLOR, "mean_prebound_fraction"),
            ],
            "y_min": 0.0,
            "y_max": max(get_float(summary[(zr, "distorted")], "mean_prebound_fraction") or 0.0 for zr in zr_values) + 0.03,
        },
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left_margin}" y="32" font-size="22" font-weight="bold" fill="{TEXT_COLOR}">Updated Prebound Zr-BDC Scan</text>',
        f'<text x="{left_margin}" y="52" font-size="12" fill="{TEXT_COLOR}">Control vs distorted/prebound branch from the refined 5-repeat scan.</text>',
    ]

    legend_x = width - 250
    legend_y = 30
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{CONTROL_COLOR}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-size="12" fill="{TEXT_COLOR}">control</text>')
    lines.append(f'<line x1="{legend_x}" y1="{legend_y + 20}" x2="{legend_x + 28}" y2="{legend_y + 20}" stroke="{DISTORTED_COLOR}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x + 36}" y="{legend_y + 24}" font-size="12" fill="{TEXT_COLOR}">distorted</text>')
    lines.append(f'<line x1="{legend_x + 110}" y1="{legend_y + 20}" x2="{legend_x + 138}" y2="{legend_y + 20}" stroke="{PREBOUND_COLOR}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x + 146}" y="{legend_y + 24}" font-size="12" fill="{TEXT_COLOR}">prebound fraction</text>')

    for index, panel in enumerate(panels):
        col = index % 2
        row = index // 2
        left = left_margin + col * (panel_width + col_gap)
        top = top_margin + row * (panel_height + row_gap)
        y_ticks = [panel["y_min"] + (panel["y_max"] - panel["y_min"]) * tick_index / 4.0 for tick_index in range(5)]
        add_panel_axes(
            lines,
            panel["title"],
            "Total Zr concentration (mM)",
            panel["y_label"],
            x_ticks,
            y_ticks,
            left,
            top,
            panel_width,
            panel_height,
            x_min,
            x_max,
            panel["y_min"],
            panel["y_max"],
        )
        for mode, color, key in panel["series"]:
            ys = [get_float(summary[(zr_value, mode)], key) for zr_value in zr_values]
            polyline = build_polyline(
                zr_values,
                ys,
                x_min,
                x_max,
                panel["y_min"],
                panel["y_max"],
                left,
                top,
                panel_width,
                panel_height,
            )
            if polyline:
                lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
            for zr_value, y_value in zip(zr_values, ys):
                if y_value is None:
                    continue
                x_coord = map_x(zr_value, x_min, x_max, left, panel_width)
                y_coord = map_y(y_value, panel["y_min"], panel["y_max"], top, panel_height)
                lines.append(f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="4.5" fill="{color}"/>')

    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def render_replicates_svg(per_run_rows, output_path):
    if not per_run_rows:
        return

    zr_values = sorted({float(row["zr_conc"]) for row in per_run_rows})
    width = 980
    height = 680
    left = 90
    top = 70
    panel_width = 820
    panel_height = 240
    panel_gap = 90

    panels = [
        ("max_entities_seen", "Replicate Max Entities"),
        ("time_to_10", "Replicate Time To 10 Entities (s)"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-size="22" font-weight="bold" fill="{TEXT_COLOR}">Replicate-Level Scan Variability</text>',
        f'<text x="{left}" y="52" font-size="12" fill="{TEXT_COLOR}">Each point is one KMC trajectory from the refined scan.</text>',
    ]

    legend_x = width - 190
    legend_y = 30
    lines.append(f'<circle cx="{legend_x}" cy="{legend_y - 3}" r="5" fill="{CONTROL_COLOR}"/>')
    lines.append(f'<text x="{legend_x + 12}" y="{legend_y + 1}" font-size="12" fill="{TEXT_COLOR}">control</text>')
    lines.append(f'<circle cx="{legend_x}" cy="{legend_y + 17}" r="5" fill="{DISTORTED_COLOR}"/>')
    lines.append(f'<text x="{legend_x + 12}" y="{legend_y + 21}" font-size="12" fill="{TEXT_COLOR}">distorted</text>')

    for panel_index, (key, title) in enumerate(panels):
        current_top = top + panel_index * (panel_height + panel_gap)
        values = [get_float(row, key) for row in per_run_rows if get_float(row, key) is not None]
        if not values:
            continue
        y_min = 0.0
        y_max = max(values) + (20.0 if "time" in key else 2.0)
        y_ticks = [y_min + (y_max - y_min) * tick_index / 4.0 for tick_index in range(5)]
        add_panel_axes(
            lines,
            title,
            "Total Zr concentration (mM)",
            title.split("(")[0].strip(),
            zr_values,
            y_ticks,
            left,
            current_top,
            panel_width,
            panel_height,
            min(zr_values),
            max(zr_values),
            y_min,
            y_max,
        )
        for mode, color, offset in (("control", CONTROL_COLOR, -8), ("distorted", DISTORTED_COLOR, 8)):
            mode_rows = [row for row in per_run_rows if row["mode"] == mode]
            for row in mode_rows:
                value = get_float(row, key)
                if value is None:
                    continue
                x_coord = map_x(float(row["zr_conc"]), min(zr_values), max(zr_values), left, panel_width) + offset
                y_coord = map_y(value, y_min, y_max, current_top, panel_height)
                lines.append(f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="4.5" fill="{color}" opacity="0.75"/>')

    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    zr_values, summary = load_summary_rows(args.summary_csv)
    per_run_rows = load_per_run_rows(args.per_run_csv)
    render_summary_svg(zr_values, summary, args.output_summary_svg)
    if args.output_replicates_svg is not None:
        render_replicates_svg(per_run_rows, args.output_replicates_svg)


if __name__ == "__main__":
    main()
