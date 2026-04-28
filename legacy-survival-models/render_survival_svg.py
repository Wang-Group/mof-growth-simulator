import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a pure-SVG Kaplan-Meier style survival plot from a flat CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input flat survival CSV.")
    parser.add_argument("--group-column", required=True, help="Column used to split curves.")
    parser.add_argument(
        "--time-column",
        default="time_seconds",
        help="Observed time column. Default: time_seconds.",
    )
    parser.add_argument(
        "--event-column",
        default="event",
        help="Event indicator column (1=event, 0=censored). Default: event.",
    )
    parser.add_argument("--output-svg", required=True, help="Output SVG path.")
    parser.add_argument(
        "--output-km-csv",
        default=None,
        help="Optional output CSV containing the Kaplan-Meier step data.",
    )
    parser.add_argument(
        "--output-risk-csv",
        default=None,
        help="Optional output CSV containing the at-risk table values.",
    )
    parser.add_argument(
        "--title",
        default="Kaplan-Meier style survival plot",
        help="Plot title.",
    )
    return parser.parse_args()


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def maybe_numeric(value):
    try:
        return (0, float(value))
    except ValueError:
        return (1, str(value))


def load_grouped_rows(csv_path, group_column, time_column, event_column):
    grouped = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get(group_column):
                continue
            time_value = row.get(time_column)
            if not time_value:
                continue
            grouped[row[group_column]].append(
                {
                    "time": float(time_value),
                    "event": int(float(row[event_column])),
                }
            )
    return grouped


def compute_kaplan_meier(group_rows):
    rows = sorted(group_rows, key=lambda item: (item["time"], -item["event"]))
    n_at_risk = len(rows)
    survival = 1.0
    km_points = [{"time": 0.0, "survival": 1.0, "n_at_risk": n_at_risk, "events": 0, "censored": 0}]
    censor_marks = []

    index = 0
    while index < len(rows):
        time_value = rows[index]["time"]
        events = 0
        censored = 0
        while index < len(rows) and rows[index]["time"] == time_value:
            if rows[index]["event"] == 1:
                events += 1
            else:
                censored += 1
            index += 1

        if events:
            survival *= (1.0 - events / n_at_risk)
            km_points.append(
                {
                    "time": time_value,
                    "survival": survival,
                    "n_at_risk": n_at_risk,
                    "events": events,
                    "censored": censored,
                }
            )

        for _ in range(censored):
            censor_marks.append({"time": time_value, "survival": survival})

        n_at_risk -= events + censored

    return km_points, censor_marks


def count_at_risk(group_rows, time_value):
    return sum(1 for row in group_rows if row["time"] >= time_value)


def nice_log_ticks(time_min, time_max):
    exp_min = int(math.floor(math.log10(time_min)))
    exp_max = int(math.ceil(math.log10(time_max)))
    ticks = []
    for exponent in range(exp_min, exp_max + 1):
        for base in (1, 3):
            tick_value = base * (10 ** exponent)
            if time_min <= tick_value <= time_max:
                ticks.append(float(tick_value))
    return ticks or [time_min, time_max]


def choose_risk_times(all_times):
    time_min = min(t for t in all_times if t > 0)
    time_max = max(all_times)
    ticks = nice_log_ticks(time_min, time_max)
    if len(ticks) > 6:
        ticks = ticks[::2]
    return ticks[:6]


def map_x_log(value, x_min, x_max, left, width):
    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    log_value = math.log10(max(value, x_min))
    fraction = (log_value - log_min) / max(log_max - log_min, 1e-12)
    return left + fraction * width


def map_y_linear(value, y_min, y_max, top, height):
    fraction = (value - y_min) / max(y_max - y_min, 1e-12)
    return top + height - fraction * height


def format_time_tick(value):
    if value >= 1e6:
        return f"{value / 1e6:.1f}e6"
    if value >= 1e3:
        return f"{value / 1e3:.0f}e3"
    return f"{value:.0f}"


def km_polyline(points, x_min, x_max, left, top, width, height):
    if not points:
        return ""
    path = []
    current_survival = points[0]["survival"]
    path.append((left, map_y_linear(current_survival, 0.0, 1.0, top, height)))

    for point in points[1:]:
        step_x = map_x_log(point["time"], x_min, x_max, left, width)
        prev_y = map_y_linear(current_survival, 0.0, 1.0, top, height)
        path.append((step_x, prev_y))
        current_survival = point["survival"]
        new_y = map_y_linear(current_survival, 0.0, 1.0, top, height)
        path.append((step_x, new_y))

    path.append((map_x_log(x_max, x_min, x_max, left, width), map_y_linear(current_survival, 0.0, 1.0, top, height)))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in path)


def render_svg(grouped_rows, output_svg, title):
    group_values = sorted(grouped_rows, key=maybe_numeric)
    km_by_group = {}
    censor_by_group = {}
    all_times = []

    for group_value in group_values:
        km_points, censor_marks = compute_kaplan_meier(grouped_rows[group_value])
        km_by_group[group_value] = km_points
        censor_by_group[group_value] = censor_marks
        all_times.extend(row["time"] for row in grouped_rows[group_value])

    positive_times = [t for t in all_times if t > 0]
    x_min = 10 ** math.floor(math.log10(min(positive_times)))
    x_max = 10 ** math.ceil(math.log10(max(positive_times)))
    risk_times = choose_risk_times(positive_times)

    width = 1280
    height = max(860, 620 + 36 * len(group_values))
    left = 110
    right = 60
    top = 90
    plot_width = width - left - right
    plot_height = 430
    risk_top = 610
    row_height = 34

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]
    color_by_group = {group: colors[index % len(colors)] for index, group in enumerate(group_values)}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; }',
        '</style>',
        f'<text x="{left}" y="38" font-size="27" font-weight="bold" fill="#1f1f1f">{svg_escape(title)}</text>',
        '<text x="{left}" y="63" font-size="13" fill="#555555">Events are trajectory completions; unfinished trajectories are treated as right-censored at their final observed time.</text>',
    ]

    for tick in nice_log_ticks(x_min, x_max):
        x = map_x_log(tick, x_min, x_max, left, plot_width)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 24:.2f}" text-anchor="middle" font-size="12" fill="#333333">{format_time_tick(tick)}</text>')
    for y_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = map_y_linear(y_tick, 0.0, 1.0, top, plot_height)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#333333">{y_tick:.2f}</text>')

    lines.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="1.5"/>')
    lines.append(f'<text x="{left + plot_width / 2:.2f}" y="{top + plot_height + 54:.2f}" text-anchor="middle" font-size="13" fill="#333333">Observed time (s, log scale)</text>')
    y_center = top + plot_height / 2.0
    lines.append(
        f'<text x="{left - 66}" y="{y_center:.2f}" transform="rotate(-90 {left - 66} {y_center:.2f})" text-anchor="middle" font-size="13" fill="#333333">Survival probability</text>'
    )

    for group_value in group_values:
        color = color_by_group[group_value]
        polyline = km_polyline(km_by_group[group_value], x_min, x_max, left, top, plot_width, plot_height)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for mark in censor_by_group[group_value]:
            x = map_x_log(mark["time"], x_min, x_max, left, plot_width)
            y = map_y_linear(mark["survival"], 0.0, 1.0, top, plot_height)
            lines.append(f'<line x1="{x:.2f}" y1="{y - 6:.2f}" x2="{x:.2f}" y2="{y + 6:.2f}" stroke="{color}" stroke-width="1.5"/>')

    legend_x = left + plot_width - 210
    legend_y = top + 14
    legend_height = 28 + 22 * len(group_values)
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="198" height="{legend_height}" fill="white" stroke="#cccccc"/>')
    lines.append(f'<text x="{legend_x + 10}" y="{legend_y + 18}" font-size="12" font-weight="bold" fill="#333333">Group</text>')
    for index, group_value in enumerate(group_values):
        y = legend_y + 38 + index * 22
        color = color_by_group[group_value]
        lines.append(f'<line x1="{legend_x + 10}" y1="{y - 4}" x2="{legend_x + 28}" y2="{y - 4}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 36}" y="{y}" font-size="12" fill="#333333">{svg_escape(group_value)}</text>')

    lines.append(f'<text x="{left}" y="{risk_top - 28}" font-size="18" font-weight="bold" fill="#1f1f1f">Number at risk</text>')
    lines.append(f'<text x="{left}" y="{risk_top - 8}" font-size="12" fill="#555555">Count of observations still at risk at each reference time.</text>')
    header_y = risk_top + 16
    lines.append(f'<text x="{left}" y="{header_y}" font-size="12" font-weight="bold" fill="#333333">Group</text>')
    for time_value in risk_times:
        x = map_x_log(time_value, x_min, x_max, left + 120, plot_width - 120)
        lines.append(f'<text x="{x:.2f}" y="{header_y}" text-anchor="middle" font-size="12" font-weight="bold" fill="#333333">{format_time_tick(time_value)}</text>')

    table_left = left
    table_right = left + plot_width
    for index, group_value in enumerate(group_values):
        y = risk_top + 32 + index * row_height
        if index % 2 == 0:
            lines.append(f'<rect x="{table_left}" y="{y - 16}" width="{table_right - table_left}" height="{row_height}" fill="#f8f6ef"/>')
        color = color_by_group[group_value]
        lines.append(f'<text x="{left}" y="{y}" font-size="12" fill="{color}" font-weight="bold">{svg_escape(group_value)}</text>')
        for time_value in risk_times:
            x = map_x_log(time_value, x_min, x_max, left + 120, plot_width - 120)
            risk = count_at_risk(grouped_rows[group_value], time_value)
            lines.append(f'<text x="{x:.2f}" y="{y}" text-anchor="middle" font-size="12" fill="#333333">{risk}</text>')

    lines.append("</svg>")
    Path(output_svg).write_text("\n".join(lines), encoding="utf-8")
    return km_by_group, risk_times


def write_km_csv(km_by_group, output_path):
    fieldnames = ["group", "time_seconds", "survival_probability", "n_at_risk", "events_at_time", "censored_at_time"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_value in sorted(km_by_group, key=maybe_numeric):
            for row in km_by_group[group_value]:
                writer.writerow(
                    {
                        "group": group_value,
                        "time_seconds": row["time"],
                        "survival_probability": row["survival"],
                        "n_at_risk": row["n_at_risk"],
                        "events_at_time": row["events"],
                        "censored_at_time": row["censored"],
                    }
                )


def write_risk_csv(grouped_rows, risk_times, output_path):
    fieldnames = ["group"] + [f"risk_at_{time_value:g}_s" for time_value in risk_times]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_value in sorted(grouped_rows, key=maybe_numeric):
            row = {"group": group_value}
            for time_value in risk_times:
                row[f"risk_at_{time_value:g}_s"] = count_at_risk(grouped_rows[group_value], time_value)
            writer.writerow(row)


def main():
    args = parse_args()
    grouped_rows = load_grouped_rows(
        csv_path=args.input_csv,
        group_column=args.group_column,
        time_column=args.time_column,
        event_column=args.event_column,
    )
    km_by_group, risk_times = render_svg(grouped_rows, args.output_svg, args.title)

    if args.output_km_csv:
        write_km_csv(km_by_group, args.output_km_csv)

    if args.output_risk_csv:
        write_risk_csv(grouped_rows, risk_times, args.output_risk_csv)


if __name__ == "__main__":
    main()
