import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a PNG trace image from a replay trace JSON file."
    )
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=690)
    return parser.parse_args()


def resolve_path(path_raw: str):
    path = Path(path_raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def linear_map(value, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def draw_polyline_rgba(base_image, points, color, width):
    if len(points) < 2:
        return
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.line(points, fill=color, width=width, joint="curve")
    base_image.alpha_composite(overlay)


def main():
    args = parse_args()
    trace_json_path = resolve_path(args.trace_json)
    payload = json.loads(trace_json_path.read_text(encoding="utf-8"))

    if args.output_png is None:
        output_png_path = trace_json_path.with_suffix(".png")
    else:
        output_png_path = resolve_path(args.output_png)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    width = args.width
    height = args.height
    left = 88
    right = 36
    top = 72
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    run_payloads = payload["replay_runs"]
    mean_trace = payload["mean_trace"]
    mean_alignment = payload.get("mean_alignment", "time")

    species_meta = {
        "Zr12_AA": {"color": (194, 65, 12, 255), "thin": (194, 65, 12, 46), "label": "Zr12_AA"},
        "Zr6_AA": {"color": (37, 99, 235, 255), "thin": (37, 99, 235, 46), "label": "Zr6_AA"},
        "BDC": {"color": (21, 128, 61, 255), "thin": (21, 128, 61, 46), "label": "BDC"},
    }

    all_time_values = []
    all_count_values = []
    for run in run_payloads:
        all_time_values.extend(row["sim_time_seconds"] for row in run["traces"])
        all_count_values.extend(row["total_entities"] for row in run["traces"])
    x_min = 0.0
    x_max = max(all_time_values) if all_time_values else 1.0
    y_min = 0.0
    y_max = max(all_count_values) if all_count_values else 1.0
    y_tick_max = int(math.ceil(y_max / 50.0) * 50)
    if y_tick_max <= 0:
        y_tick_max = 50

    def x_px(value):
        return linear_map(value, x_min, x_max, left, left + plot_width)

    def y_px(value):
        return linear_map(value, y_min, y_tick_max, top + plot_height, top)

    image = Image.new("RGBA", (width, height), (251, 251, 249, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    # Background and plot area
    draw.rectangle([left, top, left + plot_width, top + plot_height], fill=(255, 255, 255, 255), outline=(226, 232, 240, 255), width=1)

    # Grid and axes
    for tick_index in range(7):
        y_value = y_tick_max * tick_index / 6.0
        y = y_px(y_value)
        draw.line([(left, y), (left + plot_width, y)], fill=(226, 232, 240, 255), width=1)
        label = str(int(round(y_value)))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((left - 14 - (bbox[2] - bbox[0]), y - 6), label, fill=(71, 85, 105, 255), font=font)

    for tick_index in range(7):
        x_value = x_max * tick_index / 6.0
        x = x_px(x_value)
        draw.line([(x, top), (x, top + plot_height)], fill=(238, 242, 247, 255), width=1)
        label = f"{x_value:.3f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x - 0.5 * (bbox[2] - bbox[0]), top + plot_height + 10), label, fill=(71, 85, 105, 255), font=font)

    draw.line([(left, top + plot_height), (left + plot_width, top + plot_height)], fill=(71, 85, 105, 255), width=2)
    draw.line([(left, top), (left, top + plot_height)], fill=(71, 85, 105, 255), width=2)

    # Titles
    title = "One-shot UiO-66 growth replay: Zr12 loss with Zr6 / BDC growth"
    subtitle = f"thin lines = individual replays; thick lines = {mean_alignment}-aligned mean trajectory"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 24), title, fill=(15, 23, 42, 255), font=font)
    draw.text(((width - (subtitle_bbox[2] - subtitle_bbox[0])) / 2, 44), subtitle, fill=(71, 85, 105, 255), font=font)

    # Thin replay trajectories
    for species, meta in species_meta.items():
        for run in run_payloads:
            points = [(x_px(row["sim_time_seconds"]), y_px(row[species])) for row in run["traces"]]
            draw_polyline_rgba(image, points, meta["thin"], width=1)

    # Thick mean trajectories
    for species, meta in species_meta.items():
        points = [(x_px(row["sim_time_seconds"]), y_px(row[f"{species}_mean"])) for row in mean_trace]
        draw_polyline_rgba(image, points, meta["color"], width=4)

    # Zr12 zero line
    zr12_zero_row = next((row for row in mean_trace if row["Zr12_AA_mean"] <= 0.0), None)
    if zr12_zero_row is not None:
        x = x_px(zr12_zero_row["sim_time_seconds"])
        dash_y0 = top
        while dash_y0 < top + plot_height:
            dash_y1 = min(dash_y0 + 7, top + plot_height)
            draw.line([(x, dash_y0), (x, dash_y1)], fill=(124, 58, 237, 255), width=2)
            dash_y0 += 13
        label = "mean Zr12 reaches 0"
        draw.text((max(left + 8, x - 120), top + 38), label, fill=(91, 33, 182, 255), font=font)

    # Legend
    legend_x = left + 14
    legend_y = top + 16
    for index, species in enumerate(["Zr12_AA", "Zr6_AA", "BDC"]):
        meta = species_meta[species]
        y = legend_y + index * 22
        draw.line([(legend_x, y), (legend_x + 24, y)], fill=meta["color"], width=3)
        draw.text((legend_x + 32, y - 7), meta["label"], fill=(51, 65, 85, 255), font=font)

    # Axis labels
    x_label = "Simulated time (s)"
    x_bbox = draw.textbbox((0, 0), x_label, font=font)
    draw.text(((left + plot_width / 2) - 0.5 * (x_bbox[2] - x_bbox[0]), height - 28), x_label, fill=(15, 23, 42, 255), font=font)
    y_label = "Entity count"
    y_label_image = Image.new("RGBA", (140, 24), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label_image, "RGBA")
    y_draw.text((0, 4), y_label, fill=(15, 23, 42, 255), font=font)
    y_label_image = y_label_image.rotate(90, expand=True)
    image.alpha_composite(y_label_image, (8, int(top + plot_height / 2 - y_label_image.size[1] / 2)))

    image.convert("RGB").save(output_png_path)
    print(output_png_path.as_posix())


if __name__ == "__main__":
    main()
