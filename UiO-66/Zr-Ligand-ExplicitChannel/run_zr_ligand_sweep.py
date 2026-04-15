import argparse
import csv
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_ZR_VALUES = [16.0, 32.0, 64.0]
CONTROL_COLOR = "#1f77b4"
DISTORTED_COLOR = "#d95f02"


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched Zr sweep for control vs distorted-linker cases and generate "
            "entity-vs-time growth-curve SVGs."
        )
    )
    parser.add_argument(
        "--zr-values",
        default=",".join(str(value) for value in DEFAULT_ZR_VALUES),
        help="Comma-separated Zr concentrations to simulate.",
    )
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--max-entities", type=int, default=60)
    parser.add_argument("--output-root", default="output/zr_ligand_sweep")
    parser.add_argument("--output-inter", type=int, default=0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--dissolution-update-interval-steps", type=int, default=1000000)
    parser.add_argument("--zr6-percentage", type=float, default=1.0)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.789387907185137)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=1.32975172557788)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=0.0)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--linker-conc", type=float, default=4.0)
    parser.add_argument("--bumping-threshold", type=float, default=2.0)
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
        help="Optional direct override for the effective distorted-linker association constant.",
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
        help=(
            "Optional extra irreversible sink for the older two-step model. "
            "Leave at 0 to keep only the 1:1 prebound Zr-BDC species."
        ),
    )
    parser.add_argument("--distorted-num-sites-on-cluster", type=int, default=12)
    parser.add_argument("--distorted-num-sites-on-linker", type=int, default=2)
    return parser.parse_args()


def run_case(script_path, base_args, output_root, zr_value, distorted_enabled):
    label = f"{'distorted' if distorted_enabled else 'control'}_zr{int(zr_value)}"
    command = [
        sys.executable,
        str(script_path),
        "--zr6-percentage", str(base_args.zr6_percentage),
        "--zr-conc", str(zr_value),
        "--entropy-correction-coefficient", str(base_args.entropy_correction_coefficient),
        "--equilibrium-constant-coefficient", str(base_args.equilibrium_constant_coefficient),
        "--h2o-dmf-ratio", str(base_args.h2o_dmf_ratio),
        "--capping-agent-conc", str(base_args.capping_agent_conc),
        "--linker-conc", str(base_args.linker_conc),
        "--total-steps", str(base_args.total_steps),
        "--bumping-threshold", str(base_args.bumping_threshold),
        "--max-entities", str(base_args.max_entities),
        "--output-inter", str(base_args.output_inter),
        "--exchange-rxn-time-seconds", str(base_args.exchange_rxn_time_seconds),
        "--dissolution-update-interval-steps", str(base_args.dissolution_update_interval_steps),
        "--output-root", str(output_root),
        "--basename", label,
        "--distorted-chemistry-model", str(base_args.distorted_chemistry_model),
        "--distorted-num-sites-on-cluster", str(base_args.distorted_num_sites_on_cluster),
        "--distorted-num-sites-on-linker", str(base_args.distorted_num_sites_on_linker),
    ]
    if distorted_enabled:
        command.append("--enable-distorted-linker")
    if base_args.distorted_ligand_association_constant is not None:
        command.extend(
            [
                "--distorted-ligand-association-constant",
                str(base_args.distorted_ligand_association_constant),
            ]
        )
    if base_args.distorted_site_equilibrium_constant is not None:
        command.extend(
            [
                "--distorted-site-equilibrium-constant",
                str(base_args.distorted_site_equilibrium_constant),
            ]
        )
    command.extend(
        [
            "--distorted-second-step-equivalents",
            str(base_args.distorted_second_step_equivalents),
        ]
    )
    subprocess.run(command, check=True, cwd=script_path.parent)
    return output_root / label


def load_trace(run_dir):
    with open(run_dir / "entities_number.pkl", "rb") as handle:
        data = pickle.load(handle)
    with open(run_dir / "launcher_config.json", "r", encoding="utf-8") as handle:
        launcher = json.load(handle)
    exchange_time = launcher["config"]["EXCHANGE_RXN_TIME_SECONDS"]
    times = [row[0] * exchange_time for row in data]
    entities = [row[1] for row in data]
    return {
        "times_seconds": times,
        "entities": entities,
        "exchange_rxn_time_seconds": exchange_time,
    }


def load_run_summary(run_dir):
    with open(run_dir / "run_summary.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_chemistry_summary(run_dir):
    chemistry_path = run_dir / "chemistry_summary.json"
    if not chemistry_path.exists():
        return None
    with open(chemistry_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def polyline_points(times, values, x_min, x_max, y_max, left, top, width, height):
    if not times:
        return ""
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max, 1)
    points = []
    for time_value, entity_value in zip(times, values):
        x_coord = left + (time_value - x_min) / x_span * width
        y_coord = top + height - (entity_value / y_span) * height
        points.append(f"{x_coord:.2f},{y_coord:.2f}")
    return " ".join(points)


def build_growth_curve_svg(results_by_zr, output_path):
    panel_width = 360
    panel_height = 220
    left_margin = 70
    right_margin = 30
    top_margin = 50
    bottom_margin = 55
    panel_gap = 40
    width = left_margin + right_margin + panel_width
    height = top_margin + bottom_margin + len(results_by_zr) * panel_height + (len(results_by_zr) - 1) * panel_gap

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #222; }',
        '.title { font-size: 18px; font-weight: bold; }',
        '.subtitle { font-size: 12px; fill: #555; }',
        '.axis { stroke: #333; stroke-width: 1.2; }',
        '.grid { stroke: #ddd; stroke-width: 1; }',
        '.legend { font-size: 12px; }',
        '.panel-title { font-size: 14px; font-weight: bold; }',
        '.tick { font-size: 11px; fill: #555; }',
        '</style>',
        f'<text x="{left_margin}" y="28" class="title">UiO-66 Growth Curves</text>',
        f'<text x="{left_margin}" y="44" class="subtitle">Number of entities vs time. Control vs prebound Zr-BDC branch derived from KC.</text>',
    ]

    legend_y = top_margin - 10
    svg_lines.append(f'<line x1="{width - 230}" y1="{legend_y}" x2="{width - 200}" y2="{legend_y}" stroke="{CONTROL_COLOR}" stroke-width="3"/>')
    svg_lines.append(f'<text x="{width - 190}" y="{legend_y + 4}" class="legend">Control</text>')
    svg_lines.append(f'<line x1="{width - 120}" y1="{legend_y}" x2="{width - 90}" y2="{legend_y}" stroke="{DISTORTED_COLOR}" stroke-width="3"/>')
    svg_lines.append(f'<text x="{width - 80}" y="{legend_y + 4}" class="legend">Distorted</text>')

    for index, (zr_value, payload) in enumerate(results_by_zr):
        top = top_margin + index * (panel_height + panel_gap)
        bottom = top + panel_height
        x_min = 0.0
        x_max = max(
            max(payload["control_trace"]["times_seconds"], default=0.0),
            max(payload["distorted_trace"]["times_seconds"], default=0.0),
            1.0,
        )
        y_max = max(
            max(payload["control_trace"]["entities"], default=0),
            max(payload["distorted_trace"]["entities"], default=0),
            1,
        )

        svg_lines.append(f'<text x="{left_margin}" y="{top - 12}" class="panel-title">Zr = {zr_value:g} mM</text>')
        for grid_index in range(5):
            y_value = y_max * grid_index / 4.0
            y_coord = top + panel_height - (y_value / y_max) * panel_height
            svg_lines.append(f'<line x1="{left_margin}" y1="{y_coord:.2f}" x2="{left_margin + panel_width}" y2="{y_coord:.2f}" class="grid"/>')
            svg_lines.append(f'<text x="{left_margin - 10}" y="{y_coord + 4:.2f}" text-anchor="end" class="tick">{int(round(y_value))}</text>')
        for grid_index in range(5):
            x_value = x_max * grid_index / 4.0
            x_coord = left_margin + (x_value / x_max) * panel_width
            svg_lines.append(f'<line x1="{x_coord:.2f}" y1="{top}" x2="{x_coord:.2f}" y2="{bottom}" class="grid"/>')
            svg_lines.append(f'<text x="{x_coord:.2f}" y="{bottom + 18}" text-anchor="middle" class="tick">{x_value:.0f}</text>')

        svg_lines.append(f'<line x1="{left_margin}" y1="{bottom}" x2="{left_margin + panel_width}" y2="{bottom}" class="axis"/>')
        svg_lines.append(f'<line x1="{left_margin}" y1="{top}" x2="{left_margin}" y2="{bottom}" class="axis"/>')

        control_points = polyline_points(
            payload["control_trace"]["times_seconds"],
            payload["control_trace"]["entities"],
            x_min,
            x_max,
            y_max,
            left_margin,
            top,
            panel_width,
            panel_height,
        )
        distorted_points = polyline_points(
            payload["distorted_trace"]["times_seconds"],
            payload["distorted_trace"]["entities"],
            x_min,
            x_max,
            y_max,
            left_margin,
            top,
            panel_width,
            panel_height,
        )
        svg_lines.append(f'<polyline fill="none" stroke="{CONTROL_COLOR}" stroke-width="2.2" points="{control_points}"/>')
        svg_lines.append(f'<polyline fill="none" stroke="{DISTORTED_COLOR}" stroke-width="2.2" points="{distorted_points}"/>')

        annotation_x = left_margin + panel_width - 5
        control_fail = payload["control_summary"]["event_num_grow_fail"]
        distorted_fail = payload["distorted_summary"]["event_num_grow_fail"]
        distorted_fraction = payload["distorted_summary"].get(
            "prebound_zr_bdc_fraction",
            payload["distorted_summary"]["distorted_linker_fraction"],
        )
        off_pathway_fraction = payload["distorted_summary"].get("off_pathway_linker_fraction", 0.0)
        svg_lines.append(
            f'<text x="{annotation_x}" y="{top + 16}" text-anchor="end" class="subtitle">failures: control {control_fail}, distorted {distorted_fail}</text>'
        )
        svg_lines.append(
            f'<text x="{annotation_x}" y="{top + 32}" text-anchor="end" class="subtitle">prebound fraction: {distorted_fraction:.3f}, off-pathway: {off_pathway_fraction:.3f}</text>'
        )

    svg_lines.append(f'<text x="{left_margin + panel_width / 2:.2f}" y="{height - 14}" text-anchor="middle" class="subtitle">Time (s)</text>')
    svg_lines.append(
        f'<text x="18" y="{top_margin + (height - top_margin - bottom_margin) / 2:.2f}" transform="rotate(-90 18 {top_margin + (height - top_margin - bottom_margin) / 2:.2f})" text-anchor="middle" class="subtitle">Number of entities</text>'
    )
    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_summary_csv(results_by_zr, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "zr_conc",
                "mode",
                "final_entities",
                "max_entities_seen",
                "simulated_time_seconds",
                "event_num_grow",
                "event_num_grow_success",
                "event_num_grow_fail",
                "event_num_remove",
                "distorted_linker_fraction",
                "off_pathway_linker_fraction",
                "effective_zr6_conc",
                "effective_linker_conc",
                "external_addition_activity",
            ],
        )
        writer.writeheader()
        for zr_value, payload in results_by_zr:
            for mode in ("control", "distorted"):
                summary = payload[f"{mode}_summary"]
                trace = payload[f"{mode}_trace"]
                writer.writerow(
                    {
                        "zr_conc": zr_value,
                        "mode": mode,
                        "final_entities": summary["final_entities"],
                        "max_entities_seen": max(trace["entities"]) if trace["entities"] else 0,
                        "simulated_time_seconds": summary["simulated_time_seconds"],
                        "event_num_grow": summary["event_num_grow"],
                        "event_num_grow_success": summary["event_num_grow_success"],
                        "event_num_grow_fail": summary["event_num_grow_fail"],
                        "event_num_remove": summary["event_num_remove"],
                        "distorted_linker_fraction": summary["distorted_linker_fraction"],
                        "off_pathway_linker_fraction": summary.get("off_pathway_linker_fraction", 0.0),
                        "effective_zr6_conc": summary.get("effective_zr6_conc", 0.0),
                        "effective_linker_conc": summary.get("effective_linker_conc", 0.0),
                        "external_addition_activity": summary.get("external_addition_activity", 1.0),
                    }
                )


def main():
    args = parse_args()
    zr_values = parse_csv_floats(args.zr_values)
    root = Path(args.output_root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "run_zr_ligand_case.py"

    results_by_zr = []
    for zr_value in zr_values:
        control_dir = run_case(script_path, args, root, zr_value, distorted_enabled=False)
        distorted_dir = run_case(script_path, args, root, zr_value, distorted_enabled=True)
        control_trace = load_trace(control_dir)
        distorted_trace = load_trace(distorted_dir)
        control_summary = load_run_summary(control_dir)
        distorted_summary = load_run_summary(distorted_dir)
        distorted_chemistry = load_chemistry_summary(distorted_dir)
        payload = {
            "control_dir": control_dir.as_posix(),
            "distorted_dir": distorted_dir.as_posix(),
            "control_trace": control_trace,
            "distorted_trace": distorted_trace,
            "control_summary": control_summary,
            "distorted_summary": distorted_summary,
            "distorted_chemistry": distorted_chemistry,
        }
        results_by_zr.append((zr_value, payload))

    plot_path = root / "growth_curves.svg"
    build_growth_curve_svg(results_by_zr, plot_path)
    csv_path = root / "sweep_summary.csv"
    write_summary_csv(results_by_zr, csv_path)

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plot_path": plot_path.as_posix(),
        "csv_path": csv_path.as_posix(),
        "results": [
            {
                "zr_conc": zr_value,
                "control_dir": payload["control_dir"],
                "distorted_dir": payload["distorted_dir"],
                "control_summary": payload["control_summary"],
                "distorted_summary": payload["distorted_summary"],
                "distorted_chemistry": payload["distorted_chemistry"],
            }
            for zr_value, payload in results_by_zr
        ],
    }
    summary_path = root / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
