import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

import build_internal_zr12_seed as core_builder
import fragment_cleanup
import probe_zr6_only_growth as probe


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_JSON = (
    SCRIPT_DIR
    / "output"
    / "mixed_nuclei"
    / "zr6_only_growth_runs"
    / "oneshot_seed160_eq2_cap180_zr5000_pruned"
    / "oneshot_seed160_eq2_cap180_zr5000_pruned.summary.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a fixed Zr6-only growth run and plot Zr12/Zr6/BDC counts "
            "as a function of simulated time."
        )
    )
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON.as_posix())
    parser.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=1,
        help="Record a regular snapshot every N MC steps. Zr12-loss events are always recorded.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to replay from the summary.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def resolve_output_dir(summary_path: Path, output_dir_raw: str | None):
    if output_dir_raw is None:
        path = summary_path.parent
    else:
        path = Path(output_dir_raw)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(path_raw: str):
    path = Path(path_raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def snapshot_row(assembly, step, sim_time_seconds):
    counts = core_builder.entity_counts(assembly)
    return {
        "step": int(step),
        "sim_time_seconds": float(sim_time_seconds),
        "total_entities": int(counts["total_entities"]),
        "Zr6_AA": int(counts["Zr6_AA"]),
        "Zr12_AA": int(counts["Zr12_AA"]),
        "BDC": int(counts["BDC"]),
    }


def trace_growth_case(
    *,
    seed_path: Path,
    candidate: dict,
    config: dict,
    rng_seed: int,
    snapshot_every_steps: int,
):
    assembly = probe.load_seed_assembly(
        seed_path,
        zr6_percentage=1.0,
        bumping_threshold=config["bumping_threshold"],
    )

    seed_counts = probe.seed_counts(seed_path)
    entropy_table = probe.build_entropy_table(
        seed_counts["total_entities"] + config["max_entities_delta"] + 50,
        config["entropy_correction_coefficient"],
    )

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    max_entities = seed_counts["total_entities"] + config["max_entities_delta"]
    timing = 0.0
    step = 0
    traces = [snapshot_row(assembly, step=0, sim_time_seconds=0.0)]
    event_counts = {
        "link": 0,
        "grow_attempt": 0,
        "grow_success": 0,
        "grow_fail": 0,
        "remove": 0,
        "zr6_add": 0,
        "bdc_add": 0,
        "fragment_prune": 0,
        "fragment_pruned_entities": 0,
        "fragment_pruned_components": 0,
    }
    termination_reason = "requested_steps_completed"
    formate_benzoate_ratio = None

    while step < config["total_steps"]:
        sim_time_seconds = timing * candidate["exchange_rxn_time_seconds"]

        if len(assembly.entities) > max_entities:
            termination_reason = "max_entities_exceeded"
            break

        if step == 0 or (
            config["dissolution_update_interval_steps"] is not None
            and config["dissolution_update_interval_steps"] > 0
            and step % config["dissolution_update_interval_steps"] == 0
        ):
            _, formate_benzoate_ratio = probe.dissolution_probability(
                sim_time_seconds,
                candidate["equilibrium_constant_coefficient"],
                config["h2o_dmf_ratio"],
                candidate["capping_agent_conc"],
                candidate["linker_conc"],
            )

        before_counts = core_builder.entity_counts(assembly)
        step += 1
        flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(
            formate_benzoate_ratio,
            entropy_table,
            candidate["cluster_add_probability"],
        )

        if flag == 0:
            assembly.link_internal_carboxylate(selected_pair)
            event_counts["link"] += 1
        elif flag == 1:
            before_entities = len(assembly.entities)
            event_counts["grow_attempt"] += 1
            carboxylate_type = selected_carboxylate.carboxylate_type
            assembly.grow_one_step(selected_carboxylate)
            after_entities = len(assembly.entities)
            if after_entities == before_entities + 1:
                event_counts["grow_success"] += 1
                if carboxylate_type == "benzoate":
                    event_counts["zr6_add"] += 1
                else:
                    event_counts["bdc_add"] += 1
            else:
                event_counts["grow_fail"] += 1
        elif flag == -1:
            assembly.remove_linkage(selected_pair)
            event_counts["remove"] += 1
            prune_stats = fragment_cleanup.prune_disconnected_fragments(assembly)
            if prune_stats["removed_entity_count"] > 0:
                event_counts["fragment_prune"] += 1
                event_counts["fragment_pruned_entities"] += prune_stats["removed_entity_count"]
                event_counts["fragment_pruned_components"] += prune_stats["removed_component_count"]

        after_counts = core_builder.entity_counts(assembly)
        random_draw = max(random.random(), 1e-12)
        timing -= math.log(random_draw) / max(total_growth_rate, 1e-12)
        sim_time_seconds_after = timing * candidate["exchange_rxn_time_seconds"]

        should_record = (
            snapshot_every_steps > 0
            and step % snapshot_every_steps == 0
        ) or after_counts["Zr12_AA"] != before_counts["Zr12_AA"]
        if should_record:
            traces.append(snapshot_row(assembly, step=step, sim_time_seconds=sim_time_seconds_after))

    if traces[-1]["step"] != step:
        traces.append(snapshot_row(assembly, step=step, sim_time_seconds=timing * candidate["exchange_rxn_time_seconds"]))

    return {
        "seed_path": seed_path.as_posix(),
        "rng_seed": int(rng_seed),
        "termination_reason": termination_reason,
        "traces": traces,
        "final_counts": core_builder.entity_counts(assembly),
        "event_counts": event_counts,
    }


def aggregate_mean_trace(run_payloads):
    values_by_step = defaultdict(list)
    for run in run_payloads:
        for row in run["traces"]:
            values_by_step[row["step"]].append(row)

    mean_rows = []
    for step in sorted(values_by_step):
        rows = values_by_step[step]
        mean_rows.append(
            {
                "step": int(step),
                "sim_time_seconds_mean": float(np.mean([row["sim_time_seconds"] for row in rows])),
                "Zr6_AA_mean": float(np.mean([row["Zr6_AA"] for row in rows])),
                "Zr12_AA_mean": float(np.mean([row["Zr12_AA"] for row in rows])),
                "BDC_mean": float(np.mean([row["BDC"] for row in rows])),
                "total_entities_mean": float(np.mean([row["total_entities"] for row in rows])),
                "replicate_count": int(len(rows)),
            }
        )
    return mean_rows


def svg_polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def linear_map(value, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def make_svg_plot(run_payloads, mean_trace, svg_path: Path):
    species_meta = {
        "Zr12_AA": {"color": "#c2410c", "label": "Zr12_AA"},
        "Zr6_AA": {"color": "#2563eb", "label": "Zr6_AA"},
        "BDC": {"color": "#15803d", "label": "BDC"},
    }

    width = 1120
    height = 690
    left = 88
    right = 36
    top = 72
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom

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

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fbfbf9" />',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>',
        '<text x="50%" y="34" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="23" fill="#0f172a">'
        + escape("One-shot UiO-66 growth replay: Zr12 loss with Zr6 / BDC growth")
        + "</text>",
        '<text x="50%" y="58" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#475569">'
        + escape("thin lines = individual replays; thick lines = mean trajectory")
        + "</text>",
    ]

    for tick_index in range(7):
        y_value = y_tick_max * tick_index / 6.0
        y = y_px(y_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">'
            f"{int(round(y_value))}</text>"
        )

    for tick_index in range(7):
        x_value = x_max * tick_index / 6.0
        x = x_px(x_value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#eef2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">'
            f"{x_value:.3f}</text>"
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" '
        'stroke="#475569" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#475569" stroke-width="1.2"/>'
    )

    for species, meta in species_meta.items():
        for run in run_payloads:
            points = [(x_px(row["sim_time_seconds"]), y_px(row[species])) for row in run["traces"]]
            parts.append(
                f'<polyline points="{svg_polyline(points)}" fill="none" stroke="{meta["color"]}" '
                'stroke-width="1.0" stroke-opacity="0.18" stroke-linecap="round" stroke-linejoin="round"/>'
            )

    for species, meta in species_meta.items():
        points = [
            (x_px(row["sim_time_seconds_mean"]), y_px(row[f"{species}_mean"]))
            for row in mean_trace
        ]
        parts.append(
            f'<polyline points="{svg_polyline(points)}" fill="none" stroke="{meta["color"]}" '
            'stroke-width="3.1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    zr12_zero_row = next((row for row in mean_trace if row["Zr12_AA_mean"] <= 0.0), None)
    if zr12_zero_row is not None:
        x = x_px(zr12_zero_row["sim_time_seconds_mean"])
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        parts.append(
            f'<text x="{max(left + 8, x - 120):.2f}" y="{top + 48}" '
            'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#5b21b6">'
            + escape("mean Zr12 reaches 0")
            + "</text>"
        )

    legend_x = left + 14
    legend_y = top + 16
    for index, species in enumerate(["Zr12_AA", "Zr6_AA", "BDC"]):
        meta = species_meta[species]
        y = legend_y + index * 22
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" '
            f'stroke="{meta["color"]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#334155">{escape(meta["label"])}</text>'
        )

    parts.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" '
        'font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#0f172a">'
        + escape("Simulated time (s)")
        + "</text>"
    )
    parts.append(
        f'<text x="22" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        'font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#0f172a" '
        f'transform="rotate(-90 22 {top + plot_height / 2:.2f})">'
        + escape("Entity count")
        + "</text>"
    )
    parts.append("</svg>")

    svg_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    summary_path, summary_payload = load_summary(args.summary_json)
    output_dir = resolve_output_dir(summary_path, args.output_dir)

    run_rows = summary_payload["runs"]
    if args.max_runs is not None:
        run_rows = run_rows[: args.max_runs]

    candidate = summary_payload["candidate"]
    config = summary_payload["config"]
    basename = args.basename or f"{summary_path.stem}_trace"

    replay_runs = []
    for run in run_rows:
        replay_payload = trace_growth_case(
            seed_path=Path(run["seed_path"]),
            candidate=candidate,
            config=config,
            rng_seed=run["rng_seed"],
            snapshot_every_steps=args.snapshot_every_steps,
        )
        replay_payload["stored_final_counts"] = run["end_counts"]
        replay_payload["stored_json_path"] = run.get("json_path")
        replay_runs.append(replay_payload)

    mean_trace = aggregate_mean_trace(replay_runs)

    svg_path = output_dir / f"{basename}.svg"
    json_path = output_dir / f"{basename}.json"

    make_svg_plot(replay_runs, mean_trace, svg_path)

    json_payload = {
        "summary_json": summary_path.as_posix(),
        "snapshot_every_steps": args.snapshot_every_steps,
        "candidate": candidate,
        "config": config,
        "replay_runs": replay_runs,
        "mean_trace": mean_trace,
        "svg_path": svg_path.as_posix(),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_json": summary_path.as_posix(),
                "svg_path": svg_path.as_posix(),
                "json_path": json_path.as_posix(),
                "replayed_runs": len(replay_runs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
