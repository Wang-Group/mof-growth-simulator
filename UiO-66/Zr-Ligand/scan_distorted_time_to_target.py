import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from UiO66_Assembly_Large_Correction_20250811 import Assembly, Zr6_AA
from distorted_ligand_model import (
    PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    compute_prebound_chemistry_state,
    solve_linker_carboxylate_to_acid_ratio,
    zr6_cluster_add_probability,
)


DEFAULT_ZR_VALUES = [32, 40, 48, 64, 80, 96, 128]

END_DMF_DECOMPOSITION_CONC = 560.0
EXP_TIME_HOURS = 3.0
CORRECTION_TERM_FOR_DEPROTONATION = 10 ** (3.51 - 4.74)
H2O_PURE = 55500.0
DMF_PURE = 12900.0
H2O_FORMATE_COEFFICIENT = 0.01
DMF_FORMATE_COEFFICIENT = 0.01
NUM_CARBOXYLATE_ON_LINKER = 2
EQUILIBRIUM_CONSTANT = 1.64
ENTROPY_GAIN = 30.9


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run distorted/prebound UiO-66 trajectories until a target entity count "
            "is reached or a very large max-step cap is hit."
        )
    )
    parser.add_argument(
        "--zr-values",
        default=",".join(str(value) for value in DEFAULT_ZR_VALUES),
        help="Comma-separated Zr concentrations to scan.",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--target-entities", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10_000_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Rewrite the per-run and summary outputs after this many finished jobs.",
    )
    parser.add_argument("--output-root", default="output/distorted_time_to_target")
    parser.add_argument("--zr6-percentage", type=float, default=1.0)
    parser.add_argument("--entropy-correction-coefficient", type=float, default=0.789387907185137)
    parser.add_argument("--equilibrium-constant-coefficient", type=float, default=1.32975172557788)
    parser.add_argument("--h2o-dmf-ratio", type=float, default=0.0)
    parser.add_argument("--capping-agent-conc", type=float, default=300.0)
    parser.add_argument("--linker-conc", type=float, default=4.0)
    parser.add_argument("--bumping-threshold", type=float, default=2.0)
    parser.add_argument("--exchange-rxn-time-seconds", type=float, default=0.1)
    parser.add_argument("--dissolution-update-interval-steps", type=int, default=1000000)
    parser.add_argument(
        "--distorted-chemistry-model",
        default=PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
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
        help="Legacy sink parameter. Keep at 0 for the current single-step model.",
    )
    parser.add_argument("--distorted-num-sites-on-cluster", type=int, default=12)
    parser.add_argument("--distorted-num-sites-on-linker", type=int, default=2)
    return parser.parse_args()


def entropy_assembly(corrected_entropy_gain, num_entity, limit=150):
    entity_extra_gain = 0.35
    if num_entity >= limit:
        return math.exp(corrected_entropy_gain)
    return math.exp(corrected_entropy_gain + entity_extra_gain * (1.0 - math.log(num_entity + 1) / math.log(limit)))


def build_entropy_correction_table(entropy_correction_coefficient, size=5000):
    corrected_entropy_gain = ENTROPY_GAIN * entropy_correction_coefficient
    return [entropy_assembly(corrected_entropy_gain, entity_count) for entity_count in range(size)]


def build_case_inputs(job):
    dmf_conc = DMF_PURE / (1.0 + job["h2o_dmf_ratio"])
    h2o_conc = H2O_PURE * job["h2o_dmf_ratio"] / (1.0 + job["h2o_dmf_ratio"])
    h2o_formate_coord_power = h2o_conc / job["capping_agent_conc"] * H2O_FORMATE_COEFFICIENT
    dmf_formate_coord_power = dmf_conc / job["capping_agent_conc"] * DMF_FORMATE_COEFFICIENT
    effective_equilibrium_constant = (
        job["equilibrium_constant_coefficient"]
        * EQUILIBRIUM_CONSTANT
        / (1.0 + h2o_formate_coord_power + dmf_formate_coord_power)
    )
    distorted_state = compute_prebound_chemistry_state(
        zr_conc=job["zr_conc"],
        linker_conc=job["linker_conc"],
        equilibrium_constant_coefficient=job["equilibrium_constant_coefficient"],
        h2o_dmf_ratio=job["h2o_dmf_ratio"],
        capping_agent_conc=job["capping_agent_conc"],
        zr6_percentage=job["zr6_percentage"],
        model_name=job["distorted_chemistry_model"],
        association_constant_override=job["distorted_ligand_association_constant"],
        site_equilibrium_constant_override=job["distorted_site_equilibrium_constant"],
        dimethylamine_conc=0.0,
        second_step_equivalents=job["distorted_second_step_equivalents"],
        num_sites_on_cluster=job["distorted_num_sites_on_cluster"],
        num_sites_on_linker=job["distorted_num_sites_on_linker"],
    )
    linker_conc_for_growth = max(float(distorted_state["free_linker_conc"]), 0.0)
    zr6_conc_for_growth = max(float(distorted_state["free_zr6_conc"]), 0.0)
    effective_zr6_fraction_for_growth = (
        zr6_conc_for_growth / distorted_state["total_zr6_conc"]
        if distorted_state["total_zr6_conc"] > 0.0
        else 0.0
    )
    zr6_conc_adding_probability = zr6_cluster_add_probability(
        zr6_conc=zr6_conc_for_growth,
        linker_conc=linker_conc_for_growth,
        num_carboxylate_on_linker=NUM_CARBOXYLATE_ON_LINKER,
    )
    external_addition_activity = float(
        np.sqrt(
            effective_zr6_fraction_for_growth * distorted_state["free_linker_fraction"]
        )
    )
    external_addition_activity = min(max(external_addition_activity, 0.0), 1.0)
    return {
        "effective_equilibrium_constant": effective_equilibrium_constant,
        "distorted_state": distorted_state,
        "linker_conc_for_growth": linker_conc_for_growth,
        "zr6_conc_for_growth": zr6_conc_for_growth,
        "effective_zr6_fraction_for_growth": effective_zr6_fraction_for_growth,
        "zr6_conc_adding_probability": zr6_conc_adding_probability,
        "external_addition_activity": external_addition_activity,
    }


def dissolution_probability(
    time_passed_seconds,
    dmf_decomposition_rate,
    capping_agent_conc,
    effective_linker_conc,
    effective_equilibrium_constant,
):
    dimethylamine_conc = time_passed_seconds * dmf_decomposition_rate
    if dimethylamine_conc > END_DMF_DECOMPOSITION_CONC:
        dimethylamine_conc = END_DMF_DECOMPOSITION_CONC
    if effective_linker_conc <= 0.0:
        return 1.0, float("inf")

    linker_carboxylate_to_acid_ratio = solve_linker_carboxylate_to_acid_ratio(
        dimethylamine_conc=dimethylamine_conc,
        capping_agent_conc=capping_agent_conc,
        linker_conc=effective_linker_conc,
        correction_term_for_deprotonation=CORRECTION_TERM_FOR_DEPROTONATION,
        num_carboxylate_on_linker=NUM_CARBOXYLATE_ON_LINKER,
    )
    formate_to_acid_ratio = linker_carboxylate_to_acid_ratio * CORRECTION_TERM_FOR_DEPROTONATION
    linker_carboxylic_acid_conc = effective_linker_conc * (1.0 / (1.0 + linker_carboxylate_to_acid_ratio)) * NUM_CARBOXYLATE_ON_LINKER
    if linker_carboxylic_acid_conc <= 0.0:
        return 1.0, float("inf")
    formic_acid_conc = capping_agent_conc * (1.0 / (1.0 + formate_to_acid_ratio))
    formate_benzoate_ratio = formic_acid_conc / linker_carboxylic_acid_conc / effective_equilibrium_constant
    dissolution_probability_value = formate_benzoate_ratio / (formate_benzoate_ratio + 1.0)
    return dissolution_probability_value, formate_benzoate_ratio


def run_case(job):
    seed = int(job["seed"])
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    case_inputs = build_case_inputs(job)
    entropy_correction_table = build_entropy_correction_table(job["entropy_correction_coefficient"])
    dmf_decomposition_rate = END_DMF_DECOMPOSITION_CONC / (EXP_TIME_HOURS * 3.6 * 10 ** 3)

    assembly = Assembly(
        Zr6_AA(),
        job["zr6_percentage"],
        ENTROPY_GAIN,
        job["bumping_threshold"],
        distorted_linker_fraction=case_inputs["distorted_state"]["prebound_zr_bdc_fraction"],
    )

    timing = 0.0
    formate_benzoate_ratio = 1.0
    reached_target = len(assembly.entities) >= job["target_entities"]
    time_to_target_seconds = 0.0 if reached_target else None

    start_wall = time.time()
    for step in range(int(job["max_steps"])):
        if len(assembly.entities) >= job["target_entities"]:
            reached_target = True
            time_to_target_seconds = timing * job["exchange_rxn_time_seconds"]
            break

        if step == 0 or (
            job["dissolution_update_interval_steps"] is not None
            and job["dissolution_update_interval_steps"] > 0
            and step % job["dissolution_update_interval_steps"] == 0
        ):
            _, formate_benzoate_ratio = dissolution_probability(
                timing * job["exchange_rxn_time_seconds"],
                dmf_decomposition_rate,
                job["capping_agent_conc"],
                case_inputs["linker_conc_for_growth"],
                case_inputs["effective_equilibrium_constant"],
            )

        flag, selected_carboxylate, total_growth_rate, selected_pair = assembly.next_thing_to_do(
            formate_benzoate_ratio,
            entropy_correction_table,
            case_inputs["zr6_conc_adding_probability"],
            case_inputs["external_addition_activity"],
        )
        total_growth_rate = max(float(total_growth_rate), 1e-12)
        dt = -math.log(max(random.random(), 1e-12)) / total_growth_rate

        if flag == 0:
            assembly.link_internal_carboxylate(selected_pair)
        elif flag == 1:
            assembly.grow_one_step(selected_carboxylate)
        elif flag == -1:
            assembly.remove_linkage(selected_pair)

        timing += dt

        if len(assembly.entities) >= job["target_entities"]:
            reached_target = True
            time_to_target_seconds = timing * job["exchange_rxn_time_seconds"]
            break
    else:
        step = int(job["max_steps"]) - 1

    wall_seconds = time.time() - start_wall
    return {
        "zr_conc": job["zr_conc"],
        "repeat_index": job["repeat_index"],
        "seed": seed,
        "reached_target": bool(reached_target),
        "time_to_target_seconds": time_to_target_seconds,
        "final_entities": len(assembly.entities),
        "steps_executed": step + 1,
        "simulated_time_seconds": timing * job["exchange_rxn_time_seconds"],
        "prebound_fraction": case_inputs["distorted_state"]["prebound_zr_bdc_fraction"],
        "effective_zr6_conc": case_inputs["zr6_conc_for_growth"],
        "effective_zr6_fraction_for_growth": case_inputs["effective_zr6_fraction_for_growth"],
        "effective_linker_conc": case_inputs["linker_conc_for_growth"],
        "external_addition_activity": case_inputs["external_addition_activity"],
        "distorted_chemistry_model": case_inputs["distorted_state"]["model_name"],
        "prebound_growth_attempts": assembly.prebound_growth_attempts,
        "prebound_growth_successes": assembly.prebound_growth_successes,
        "prebound_growth_failures": assembly.prebound_growth_failures,
        "prebound_entities_added": assembly.prebound_entities_added,
        "prebound_linkages_formed": assembly.prebound_linkages_formed,
        "prebound_free_growth_site_delta": assembly.prebound_free_growth_site_delta,
        "prebound_ready_pair_delta": assembly.prebound_ready_pair_delta,
        "wall_seconds": wall_seconds,
    }


def write_per_run_csv(rows, output_path):
    fieldnames = [
        "zr_conc",
        "repeat_index",
        "seed",
        "reached_target",
        "time_to_target_seconds",
        "final_entities",
        "steps_executed",
        "simulated_time_seconds",
        "prebound_fraction",
        "effective_zr6_conc",
        "effective_zr6_fraction_for_growth",
        "effective_linker_conc",
        "external_addition_activity",
        "distorted_chemistry_model",
        "prebound_growth_attempts",
        "prebound_growth_successes",
        "prebound_growth_failures",
        "prebound_entities_added",
        "prebound_linkages_formed",
        "prebound_free_growth_site_delta",
        "prebound_ready_pair_delta",
        "wall_seconds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def write_summary_csv(summary_rows, output_path):
    fieldnames = [
        "zr_conc",
        "runs",
        "reach_fraction_target",
        "mean_time_to_target_seconds",
        "std_time_to_target_seconds",
        "median_time_to_target_seconds",
        "mean_final_entities",
        "mean_simulated_time_seconds",
        "mean_steps_executed",
        "mean_prebound_fraction",
        "mean_prebound_growth_attempts",
        "mean_prebound_growth_successes",
        "mean_prebound_growth_failures",
        "mean_prebound_entities_added",
        "mean_prebound_linkages_formed",
        "mean_prebound_free_growth_site_delta",
        "mean_prebound_ready_pair_delta",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def build_summary_rows(rows, zr_values):
    summary_rows = []
    for zr_value in zr_values:
        zr_rows = [row for row in rows if row["zr_conc"] == float(zr_value)]
        if not zr_rows:
            continue
        target_times = [row["time_to_target_seconds"] for row in zr_rows if row["time_to_target_seconds"] is not None]
        summary_rows.append(
            {
                "zr_conc": float(zr_value),
                "runs": len(zr_rows),
                "reach_fraction_target": sum(1 for row in zr_rows if row["reached_target"]) / len(zr_rows),
                "mean_time_to_target_seconds": mean_or_none(target_times),
                "std_time_to_target_seconds": stdev_or_none(target_times),
                "median_time_to_target_seconds": statistics.median(target_times) if target_times else None,
                "mean_final_entities": statistics.mean(row["final_entities"] for row in zr_rows),
                "mean_simulated_time_seconds": statistics.mean(row["simulated_time_seconds"] for row in zr_rows),
                "mean_steps_executed": statistics.mean(row["steps_executed"] for row in zr_rows),
                "mean_prebound_fraction": statistics.mean(row["prebound_fraction"] for row in zr_rows),
                "mean_prebound_growth_attempts": statistics.mean(row["prebound_growth_attempts"] for row in zr_rows),
                "mean_prebound_growth_successes": statistics.mean(row["prebound_growth_successes"] for row in zr_rows),
                "mean_prebound_growth_failures": statistics.mean(row["prebound_growth_failures"] for row in zr_rows),
                "mean_prebound_entities_added": statistics.mean(row["prebound_entities_added"] for row in zr_rows),
                "mean_prebound_linkages_formed": statistics.mean(row["prebound_linkages_formed"] for row in zr_rows),
                "mean_prebound_free_growth_site_delta": statistics.mean(row["prebound_free_growth_site_delta"] for row in zr_rows),
                "mean_prebound_ready_pair_delta": statistics.mean(row["prebound_ready_pair_delta"] for row in zr_rows),
            }
        )
    return summary_rows


def render_svg(summary_rows, output_path, target_entities):
    width = 920
    height = 620
    left = 90
    top = 70
    panel_width = 340
    panel_height = 190
    col_gap = 90
    row_gap = 90
    text_color = "#222222"
    grid_color = "#d9d9d9"
    axis_color = "#333333"
    line_color = "#d95f02"
    fraction_color = "#2a9d8f"

    zr_values = [row["zr_conc"] for row in summary_rows]
    x_min = min(zr_values)
    x_max = max(zr_values)

    def map_x(value, left_coord):
        return left_coord + (value - x_min) / max(x_max - x_min, 1e-12) * panel_width

    def map_y(value, y_min, y_max, top_coord):
        return top_coord + panel_height - (value - y_min) / max(y_max - y_min, 1e-12) * panel_height

    def add_axes(lines, left_coord, top_coord, title, y_label, y_min, y_max):
        lines.append(f'<text x="{left_coord}" y="{top_coord - 16}" font-size="15" font-weight="bold" fill="{text_color}">{title}</text>')
        for tick_index in range(5):
            y_tick = y_min + (y_max - y_min) * tick_index / 4.0
            y_coord = map_y(y_tick, y_min, y_max, top_coord)
            lines.append(f'<line x1="{left_coord}" y1="{y_coord:.2f}" x2="{left_coord + panel_width}" y2="{y_coord:.2f}" stroke="{grid_color}" stroke-width="1"/>')
            lines.append(f'<text x="{left_coord - 10}" y="{y_coord + 4:.2f}" text-anchor="end" font-size="11" fill="{text_color}">{y_tick:.2f}</text>')
        for zr_value in zr_values:
            x_coord = map_x(zr_value, left_coord)
            lines.append(f'<line x1="{x_coord:.2f}" y1="{top_coord}" x2="{x_coord:.2f}" y2="{top_coord + panel_height}" stroke="{grid_color}" stroke-width="1"/>')
            lines.append(f'<text x="{x_coord:.2f}" y="{top_coord + panel_height + 18}" text-anchor="middle" font-size="11" fill="{text_color}">{zr_value:g}</text>')
        lines.append(f'<line x1="{left_coord}" y1="{top_coord + panel_height}" x2="{left_coord + panel_width}" y2="{top_coord + panel_height}" stroke="{axis_color}" stroke-width="1.3"/>')
        lines.append(f'<line x1="{left_coord}" y1="{top_coord}" x2="{left_coord}" y2="{top_coord + panel_height}" stroke="{axis_color}" stroke-width="1.3"/>')
        lines.append(f'<text x="{left_coord + panel_width / 2:.2f}" y="{top_coord + panel_height + 40}" text-anchor="middle" font-size="12" fill="{text_color}">Total Zr concentration (mM)</text>')
        lines.append(
            f'<text x="{left_coord - 52}" y="{top_coord + panel_height / 2:.2f}" transform="rotate(-90 {left_coord - 52} {top_coord + panel_height / 2:.2f})" text-anchor="middle" font-size="12" fill="{text_color}">{y_label}</text>'
        )

    def add_series(lines, left_coord, top_coord, y_min, y_max, values, color):
        points = []
        for zr_value, y_value in zip(zr_values, values):
            if y_value is None:
                continue
            x_coord = map_x(zr_value, left_coord)
            y_coord = map_y(y_value, y_min, y_max, top_coord)
            points.append(f"{x_coord:.2f},{y_coord:.2f}")
            lines.append(f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="4.5" fill="{color}"/>')
        if points:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')

    mean_times = [row["mean_time_to_target_seconds"] for row in summary_rows if row["mean_time_to_target_seconds"] is not None]
    median_times = [row["median_time_to_target_seconds"] for row in summary_rows if row["median_time_to_target_seconds"] is not None]
    max_time = max(mean_times + median_times + [1.0])
    max_fraction = 1.0
    max_final_entities = max(row["mean_final_entities"] for row in summary_rows) + 1.0
    max_prebound = max(row["mean_prebound_fraction"] for row in summary_rows) + 0.02

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-size="22" font-weight="bold" fill="{text_color}">Distorted/Prebound Time To {target_entities} Entities</text>',
        f'<text x="{left}" y="52" font-size="12" fill="{text_color}">Repeated runs per Zr, stopping when {target_entities} entities is reached or the large step cap is hit.</text>',
    ]

    add_axes(lines, left, top, f"Mean Time To {target_entities}", f"Time to {target_entities} (s)", 0.0, max_time + 20.0)
    add_series(lines, left, top, 0.0, max_time + 20.0, [row["mean_time_to_target_seconds"] for row in summary_rows], line_color)

    right_left = left + panel_width + col_gap
    add_axes(lines, right_left, top, f"Reach Fraction To {target_entities}", "Reach fraction", 0.0, max_fraction)
    add_series(lines, right_left, top, 0.0, max_fraction, [row["reach_fraction_target"] for row in summary_rows], line_color)

    bottom_top = top + panel_height + row_gap
    add_axes(lines, left, bottom_top, "Mean Final Entities", "Final entities", 0.0, max_final_entities)
    add_series(lines, left, bottom_top, 0.0, max_final_entities, [row["mean_final_entities"] for row in summary_rows], line_color)

    add_axes(lines, right_left, bottom_top, "Mean Prebound Fraction", "Prebound fraction", 0.0, max_prebound)
    add_series(lines, right_left, bottom_top, 0.0, max_prebound, [row["mean_prebound_fraction"] for row in summary_rows], fraction_color)

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    zr_values = parse_csv_floats(args.zr_values)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    output_root = output_root / f"scan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    per_run_path = output_root / "time_to_20_per_run.csv"
    summary_path = output_root / "time_to_20_summary.csv"
    svg_path = output_root / "time_to_20_summary.svg"
    json_path = output_root / "time_to_20_summary.json"

    jobs = []
    for zr_index, zr_value in enumerate(zr_values):
        for repeat_index in range(args.repeats):
            jobs.append(
                {
                    "zr_conc": float(zr_value),
                    "repeat_index": int(repeat_index),
                    "seed": 41000 + zr_index * 100 + repeat_index,
                    "target_entities": int(args.target_entities),
                    "max_steps": int(args.max_steps),
                    "zr6_percentage": float(args.zr6_percentage),
                    "entropy_correction_coefficient": float(args.entropy_correction_coefficient),
                    "equilibrium_constant_coefficient": float(args.equilibrium_constant_coefficient),
                    "h2o_dmf_ratio": float(args.h2o_dmf_ratio),
                    "capping_agent_conc": float(args.capping_agent_conc),
                    "linker_conc": float(args.linker_conc),
                    "bumping_threshold": float(args.bumping_threshold),
                    "exchange_rxn_time_seconds": float(args.exchange_rxn_time_seconds),
                    "dissolution_update_interval_steps": int(args.dissolution_update_interval_steps),
                    "distorted_chemistry_model": args.distorted_chemistry_model,
                    "distorted_ligand_association_constant": args.distorted_ligand_association_constant,
                    "distorted_site_equilibrium_constant": args.distorted_site_equilibrium_constant,
                    "distorted_second_step_equivalents": float(args.distorted_second_step_equivalents),
                    "distorted_num_sites_on_cluster": int(args.distorted_num_sites_on_cluster),
                    "distorted_num_sites_on_linker": int(args.distorted_num_sites_on_linker),
                }
            )

    config_path = output_root / "scan_config.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "zr_values": zr_values,
                "repeats": args.repeats,
                "target_entities": args.target_entities,
                "max_steps": args.max_steps,
                "distorted_chemistry_model": args.distorted_chemistry_model,
                "distorted_ligand_association_constant": args.distorted_ligand_association_constant,
                "distorted_site_equilibrium_constant": args.distorted_site_equilibrium_constant,
                "distorted_second_step_equivalents": args.distorted_second_step_equivalents,
                "distorted_num_sites_on_cluster": args.distorted_num_sites_on_cluster,
                "distorted_num_sites_on_linker": args.distorted_num_sites_on_linker,
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
            row = future.result()
            rows.append(row)
            print(
                f"finished zr={row['zr_conc']:g} rep={row['repeat_index']} "
                f"reached_20={row['reached_target']} time={row['time_to_target_seconds']}"
            )
            if len(rows) % max(1, int(args.checkpoint_every)) == 0:
                rows.sort(key=lambda item: (item["zr_conc"], item["repeat_index"]))
                write_per_run_csv(rows, per_run_path)
                summary_rows = build_summary_rows(rows, zr_values)
                write_summary_csv(summary_rows, summary_path)
                render_svg(summary_rows, svg_path, args.target_entities)
                json_path.write_text(
                    json.dumps(
                        {
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "output_root": output_root.as_posix(),
                            "per_run_csv": per_run_path.as_posix(),
                            "summary_csv": summary_path.as_posix(),
                            "summary_svg": svg_path.as_posix(),
                            "completed_jobs": len(rows),
                            "total_jobs": len(jobs),
                            "summary_rows": summary_rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    rows.sort(key=lambda item: (item["zr_conc"], item["repeat_index"]))
    write_per_run_csv(rows, per_run_path)
    summary_rows = build_summary_rows(rows, zr_values)
    write_summary_csv(summary_rows, summary_path)
    render_svg(summary_rows, svg_path, args.target_entities)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": output_root.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "summary_csv": summary_path.as_posix(),
        "summary_svg": svg_path.as_posix(),
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
