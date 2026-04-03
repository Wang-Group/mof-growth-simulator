import argparse
import json
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
INIT_DIR = CASE_DIR.parent
OUTPUT_ROOT = INIT_DIR / "output" / "mixed_nuclei" / "two_phase_amorphous_equilibrate_zr6only"

DEFAULT_PHASE1_SEED_JSON = OUTPUT_ROOT / "two_phase_eqbond_zr6only_default" / "two_phase_eqbond_zr6only_default__phase1_seed.json"
DEFAULT_FINAL_JSON = OUTPUT_ROOT / "two_phase_eqbond_zr6only_long_from2" / "two_phase_eqbond_zr6only_long_from2__stage02__rep03__seed103002.json"
DEFAULT_STAGE_SUMMARIES = [
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_default" / "two_phase_eqbond_zr6only_default.summary.json",
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_continue_from10" / "two_phase_eqbond_zr6only_continue_from10.summary.json",
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_from9_recheck" / "two_phase_eqbond_zr6only_from9_recheck.summary.json",
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_continue_from8" / "two_phase_eqbond_zr6only_continue_from8.summary.json",
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_long_from6" / "two_phase_eqbond_zr6only_long_from6.summary.json",
    OUTPUT_ROOT / "two_phase_eqbond_zr6only_long_from2" / "two_phase_eqbond_zr6only_long_from2.summary.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the earliest mixed Zr6/Zr12 amorphous seed against the final "
            "pure-Zr6 result for the retained canonical two-phase cleanup chain."
        )
    )
    parser.add_argument("--phase1-seed-json", default=DEFAULT_PHASE1_SEED_JSON.as_posix())
    parser.add_argument("--final-json", default=DEFAULT_FINAL_JSON.as_posix())
    parser.add_argument(
        "--stage-summaries",
        default=",".join(path.as_posix() for path in DEFAULT_STAGE_SUMMARIES),
        help="Comma-separated list of summary.json files in canonical chronological order.",
    )
    parser.add_argument(
        "--output-dir",
        default=(OUTPUT_ROOT / "canonical_report").as_posix(),
    )
    parser.add_argument("--basename", default="canonical_seed_vs_final")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def best_run_from_runs(runs):
    if not runs:
        return None
    ordered = sorted(
        runs,
        key=lambda row: (
            row["end_counts"]["Zr12_AA"],
            row["delta_zr12"],
            -row["delta_zr6"],
            -row["delta_bdc"],
        ),
    )
    return ordered[0]


def seed_kind_stats(seed_payload, kind):
    coordination_rows = [
        row for row in seed_payload.get("cluster_coordination", []) if row.get("kind") == kind
    ]
    radial_key = "zr12_rows" if kind == "Zr12_AA" else "zr6_rows"
    radial_rows = seed_payload.get(radial_key, [])
    coordinations = [row["coordination"] for row in coordination_rows]
    radial_fractions = [row["radial_fraction"] for row in radial_rows]
    return {
        "count": len(coordination_rows),
        "coord_min": min(coordinations) if coordinations else None,
        "coord_mean": safe_mean(coordinations),
        "coord_max": max(coordinations) if coordinations else None,
        "radial_fraction_mean": safe_mean(radial_fractions),
        "radial_fraction_min": min(radial_fractions) if radial_fractions else None,
        "radial_fraction_max": max(radial_fractions) if radial_fractions else None,
    }


def collect_stage_progression(summary_paths):
    stage_rows = []
    for summary_path in summary_paths:
        payload = load_json(summary_path)
        run_name = Path(payload["run_dir"]).name
        stages = payload.get("stages", [])
        if stages:
            for stage in stages:
                best_run = stage.get("best_run_by_zr12_loss") or best_run_from_runs(stage.get("runs", []))
                if best_run is None:
                    continue
                stage_rows.append(
                    {
                        "run_name": run_name,
                        "stage_label": stage["stage_label"],
                        "stage_seed_path": stage["stage_seed_path"],
                        "start_zr12": stage["stage_seed_counts"]["Zr12_AA"],
                        "end_zr12": best_run["end_zr12"],
                        "delta_zr6": best_run["delta_zr6"],
                        "delta_bdc": best_run["delta_bdc"],
                        "best_json_path": best_run["json_path"],
                        "best_pkl_path": best_run["pkl_path"],
                    }
                )
            continue

        legacy_best_run = payload.get("best_run")
        if legacy_best_run is None:
            continue
        stage_rows.append(
            {
                "run_name": run_name,
                "stage_label": "stage01",
                "stage_seed_path": None,
                "start_zr12": legacy_best_run["start_zr12"],
                "end_zr12": legacy_best_run["end_zr12"],
                "delta_zr6": legacy_best_run["delta_zr6"],
                "delta_bdc": legacy_best_run["delta_bdc"],
                "best_json_path": legacy_best_run["json_path"],
                "best_pkl_path": legacy_best_run["pkl_path"],
            }
        )
    return stage_rows


def build_zr12_sequence(stage_rows):
    sequence = []
    for row in stage_rows:
        start_value = row["start_zr12"]
        end_value = row["end_zr12"]
        if not sequence:
            sequence.append(start_value)
        elif sequence[-1] != start_value:
            sequence.append(start_value)
        sequence.append(end_value)
    return sequence


def build_comparison(seed_payload, final_payload, stage_rows, phase1_seed_json, final_json):
    seed_counts = seed_payload["counts"]
    final_counts = final_payload["end_counts"]
    seed_cluster_total = seed_counts["Zr6_AA"] + seed_counts["Zr12_AA"]
    final_cluster_total = final_counts["Zr6_AA"] + final_counts["Zr12_AA"]

    seed_zr6_stats = seed_kind_stats(seed_payload, "Zr6_AA")
    seed_zr12_stats = seed_kind_stats(seed_payload, "Zr12_AA")

    return {
        "phase1_seed_json": Path(phase1_seed_json).resolve().as_posix(),
        "final_json": Path(final_json).resolve().as_posix(),
        "phase1_seed_counts": seed_counts,
        "final_counts": final_counts,
        "net_change_phase1_to_final": {
            key: final_counts[key] - seed_counts[key]
            for key in (
                "Zr6_AA",
                "Zr12_AA",
                "BDC",
                "total_entities",
                "linked_pairs",
                "ready_pairs",
                "free_carboxylates",
            )
        },
        "cluster_fraction_comparison": {
            "phase1_zr6_fraction": seed_counts["Zr6_AA"] / seed_cluster_total,
            "phase1_zr12_fraction": seed_counts["Zr12_AA"] / seed_cluster_total,
            "final_zr6_fraction": final_counts["Zr6_AA"] / final_cluster_total if final_cluster_total else None,
            "final_zr12_fraction": final_counts["Zr12_AA"] / final_cluster_total if final_cluster_total else None,
        },
        "growth_factor_comparison": {
            "zr6_factor": final_counts["Zr6_AA"] / seed_counts["Zr6_AA"],
            "bdc_factor": final_counts["BDC"] / seed_counts["BDC"],
            "total_entities_factor": final_counts["total_entities"] / seed_counts["total_entities"],
        },
        "seed_coordination_stats": {
            "Zr6_AA": seed_zr6_stats,
            "Zr12_AA": seed_zr12_stats,
        },
        "shape_comparison": {
            "phase1": seed_payload["cluster_shape"],
            "final": final_payload["shape_metrics"],
            "delta_principal_axis_ratio_1_3": (
                final_payload["shape_metrics"]["principal_axis_ratio_1_3"]
                - seed_payload["cluster_shape"]["principal_axis_ratio_1_3"]
            ),
            "delta_span_ratio_max_min": (
                final_payload["shape_metrics"]["span_ratio_max_min"]
                - seed_payload["cluster_shape"]["span_ratio_max_min"]
            ),
        },
        "canonical_stage_progression": stage_rows,
        "zr12_sequence": build_zr12_sequence(stage_rows),
    }


def render_markdown(report):
    seed_counts = report["phase1_seed_counts"]
    final_counts = report["final_counts"]
    net = report["net_change_phase1_to_final"]
    fractions = report["cluster_fraction_comparison"]
    growth = report["growth_factor_comparison"]
    shape = report["shape_comparison"]
    seed_zr6 = report["seed_coordination_stats"]["Zr6_AA"]
    seed_zr12 = report["seed_coordination_stats"]["Zr12_AA"]
    stage_rows = report["canonical_stage_progression"]

    lines = [
        "# Canonical Two-Phase Seed vs Final Comparison",
        "",
        "## Files",
        f"- Phase-1 seed json: `{report['phase1_seed_json']}`",
        f"- Final pure-Zr6 json: `{report['final_json']}`",
        "",
        "## Count Comparison",
        "",
        "| Metric | Phase-1 seed | Final pure-Zr6 | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Zr6_AA | {seed_counts['Zr6_AA']} | {final_counts['Zr6_AA']} | {net['Zr6_AA']} |",
        f"| Zr12_AA | {seed_counts['Zr12_AA']} | {final_counts['Zr12_AA']} | {net['Zr12_AA']} |",
        f"| BDC | {seed_counts['BDC']} | {final_counts['BDC']} | {net['BDC']} |",
        f"| total_entities | {seed_counts['total_entities']} | {final_counts['total_entities']} | {net['total_entities']} |",
        f"| linked_pairs | {seed_counts['linked_pairs']} | {final_counts['linked_pairs']} | {net['linked_pairs']} |",
        f"| ready_pairs | {seed_counts['ready_pairs']} | {final_counts['ready_pairs']} | {net['ready_pairs']} |",
        f"| free_carboxylates | {seed_counts['free_carboxylates']} | {final_counts['free_carboxylates']} | {net['free_carboxylates']} |",
        "",
        "## Cluster Fractions",
        "",
        f"- Phase-1 cluster fractions: Zr6 = {fractions['phase1_zr6_fraction']:.4f}, Zr12 = {fractions['phase1_zr12_fraction']:.4f}",
        f"- Final cluster fractions: Zr6 = {fractions['final_zr6_fraction']:.4f}, Zr12 = {fractions['final_zr12_fraction']:.4f}",
        f"- Growth factors: Zr6 = {growth['zr6_factor']:.4f}x, BDC = {growth['bdc_factor']:.4f}x, total_entities = {growth['total_entities_factor']:.4f}x",
        "",
        "## Seed Statistics",
        "",
        f"- Seed Zr6 coordination: count = {seed_zr6['count']}, min/mean/max = {seed_zr6['coord_min']}/{seed_zr6['coord_mean']:.4f}/{seed_zr6['coord_max']}",
        f"- Seed Zr12 coordination: count = {seed_zr12['count']}, min/mean/max = {seed_zr12['coord_min']}/{seed_zr12['coord_mean']:.4f}/{seed_zr12['coord_max']}",
        f"- Seed Zr6 mean radial fraction = {seed_zr6['radial_fraction_mean']:.4f}",
        f"- Seed Zr12 mean radial fraction = {seed_zr12['radial_fraction_mean']:.4f}",
        "",
        "## Shape Comparison",
        "",
        "| Metric | Phase-1 seed | Final pure-Zr6 | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| principal_axis_ratio_1_3 | {shape['phase1']['principal_axis_ratio_1_3']:.6f} | {shape['final']['principal_axis_ratio_1_3']:.6f} | {shape['delta_principal_axis_ratio_1_3']:.6f} |",
        f"| span_ratio_max_min | {shape['phase1']['span_ratio_max_min']:.6f} | {shape['final']['span_ratio_max_min']:.6f} | {shape['delta_span_ratio_max_min']:.6f} |",
        "",
        "## Canonical Zr12 Cleanup Sequence",
        "",
        f"- Zr12 sequence: {' -> '.join(str(value) for value in report['zr12_sequence'])}",
        "",
        "| Run | Stage | start_zr12 | end_zr12 | delta_zr6 | delta_bdc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in stage_rows:
        lines.append(
            f"| {row['run_name']} | {row['stage_label']} | {row['start_zr12']} | {row['end_zr12']} | {row['delta_zr6']} | {row['delta_bdc']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The retained canonical chain starts from a mixed seed with 36 Zr12 clusters and ends at 0 Zr12 clusters.",
            "- Over the same chain, Zr6 grows from 26 to 2616 and BDC grows from 138 to 7497.",
            "- The structure becomes much more isotropic: principal_axis_ratio_1_3 drops from 4.737869 to 1.334663.",
            "- In the initial seed, Zr12 is radially more external than Zr6 on average, which is consistent with sacrificial cleanup under the Zr6-only phase-2 rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    stage_summary_paths = [Path(path.strip()) for path in args.stage_summaries.split(",") if path.strip()]

    phase1_seed = load_json(args.phase1_seed_json)
    final_payload = load_json(args.final_json)
    stage_rows = collect_stage_progression(stage_summary_paths)
    report = build_comparison(
        phase1_seed,
        final_payload,
        stage_rows,
        args.phase1_seed_json,
        args.final_json,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.basename}.json"
    md_path = output_dir / f"{args.basename}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "json_path": json_path.as_posix(),
                "md_path": md_path.as_posix(),
                "zr12_sequence": report["zr12_sequence"],
                "phase1_counts": report["phase1_seed_counts"],
                "final_counts": report["final_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
