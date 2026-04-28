from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from analysis_utils import build_repair_rows, write_rows_csv
from plot_missing_bdc_repair_progress import make_plot as make_repair_plot
from plot_structure_triptych import load_series, make_plot as make_triptych_plot


CASE_NAME = "mixeddef800_seed800_zr6only_candidate01_continuous"

FIXED_CONDITION = {
    "ZR6_PERCENTAGE": 1.0,
    "Zr_conc": 5000.0,
    "entropy_correction_coefficient": 0.789387907185137,
    "equilibrium_constant_coefficient": 10.0,
    "H2O_DMF_RATIO": 3e-10,
    "Capping_agent_conc": 200.0,
    "Linker_conc": 100.0,
    "BUMPING_THRESHOLD": 2.0,
    "EXCHANGE_RXN_TIME_SECONDS": 0.1,
}

CURATED_RELATIVE_FILES = [
    Path("assembly_final_entity_number_100001.pkl"),
    Path("assembly_final_entity_number_100001.mol2"),
    Path("entities_number.pkl"),
    Path("entities_number_seconds.pkl"),
    Path("segment01_seed_to6500.log"),
    Path("segment02_resume_to11000.log"),
    Path("segment03_resume_to20000.log"),
    Path("segment04_resume_to40000.log"),
    Path("segment05_resume_to40000_after_entropy_fix.log"),
    Path("segment06_resume_to60000.log"),
    Path("segment07_resume_to100000.log"),
    Path("references") / "UiO-66_R3_Mixeddef_0.40_seed800.pkl",
    Path("references") / "UiO-66_R3_Mixeddef_0.40_seed800.mol2",
    Path("references") / "assembly_2026-04-03_15-42-13_entity_number_2800.pkl",
    Path("references") / "assembly_2026-04-03_15-42-13_entity_number_2800.mol2",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_source_root = script_dir.parent / "initialize_structures1"
    parser = argparse.ArgumentParser(description="Recover the curated Missing_BDC_Zr6 workspace from initialize_structures1.")
    parser.add_argument("--source-root", type=Path, default=default_source_root)
    parser.add_argument("--case-name", default=CASE_NAME)
    parser.add_argument("--tolerance", type=float, default=1.5)
    return parser.parse_args()


def copy_required_file(source_path: Path, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def first_row_with(rows, predicate):
    for row in rows:
        if predicate(row):
            return row
    return None


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    source_root = args.source_root.resolve()
    source_case_dir = source_root / "output" / args.case_name
    local_case_dir = script_dir / "output" / args.case_name
    pristine_path = script_dir / "data" / "UiO-66_15x15x15_sphere_R3.mol2"

    if not source_case_dir.exists():
        raise FileNotFoundError(f"Missing source case directory: {source_case_dir}")

    copied_files = []
    for relative_path in CURATED_RELATIVE_FILES:
        source_path = source_case_dir / relative_path
        target_path = local_case_dir / relative_path
        copy_required_file(source_path, target_path)
        copied_files.append(target_path.as_posix())

    seed_local = local_case_dir / "references" / "UiO-66_R3_Mixeddef_0.40_seed800.pkl"
    seed_mol2_local = local_case_dir / "references" / "UiO-66_R3_Mixeddef_0.40_seed800.mol2"
    milestone_mol2_local = local_case_dir / "references" / "assembly_2026-04-03_15-42-13_entity_number_2800.mol2"
    final_mol2_local = local_case_dir / "assembly_final_entity_number_100001.mol2"

    rows = build_repair_rows(
        pristine_path=pristine_path,
        seed_path=seed_local,
        checkpoint_dir=source_case_dir,
        final_path=source_case_dir / "assembly_final_entity_number_100001.pkl",
        tracked_path=source_case_dir / "assembly_2026-04-03_15-42-13_entity_number_2800.pkl",
        tolerance=args.tolerance,
    )

    csv_path = local_case_dir / "repair_progress_vs_growth.csv"
    png_path = local_case_dir / "repair_progress_vs_growth.png"
    svg_path = local_case_dir / "repair_progress_vs_growth.svg"
    write_rows_csv(rows, csv_path)
    make_repair_plot(rows, png_path, svg_path)

    triptych_png = local_case_dir / "seed_initial_final_structure_triptych.png"
    triptych_svg = local_case_dir / "seed_initial_final_structure_triptych.svg"
    triptych_series = [
        load_series("Mixed-defect seed", seed_mol2_local),
        load_series("Tracked nucleus", milestone_mol2_local),
        load_series("Final endpoint", final_mol2_local),
    ]
    make_triptych_plot(triptych_series, triptych_png, triptych_svg)

    seed_row = next(row for row in rows if row["label"] == "reference_seed")
    final_row = next(row for row in rows if row["label"] == "final_endpoint")
    milestone_row = first_row_with(rows, lambda row: row["label"] == "tracked_milestone")
    first_zr_full = first_row_with(rows, lambda row: row["zr_fill"] >= 1.0)
    best_bdc = max(rows, key=lambda row: (row["bdc_fill"], row["total_entities"]))

    summary = {
        "recovered_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_dir": script_dir.as_posix(),
        "source_root": source_root.as_posix(),
        "source_case_dir": source_case_dir.as_posix(),
        "local_case_dir": local_case_dir.as_posix(),
        "curated_result_only": True,
        "notes": [
            "The original source case contains about 3.9 GB of checkpoints and is not fully duplicated here.",
            "This workspace keeps the seed, one tracked repaired milestone, the final endpoint, logs, entity-count traces, and regenerated figures.",
        ],
        "fixed_condition": FIXED_CONDITION,
        "seed_row": seed_row,
        "milestone_row": milestone_row,
        "first_zr_full_row": first_zr_full,
        "best_bdc_row": best_bdc,
        "final_row": final_row,
        "copied_files": copied_files,
        "generated_files": [
            csv_path.as_posix(),
            png_path.as_posix(),
            svg_path.as_posix(),
            triptych_png.as_posix(),
            triptych_svg.as_posix(),
        ],
    }
    summary_path = local_case_dir / "recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
