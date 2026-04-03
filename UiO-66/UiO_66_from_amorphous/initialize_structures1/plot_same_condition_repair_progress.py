import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

from run_defect_growth_matrix import (
    assembly_counts,
    centers_for_type,
    nearest_distances,
    quiet_build_assembly_from_mol2,
    quiet_pickle_load,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRISTINE = SCRIPT_DIR / "data" / "UiO-66_15x15x15_sphere_R3.mol2"
DEFAULT_CASE_DIR = SCRIPT_DIR / "output" / "mixeddef800_zx2_seed800_continuous_from2124"
DEFAULT_SEED = DEFAULT_CASE_DIR / "references" / "UiO-66_R3_Mixeddef_0.40_seed800.pkl"
DEFAULT_INITIAL = DEFAULT_CASE_DIR / "references" / "assembly_2026-04-03_10-39-56_entity_number_2960.pkl"
DEFAULT_FINAL = DEFAULT_CASE_DIR / "assembly_final_entity_number_3651.pkl"
DEFAULT_CSV = DEFAULT_CASE_DIR / "repair_progress_vs_growth.csv"
DEFAULT_PNG = DEFAULT_CASE_DIR / "repair_progress_vs_growth.png"
DEFAULT_SVG = DEFAULT_CASE_DIR / "repair_progress_vs_growth.svg"


def entity_count_from_path(path: Path) -> int | None:
    match = re.search(r"entity_number_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def first_snapshots_by_entity_count(case_dir: Path) -> list[Path]:
    seen_counts = set()
    selected = []
    for path in sorted(case_dir.glob("assembly_*.pkl"), key=lambda file: (file.stat().st_mtime, file.name)):
        entity_count = entity_count_from_path(path)
        if entity_count is None or entity_count in seen_counts:
            continue
        seen_counts.add(entity_count)
        selected.append(path)
    return selected


def missing_sites(pristine, seed, tolerance: float):
    pristine_bdc = centers_for_type(pristine, "Ligand")
    pristine_zr = centers_for_type(pristine, "Zr")
    seed_bdc = centers_for_type(seed, "Ligand")
    seed_zr = centers_for_type(seed, "Zr")

    missing_bdc = pristine_bdc[nearest_distances(pristine_bdc, seed_bdc) > tolerance]
    missing_zr = pristine_zr[nearest_distances(pristine_zr, seed_zr) > tolerance]
    return missing_bdc, missing_zr


def repair_metrics(assembly, missing_bdc, missing_zr, tolerance: float):
    assembly_bdc = centers_for_type(assembly, "Ligand")
    assembly_zr = centers_for_type(assembly, "Zr")
    total_entities, total_bdc, total_zr = assembly_counts(assembly)

    filled_bdc = int((nearest_distances(missing_bdc, assembly_bdc) <= tolerance).sum())
    filled_zr = int((nearest_distances(missing_zr, assembly_zr) <= tolerance).sum())

    return {
        "total_entities": total_entities,
        "total_bdc": total_bdc,
        "total_zr": total_zr,
        "missing_bdc_sites": int(len(missing_bdc)),
        "missing_zr_sites": int(len(missing_zr)),
        "filled_bdc_sites": filled_bdc,
        "filled_zr_sites": filled_zr,
        "bdc_fill": filled_bdc / len(missing_bdc) if len(missing_bdc) else 0.0,
        "zr_fill": filled_zr / len(missing_zr) if len(missing_zr) else 0.0,
    }


def row_for_snapshot(label: str, assembly_path: Path, missing_bdc, missing_zr, tolerance: float):
    assembly = quiet_pickle_load(assembly_path)
    metrics = repair_metrics(assembly, missing_bdc, missing_zr, tolerance)
    metrics["label"] = label
    metrics["file_name"] = assembly_path.name
    return metrics


def build_rows(
    pristine_path: Path,
    seed_path: Path,
    initial_path: Path,
    final_path: Path,
    case_dir: Path,
    tolerance: float,
):
    pristine = quiet_build_assembly_from_mol2(pristine_path)
    seed = quiet_pickle_load(seed_path)
    missing_bdc, missing_zr = missing_sites(pristine, seed, tolerance)

    rows = []
    rows.append(row_for_snapshot("reference_seed", seed_path, missing_bdc, missing_zr, tolerance))
    initial_row = row_for_snapshot("initial_nucleus", initial_path, missing_bdc, missing_zr, tolerance)
    initial_entities = initial_row["total_entities"]
    rows.append(initial_row)

    initial_resolved = initial_path.resolve()
    seed_entities = rows[0]["total_entities"]
    final_entities = entity_count_from_path(final_path)
    for path in first_snapshots_by_entity_count(case_dir):
        if path.resolve() == initial_resolved:
            continue
        entity_count = entity_count_from_path(path)
        if entity_count is None:
            continue
        if entity_count <= seed_entities or entity_count >= final_entities:
            continue
        rows.append(row_for_snapshot("same_condition_continuation", path, missing_bdc, missing_zr, tolerance))

    rows.append(row_for_snapshot("final_endpoint", final_path, missing_bdc, missing_zr, tolerance))
    rows.sort(key=lambda row: row["total_entities"])
    return rows


def write_csv(rows, csv_path: Path):
    fieldnames = [
        "label",
        "file_name",
        "total_entities",
        "total_bdc",
        "total_zr",
        "missing_bdc_sites",
        "missing_zr_sites",
        "filled_bdc_sites",
        "filled_zr_sites",
        "bdc_fill",
        "zr_fill",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_row_with(rows, predicate):
    for row in rows:
        if predicate(row):
            return row
    return None


def make_plot(rows, png_path: Path, svg_path: Path):
    x = [row["total_entities"] for row in rows]
    bdc_fill = [row["bdc_fill"] for row in rows]
    zr_fill = [row["zr_fill"] for row in rows]

    seed_row = next(row for row in rows if row["label"] == "reference_seed")
    initial_row = next(row for row in rows if row["label"] == "initial_nucleus")
    final_row = next(row for row in rows if row["label"] == "final_endpoint")
    first_zr_full = first_row_with(rows, lambda row: row["zr_fill"] >= 1.0)
    best_bdc = max(rows, key=lambda row: (row["bdc_fill"], row["total_entities"]))

    fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    ax.set_facecolor("#fcfbf8")
    ax.axvspan(
        initial_row["total_entities"],
        final_row["total_entities"],
        color="#eef6ff",
        alpha=0.95,
        zorder=0,
    )
    ax.axvline(2769, color="#9ca3af", linestyle="--", linewidth=1.2, label="pristine size (2769)")
    ax.axvline(initial_row["total_entities"], color="#64748b", linestyle=":", linewidth=1.4, label="initial nucleus (2960)")
    ax.axhline(1.0, color="#cbd5e1", linestyle="--", linewidth=1.0)

    ax.plot(x, bdc_fill, color="#b7791f", linewidth=2.6, marker="o", markersize=4.2, label="BDC fill")
    ax.plot(x, zr_fill, color="#2563eb", linewidth=2.6, marker="o", markersize=4.2, label="Zr fill")

    ax.scatter(
        [seed_row["total_entities"]],
        [seed_row["bdc_fill"]],
        marker="s",
        s=70,
        facecolor="white",
        edgecolor="#b7791f",
        linewidth=1.8,
        zorder=5,
    )
    ax.scatter(
        [seed_row["total_entities"]],
        [seed_row["zr_fill"]],
        marker="s",
        s=70,
        facecolor="white",
        edgecolor="#2563eb",
        linewidth=1.8,
        zorder=5,
    )
    ax.scatter(
        [initial_row["total_entities"]],
        [initial_row["bdc_fill"]],
        marker="D",
        s=68,
        facecolor="#b7791f",
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    ax.scatter(
        [initial_row["total_entities"]],
        [initial_row["zr_fill"]],
        marker="D",
        s=68,
        facecolor="#2563eb",
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    ax.scatter(
        [final_row["total_entities"]],
        [final_row["bdc_fill"]],
        marker="*",
        s=170,
        facecolor="#b7791f",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    ax.scatter(
        [final_row["total_entities"]],
        [final_row["zr_fill"]],
        marker="*",
        s=170,
        facecolor="#2563eb",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )

    ax.annotate(
        "Reference mixed-defect seed\n(artificial defect seed)",
        xy=(seed_row["total_entities"], 0.02),
        xytext=(seed_row["total_entities"] - 30, 0.17),
        ha="right",
        va="bottom",
        fontsize=9.2,
        color="#475569",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
    )
    ax.annotate(
        (
            "Initial defect-containing nucleus\n"
            f"BDC fill = {initial_row['bdc_fill']:.3f}\n"
            f"Zr fill = {initial_row['zr_fill']:.3f}"
        ),
        xy=(initial_row["total_entities"], max(initial_row["bdc_fill"], initial_row["zr_fill"])),
        xytext=(initial_row["total_entities"] + 18, 0.73),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#334155",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
    )
    ax.annotate(
        f"Final endpoint\nBDC fill = {final_row['bdc_fill']:.3f}\nZr fill = {final_row['zr_fill']:.3f}",
        xy=(final_row["total_entities"], final_row["bdc_fill"]),
        xytext=(final_row["total_entities"] - 120, 0.98),
        ha="right",
        va="top",
        fontsize=9.4,
        color="#334155",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
    )
    if first_zr_full is not None:
        ax.annotate(
            f"First Zr fill = 1.000\nat {first_zr_full['total_entities']} entities",
            xy=(first_zr_full["total_entities"], first_zr_full["zr_fill"]),
            xytext=(first_zr_full["total_entities"] - 90, 0.62),
            ha="right",
            va="bottom",
            fontsize=9.0,
            color="#1e40af",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
        )
    if best_bdc["label"] != "final_endpoint":
        ax.annotate(
            f"Best BDC fill so far\n{best_bdc['bdc_fill']:.3f} at {best_bdc['total_entities']}",
            xy=(best_bdc["total_entities"], best_bdc["bdc_fill"]),
            xytext=(best_bdc["total_entities"] - 80, best_bdc["bdc_fill"] - 0.11),
            ha="right",
            va="top",
            fontsize=9.0,
            color="#92400e",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
        )

    ax.set_xlim(min(x) - 40, max(x) + 40)
    ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel("Total entities in structure")
    ax.set_ylabel("Recovery of original missing sites")
    ax.set_title("UiO-66 same-condition mixed-defect repair trajectory")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    ax.legend(uniq_handles, uniq_labels, frameon=False, loc="lower right")

    fig.text(
        0.01,
        0.01,
        (
            "Repair fractions are referenced to the original missing sites in "
            "UiO-66_R3_Mixeddef_0.40_seed800. Continuation points summarize the first "
            "saved snapshot reaching each entity count."
        ),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#64748b",
    )

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot repair progression for the retained same-condition UiO-66 case.")
    parser.add_argument("--pristine", type=Path, default=DEFAULT_PRISTINE)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--tolerance", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(
        pristine_path=args.pristine,
        seed_path=args.seed,
        initial_path=args.initial,
        final_path=args.final,
        case_dir=args.case_dir,
        tolerance=args.tolerance,
    )
    write_csv(rows, args.csv_out)
    make_plot(rows, args.png_out, args.svg_out)
    print(args.csv_out)
    print(args.png_out)
    print(args.svg_out)


if __name__ == "__main__":
    main()
