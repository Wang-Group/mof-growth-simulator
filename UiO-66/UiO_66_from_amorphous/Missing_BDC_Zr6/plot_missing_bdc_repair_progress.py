from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis_utils import read_rows_csv


SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR / "output" / "mixeddef800_seed800_zr6only_candidate01_continuous"
DEFAULT_CSV = CASE_DIR / "repair_progress_vs_growth.csv"
DEFAULT_PNG = CASE_DIR / "repair_progress_vs_growth.png"
DEFAULT_SVG = CASE_DIR / "repair_progress_vs_growth.svg"


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
    final_row = next(row for row in rows if row["label"] == "final_endpoint")
    tracked_row = first_row_with(rows, lambda row: row["label"] == "tracked_milestone")
    first_zr_full = first_row_with(rows, lambda row: row["zr_fill"] >= 1.0)
    best_bdc = max(rows, key=lambda row: (row["bdc_fill"], row["total_entities"]))

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("#faf8f3")
    ax.set_facecolor("#fffdfa")
    ax.axhline(1.0, color="#cbd5e1", linestyle="--", linewidth=1.0, zorder=0)

    ax.plot(
        x,
        bdc_fill,
        color="#b7791f",
        linewidth=2.5,
        marker="o",
        markersize=3.8,
        label="BDC fill",
    )
    ax.plot(
        x,
        zr_fill,
        color="#2563eb",
        linewidth=2.5,
        marker="o",
        markersize=3.8,
        label="Zr fill",
    )

    ax.scatter(
        [seed_row["total_entities"]],
        [seed_row["bdc_fill"]],
        marker="s",
        s=64,
        facecolor="white",
        edgecolor="#b7791f",
        linewidth=1.7,
        zorder=5,
    )
    ax.scatter(
        [seed_row["total_entities"]],
        [seed_row["zr_fill"]],
        marker="s",
        s=64,
        facecolor="white",
        edgecolor="#2563eb",
        linewidth=1.7,
        zorder=5,
    )
    if tracked_row is not None:
        ax.scatter(
            [tracked_row["total_entities"]],
            [tracked_row["bdc_fill"]],
            marker="D",
            s=70,
            facecolor="#b7791f",
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )
        ax.scatter(
            [tracked_row["total_entities"]],
            [tracked_row["zr_fill"]],
            marker="D",
            s=70,
            facecolor="#2563eb",
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )
    ax.scatter(
        [final_row["total_entities"]],
        [final_row["bdc_fill"]],
        marker="*",
        s=180,
        facecolor="#b7791f",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    ax.scatter(
        [final_row["total_entities"]],
        [final_row["zr_fill"]],
        marker="*",
        s=180,
        facecolor="#2563eb",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )

    ax.annotate(
        (
            "Reference mixed-defect seed\n"
            f"{seed_row['total_entities']} entities"
        ),
        xy=(seed_row["total_entities"], 0.015),
        xytext=(seed_row["total_entities"] + 9000, 0.14),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#475569",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
    )
    if tracked_row is not None:
        ax.annotate(
            (
                "Tracked repaired nucleus\n"
                f"{tracked_row['total_entities']} entities\n"
                f"BDC fill = {tracked_row['bdc_fill']:.3f}"
            ),
            xy=(tracked_row["total_entities"], tracked_row["bdc_fill"]),
            xytext=(tracked_row["total_entities"] + 8000, 0.42),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
        )
    ax.annotate(
        (
            "Recovered endpoint\n"
            f"BDC fill = {final_row['bdc_fill']:.3f}\n"
            f"Zr fill = {final_row['zr_fill']:.3f}"
        ),
        xy=(final_row["total_entities"], final_row["bdc_fill"]),
        xytext=(final_row["total_entities"] - 18000, 0.965),
        ha="right",
        va="top",
        fontsize=9.3,
        color="#334155",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
    )
    if first_zr_full is not None:
        ax.annotate(
            f"First Zr fill = 1.000\nat {first_zr_full['total_entities']} entities",
            xy=(first_zr_full["total_entities"], first_zr_full["zr_fill"]),
            xytext=(first_zr_full["total_entities"] + 6500, 0.74),
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#1e40af",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
        )
    if best_bdc["label"] != "final_endpoint":
        ax.annotate(
            f"Best BDC fill before endpoint\n{best_bdc['bdc_fill']:.3f} at {best_bdc['total_entities']}",
            xy=(best_bdc["total_entities"], best_bdc["bdc_fill"]),
            xytext=(best_bdc["total_entities"] - 12000, best_bdc["bdc_fill"] - 0.14),
            ha="right",
            va="top",
            fontsize=9.0,
            color="#92400e",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 1.0},
        )

    ax.set_xlim(min(x) - 500, max(x) + 2500)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("Total entities in structure")
    ax.set_ylabel("Recovered fraction of originally missing sites")
    ax.set_title("UiO-66 missing-BDC repair under one fixed Zr6-only condition")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")

    fig.text(
        0.01,
        0.01,
        (
            "Fractions are referenced to the missing sites in UiO-66_R3_Mixeddef_0.40_seed800. "
            "The retained workspace stores a curated subset of structures; the CSV tracks the "
            "first saved checkpoint at each entity count from the original continuous run."
        ),
        ha="left",
        va="bottom",
        fontsize=8.9,
        color="#64748b",
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the recovered Missing_BDC_Zr6 repair curve from CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows_csv(args.csv)
    make_plot(rows, args.png_out, args.svg_out)
    print(args.png_out)
    print(args.svg_out)


if __name__ == "__main__":
    main()
