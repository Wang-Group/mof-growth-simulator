#!/usr/bin/env python3
"""Build the canonical calculated surface-registry diagnostic for the SI."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PILOT = HERE / "uio66_surface_registry_pilot.py"
spec = importlib.util.spec_from_file_location("surface_pilot", PILOT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {PILOT}")
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)


def load_rows():
    conditions = []
    for summary_path in sorted(HERE.glob("pilot_*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        csv_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".csv"))
        rows = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        label = (
            f"R={summary['R_sim']:g}\n"
            f"p={summary['p_Zr']:g}, E={summary['E_sim']:g}"
        )
        conditions.append((label, summary, rows))
    return conditions


def main():
    parent = surface.load_assembly(HERE / "seed" / "UiO66_n4_111_octahedron.pkl")
    outer = surface.load_assembly(HERE / "seed" / "UiO66_n6_111_octahedron.pkl")
    registry, _ = surface.build_registry(parent, outer, 0.2)
    parent_points = surface.centers_by_type(parent)
    conditions = load_rows()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1], height_ratios=[1, 1])

    ax_a = fig.add_subplot(grid[:, 0], projection="3d")
    colors = {"Zr": "#31A6A0", "Ligand": "#A8B0B8"}
    labels = {"Zr": "parent Zr6", "Ligand": "parent BDC"}
    for entity_type, points in parent_points.items():
        ax_a.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=26 if entity_type == "Zr" else 7,
            c=colors[entity_type], alpha=0.90, edgecolors="none",
            label=labels[entity_type],
        )
    for entity_type, points in registry.items():
        ax_a.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=12 if entity_type == "Zr" else 3.5,
            facecolors="none", edgecolors="#D55E00", alpha=0.18,
            linewidths=0.35,
        )
    ax_a.view_init(elev=21, azim=36)
    ax_a.set_box_aspect((1, 1, 1))
    ax_a.dist = 6.6
    ax_a.set_axis_off()
    ax_a.set_title("a  Parent and next-shell registry", loc="left", fontweight="bold", fontsize=8.8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#31A6A0", markeredgecolor="none", label="parent Zr6", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#A8B0B8", markeredgecolor="none", label="parent BDC", markersize=4),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#D55E00", alpha=0.55, label="unused ideal site", markersize=5),
    ]
    ax_a.legend(handles=handles, loc="lower center", frameon=False, fontsize=7.5)

    labels_x = [
        f"{summary['R_sim']:g}\n{summary['p_Zr']:g}\n{summary['E_sim']:g}"
        for _, summary, _ in conditions
    ]
    registered = [sum(int(row["is_parent_registry"]) for row in rows) for _, _, rows in conditions]
    off = [len(rows) - count for (_, _, rows), count in zip(conditions, registered)]
    x = np.arange(len(conditions))
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_b.bar(x, registered, color="#3B82B8", label="registry matched")
    ax_b.bar(x, off, bottom=registered, color="#D55E00", label="off registry")
    ax_b.set_yscale("symlog", linthresh=10)
    ax_b.set_ylabel("Accepted additions")
    ax_b.set_xticks(x, labels_x, fontsize=7.2)
    ax_b.set_xlabel("$R_{sim}$ / $p_{Zr}$ / $E_{sim}$")
    ax_b.set_title("b  Pilot additions", loc="left", fontweight="bold", fontsize=8.8)
    ax_b.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.4)
    ax_b.spines[["top", "right"]].set_visible(False)

    ax_c = fig.add_subplot(grid[1, 1])
    distances = [
        float(row["registry_distance_A"])
        for _, _, rows in conditions for row in rows
        if np.isfinite(float(row["registry_distance_A"]))
    ]
    bins = np.linspace(0, max(8.0, max(distances, default=8.0)), 33)
    ax_c.hist(distances, bins=bins, color="#6B7280", edgecolor="white", linewidth=0.25)
    ax_c.axvline(0.5, color="#D55E00", linestyle="--", linewidth=1.1, label="0.5 A cutoff")
    ax_c.set_yscale("symlog", linthresh=1)
    ax_c.set_xlabel("Nearest ideal-site distance (A)")
    ax_c.set_ylabel("Accepted additions")
    ax_c.set_title("c  Registry-distance diagnostic", loc="left", fontweight="bold", fontsize=8.8)
    ax_c.legend(frameon=False, fontsize=7.5)
    ax_c.spines[["top", "right"]].set_visible(False)

    ax_d = fig.add_subplot(grid[1, 2])
    max_size = [summary["maximum_post_seed_component"]["entities"] for _, summary, _ in conditions]
    cycle = [summary["maximum_post_seed_component"]["cycle_rank"] for _, summary, _ in conditions]
    width = 0.38
    ax_d.bar(x - width / 2, max_size, width, color="#31A6A0", label="entities")
    ax_d.bar(x + width / 2, cycle, width, color="#CC79A7", label="cycle rank")
    ax_d.set_xticks(x, [str(i + 1) for i in x])
    ax_d.set_xlabel("Pilot trajectory")
    ax_d.set_ylabel("Maximum observed value")
    ax_d.set_title("d  Largest new component", loc="left", fontweight="bold", fontsize=8.8)
    ax_d.legend(frameon=False, fontsize=7.5)
    ax_d.spines[["top", "right"]].set_visible(False)

    out_png = HERE / "figure_S72_surface_registry_diagnostic.png"
    out_svg = HERE / "figure_S72_surface_registry_diagnostic.svg"
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    plt.close(fig)
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
