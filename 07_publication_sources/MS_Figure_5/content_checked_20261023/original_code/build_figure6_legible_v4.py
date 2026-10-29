"""Build a legible, provenance-preserving revision of manuscript Figure 6.

Panels A and D retain the author-prepared trajectory and molecular-structure
graphics. Panels B and C are redrawn from the current BTB response-surface
table. Panel A's displayed clock is shifted from 620--685 ms to elapsed time
0--65 ms; the trajectory itself is unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent / "figure_rebuild_20260901"
CAMPAIGN = HERE.parent
ARCHIVED_FIGURE = REBUILD / "figure6_archived_base.png"
ANALYSIS_SCRIPT = REBUILD / "analyze_figure6_current_parameter_phase_v1.py"
STYLE_KERNEL = Path(r"C:\Users\Cheng Wang\.codex\skills\figure-style\kernel.py")
SOURCE_TABLE = CAMPAIGN / "common_response_surface_v2_final_condition_summary.csv"
OUTPUT_PREFIX = HERE / "figure6_legible_v4"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STYLE = load_module(STYLE_KERNEL, "figure_style_kernel")
ANALYSIS = load_module(ANALYSIS_SCRIPT, "figure6_current_analysis")


BLUE = "#5185C0"
ORANGE = "#E99D4E"
TEXT = "#202124"


def prepare_data():
    frame = pd.read_csv(SOURCE_TABLE)
    btb = frame[frame.system == "btb_mol"].copy()
    btb[["SC", "q", "off_weight"]] = btb[["SC", "q", "off_weight"]].astype(float)
    sc, q, rho, cube, n, _ = ANALYSIS.build_probability_cube(btb)
    first, total = ANALYSIS.variance_indices(cube)
    first_low, first_high, total_low, total_high = ANALYSIS.bootstrap_indices(cube, n)
    return btb, sc, q, rho, cube, first, total, first_low, first_high, total_low, total_high


def panel_a(ax, archived: Image.Image) -> None:
    # Native crop verified against the 5835 x 6514 archived source. This crop
    # includes the complete right-hand continuation that the previous x=2780
    # crop removed, but excludes the old title and absolute-time tick labels.
    image = archived.crop((180, 250, 2920, 1712))
    ax.imshow(image, extent=(0, 1, 0.17, 1), aspect="auto", interpolation="lanczos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Representative stochastic trajectory", loc="left", pad=6, color=TEXT)

    # Original tick positions correspond to 620, 630, ..., 680 ms. Relabeling
    # as 0, 10, ..., 60 ms changes only the time origin.
    tick_x = np.array([420, 780, 1135, 1495, 1855, 2210, 2570], dtype=float)
    tick_x = (tick_x - 180.0) / (2920.0 - 180.0)
    for x, label in zip(tick_x, ["0", "10", "20", "30", "40", "50", "60"]):
        ax.plot([x, x], [0.155, 0.17], color="black", lw=0.65, clip_on=False)
        ax.text(x, 0.137, label, ha="center", va="top", fontsize=7.5, color=TEXT)
    ax.text(
        0.5,
        0.025,
        "Elapsed growth time (ms)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=TEXT,
    )


def panel_b(
    ax,
    first,
    total,
    first_low,
    first_high,
    total_low,
    total_high,
) -> None:
    ANALYSIS.sensitivity_panel(
        ax,
        first,
        total,
        first_low,
        first_high,
        total_low,
        total_high,
    )
    ax.set_title("Sensitivity of persistent-assembly probability", loc="left", pad=6)
    ax.set_xlabel("Fraction of variance in persistent-assembly probability", labelpad=5)
    ax.set_yticks(
        np.arange(3),
        [
            r"$\rho_{\mathrm{off}}$",
            "M/BTB",
            r"$s_{\mathrm{close}}$",
        ],
    )
    ax.tick_params(axis="both", labelsize=7.5)
    ax.xaxis.label.set_size(8.0)
    ax.yaxis.label.set_size(8.0)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="lower right",
        handlelength=1.7,
        labelspacing=0.35,
    )


def panel_c(ax, sc, q, rho, cube) -> dict[str, int]:
    counts = ANALYSIS.phase_panel(ax, sc, q, rho, cube)
    # Keep all three axis titles inside the allocated manuscript panel. The
    # zoom changes only the camera framing, not the surface or data positions.
    ax.set_box_aspect((1.0, 0.95, 0.72), zoom=0.76)
    # phase_panel writes a centred title. Remove every title slot before adding
    # the consistent left-aligned panel title used throughout the figure.
    ax.title.set_text("")
    ax._left_title.set_text("")
    ax._right_title.set_text("")
    ax.set_title("Calculated kinetic formation boundary", loc="left", pad=8)
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$", labelpad=5)
    ax.set_ylabel("M/BTB", labelpad=6)
    ax.set_zlabel(r"$s_{\mathrm{close}}$", labelpad=5)
    # Omit the coincident q=0.01 endpoint label: in this camera view it sits on
    # top of the rho_off=1500 label. The axis still extends to q=0.01.
    q_ticks = np.array([0.1, 1.0, 10.0, 100.0])
    ax.set_yticks(np.log10(q_ticks), ["0.1", "1", "10", "100"])
    ax.tick_params(axis="x", labelsize=7.2, pad=1)
    ax.tick_params(axis="y", labelsize=7.2, pad=1)
    ax.tick_params(axis="z", labelsize=7.2, pad=1)
    ax.xaxis.label.set_size(7.8)
    ax.yaxis.label.set_size(7.8)
    ax.zaxis.label.set_size(7.8)
    legend = ax.get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        replacements = {
            "Interpolated P = 0.5 surface": "Interpolated P = 0.5 boundary",
            "Calculated transition-region conditions": r"Calculated: $0.1 \leq P \leq 0.9$",
            "Calculated: P <= 0.1": r"Calculated: $P \leq 0.1$",
            "Calculated: P >= 0.9": r"Calculated: $P \geq 0.9$",
        }
        labels = [replacements.get(label, label) for label in labels]
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=7.2,
            loc="upper left",
            bbox_to_anchor=(-0.04, 0.94),
            handletextpad=0.45,
            labelspacing=0.28,
            borderaxespad=0,
        )
    return counts


def panel_d(ax, archived: Image.Image) -> None:
    # Exclude the old giant panel letter/title while preserving all structures,
    # frequencies, outlines, and the Entity 1--13 annotation.
    image = archived.crop((2920, 240, archived.width, archived.height))
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")
    ax.set_anchor("N")
    ax.set_title("Representative structures during growth", loc="left", pad=6, color=TEXT)


def add_panel_letters(fig, mapping) -> None:
    for letter, ax in mapping:
        box = ax.get_position()
        fig.text(
            box.x0 - 0.018,
            box.y1 + 0.006,
            letter,
            ha="right",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=TEXT,
        )


def save_individual_panels(archived, data) -> None:
    _, sc, q, rho, cube, first, total, first_low, first_high, total_low, total_high = data

    specifications = [
        ("A", (6.6, 3.8), None),
        ("B", (6.6, 3.7), None),
        ("C", (6.6, 5.5), "3d"),
        ("D", (5.0, 10.0), None),
    ]
    for letter, size, projection in specifications:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection=projection) if projection else fig.add_subplot(111)
        if letter == "A":
            panel_a(ax, archived)
        elif letter == "B":
            panel_b(ax, first, total, first_low, first_high, total_low, total_high)
        elif letter == "C":
            panel_c(ax, sc, q, rho, cube)
        else:
            panel_d(ax, archived)
        fig.subplots_adjust(left=0.15 if letter == "B" else 0.08, right=0.98, top=0.91, bottom=0.14)
        add_panel_letters(fig, [(letter, ax)])
        fig.savefig(
            HERE / f"figure6_panel_{letter}_v4.png",
            dpi=450,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(
            HERE / f"figure6_panel_{letter}_v4.svg",
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)


def main() -> None:
    STYLE.apply_figure_style(frame="open", font="Arial", sizes=(9.2, 8.0, 7.5), grid=False)
    archived = Image.open(ARCHIVED_FIGURE).convert("RGB")
    data = prepare_data()
    btb, sc, q, rho, cube, first, total, first_low, first_high, total_low, total_high = data

    # 190 mm wide. The right column is tall enough to retain the detailed
    # structure matrix; the left column allocates a separate row to each plot.
    fig = plt.figure(figsize=(7.48, 10.65), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.02, 1.08],
        height_ratios=[0.74, 0.88, 1.16],
        left=0.085,
        right=0.985,
        top=0.965,
        bottom=0.145,
        wspace=0.22,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[2, 0], projection="3d")
    ax_d = fig.add_subplot(grid[:, 1])

    panel_a(ax_a, archived)
    panel_b(ax_b, first, total, first_low, first_high, total_low, total_high)
    counts = panel_c(ax_c, sc, q, rho, cube)
    panel_d(ax_d, archived)
    # The 3D axes project labels outside their nominal rectangle. Slightly
    # reduce Panel C so all labels remain within the final canvas.
    cbox = ax_c.get_position()
    ax_c.set_position([cbox.x0 + 0.006, cbox.y0 + 0.020, cbox.width - 0.010, cbox.height - 0.040])

    # Pack the three left-column panels with compact, equal visible gutters.
    # Their dimensions are unchanged; only their vertical positions move.
    ab_gutter = 0.048
    bc_gutter = 0.065
    abox = ax_a.get_position()
    bbox = ax_b.get_position()
    ax_b.set_position([bbox.x0, abox.y0 - ab_gutter - bbox.height, bbox.width, bbox.height])
    bbox = ax_b.get_position()
    cbox = ax_c.get_position()
    ax_c.set_position([cbox.x0, bbox.y0 - bc_gutter - cbox.height, cbox.width, cbox.height])

    # Scale Panel D uniformly until its visible top and bottom align with the
    # packed left column. The crop's native width/height ratio is preserved.
    dbox = ax_d.get_position()
    d_top = abox.y1
    d_bottom = ax_c.get_position().y0
    d_height = d_top - d_bottom
    d_native_ratio = (archived.width - 2920) / (archived.height - 240)
    d_width = d_height * fig.get_figheight() / fig.get_figwidth() * d_native_ratio
    ax_d.set_position([dbox.x1 - d_width, d_bottom, d_width, d_height])
    add_panel_letters(fig, [("A", ax_a), ("B", ax_b), ("C", ax_c), ("D", ax_d)])

    fig.savefig(
        OUTPUT_PREFIX.with_suffix(".png"),
        dpi=450,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        OUTPUT_PREFIX.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    save_individual_panels(archived, data)

    manifest = {
        "figure": str(OUTPUT_PREFIX.with_suffix(".png")),
        "archived_source": str(ARCHIVED_FIGURE),
        "current_parameter_source": str(SOURCE_TABLE),
        "panel_a_time_transform": "t_elapsed_ms = t_archived_ms - 620",
        "panel_a_displayed_range_ms": [0, 65],
        "grid_shape": [int(len(sc)), int(len(q)), int(len(rho))],
        "replicates_per_condition": sorted(btb.trace_count.astype(int).unique().tolist()),
        "panel_c_interpretation": "kinetic outcome boundary, not thermodynamic phase boundary",
        "panel_c_counts": counts,
        "panel_ab_gutter_figure_fraction": ab_gutter,
        "panel_bc_gutter_figure_fraction": bc_gutter,
        "panel_d_native_width_height_ratio": d_native_ratio,
        "panel_d_aspect_preserved": True,
        "panel_d_matches_left_column_vertical_bounds": True,
        "panel_b_c_axis_labels": ["rho_off", "M/BTB", "s_close"],
        "presentation_only_revision": True,
    }
    (HERE / "figure6_legible_v4_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    caption = (
        "Figure 6 | Molecular BTB-MOL trajectories and the current kinetic response surface. "
        "(A) Representative stochastic trajectory; the displayed interval is reported as elapsed "
        "time, t_elapsed = t_archived - 620 ms. The trajectory and molecular insets are unchanged. "
        "(B) Direct first-order and total-effect variance fractions for the link-removal/addition "
        "ratio, atomic metal/BTB ratio, and ring-closure coefficient over the sampled full-factorial "
        "grid. (C) Interpolated surface at persistent-assembly probability P = 0.5; points show "
        "calculated suppressed, transition, and reliable conditions. (D) Representative "
        "loop-containing structures from the author-prepared trajectory analysis. The surface in "
        "C is a kinetic outcome boundary and should not be interpreted as an equilibrium phase diagram."
    )
    (HERE / "figure6_legible_v4_caption.md").write_text(caption + "\n", encoding="utf-8")
    print(OUTPUT_PREFIX.with_suffix(".png"))


if __name__ == "__main__":
    main()
