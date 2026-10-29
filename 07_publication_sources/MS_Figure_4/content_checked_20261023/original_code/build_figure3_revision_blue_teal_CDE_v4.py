"""Rebuild manuscript Figure 3 with legible chemistry and N=10000 PXRD.

This script retains the current KMC definitions and numerical sources while
repairing layout and typography. It imports the audited data-reduction
functions from plot_figure3_prototype_v1.py, then redraws every panel at the
intended 190 mm manuscript width.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE3_ROOT = HERE.parent
CAMPAIGN = FIGURE3_ROOT.parent
BASE_SCRIPT = FIGURE3_ROOT / "plot_figure3_prototype_v1.py"
STYLE_KERNEL = CAMPAIGN.parents[2] / "external_helpers" / "figure-style" / "kernel.py"
OUTPUT_PREFIX = HERE / "figure3_revision_blue_teal_CDE_v3"
PANEL_DIR = HERE / "panels"
ORIGINAL_V11_DIR = FIGURE3_ROOT / "figure3_revision_20260904_v11"
RESPONSE_SOURCE = ORIGINAL_V11_DIR / "figure3_response_surface_source_v11.csv"
SENSITIVITY_SOURCE = ORIGINAL_V11_DIR / "figure3_local_sensitivity_source_v11.csv"

N10000_DIR = FIGURE3_ROOT / "n10000_growth_campaign_v1"
FINAL_PXRD_DIR = N10000_DIR / "final_pxrd_20260904"
PXRD_SOURCES = {
    0.1: FINAL_PXRD_DIR / "rho0p1_N10000_finite_debye_pxrd.csv",
    0.3: FINAL_PXRD_DIR / "rho0p3_N10000_finite_debye_pxrd.csv",
    1.0: FINAL_PXRD_DIR / "rho1_N10000_finite_debye_pxrd.csv",
    3.0: FINAL_PXRD_DIR / "rho3_N10000_finite_debye_pxrd.csv",
    10.0: FINAL_PXRD_DIR / "rho10_N10000_finite_debye_pxrd.csv",
}
IDEAL_PEAKS = FINAL_PXRD_DIR / "ideal_uio66_periodic_peak_positions.csv"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SCRIPT, "figure3_base")
STYLE = load_module(STYLE_KERNEL, "figure_style_kernel")

TEXT = "#20262B"
MUTED = "#66717A"
MISSING = "#D9DDE1"

# More contrastive, still-soft blue palette for probability Panels C-D.
# The upper endpoint is medium blue rather than navy so high-value regions
# remain prominent without becoming visually heavy.
RESPONSE_BLUE = LinearSegmentedColormap.from_list(
    "response_blue_mid",
    ["#EEF6FB", "#C9E0F0", "#91C1DC", "#5B9FC4", "#2F78A5"],
)

# Intermediate blue-teal palette for Panel D only. Panel C retains the blue
# band above and Panel E retains RESPONSE_TEAL below; this band bridges the
# two fixed palettes without changing data or normalization.
RESPONSE_BLUE_TEAL = LinearSegmentedColormap.from_list(
    "response_blue_teal_intermediate",
    ["#EEF8FA", "#C6E5EC", "#8FC8D3", "#58A9B8", "#2D788D"],
)

# Distinct but harmonized teal palette for the growth-rate Panel E. The hue
# change separates a rate from the two probability panels while preserving the
# same light-to-moderate contrast structure.
RESPONSE_TEAL = LinearSegmentedColormap.from_list(
    "response_teal_mid",
    ["#EEF9F7", "#C7E9E3", "#8BD1C6", "#4DAFA4", "#237F78"],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data():
    # Reuse the exact audited numerical tables exported by the canonical v11
    # run. Some upstream scratch-analysis inputs are no longer in this handoff.
    response = pd.read_csv(RESPONSE_SOURCE)
    sensitivity = pd.read_csv(SENSITIVITY_SOURCE)
    finite_rates = response.G_76_to_126_entities_per_model_time.replace([np.inf, -np.inf], np.nan)
    rate_max = float(np.nanpercentile(finite_rates, 98))
    return response, sensitivity, rate_max


def definition_panel(ax) -> None:
    """Full-height molecular definition panel with separated Zr6 and BDC glyphs."""
    # The slightly negative left bound shifts the molecular content to the
    # right without moving the composite-level panel letter.
    ax.set_xlim(-0.025, 1.0)
    ax.set_ylim(0, 1)
    # Preserve the native geometry of every molecular and framework glyph.
    # Equal x/y display scales prevent circular nodes and hexagonal linkers
    # from being flattened when Panel A occupies a rectangular grid cell.
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("NE")
    ax.axis("off")

    # Use the same visible vertical span as Panel B: the first heading sits at
    # the top edge, while the structure labels extend close to the bottom edge.
    ax.text(0.02, 1.035, "Nucleation pathway", fontweight="bold", fontsize=8.8, va="top", clip_on=False)

    # Separate reactants explicitly. The prior rendering made the node and
    # linker appear connected before the P_2->30 process began.
    BASE.draw_zr6_node(ax, (0.065, 0.805), scale=1.35)
    ax.text(0.145, 0.805, "+", fontsize=11, ha="center", va="center", color=TEXT)
    BASE.draw_bdc_linker(ax, (0.21, 0.805), (0.35, 0.805), scale=1.35)
    ax.text(0.050, 0.705, r"Zr$_6$ node", ha="center", fontsize=7.8)
    ax.text(0.300, 0.705, "BDC linker", ha="center", fontsize=7.8)
    ax.text(0.17, 0.65, r"nascent pair, $N=2$", ha="center", fontsize=8.0, color="#56616A")

    ax.add_patch(
        FancyArrowPatch(
            (0.37, 0.805),
            (0.49, 0.805),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.1,
            color=MUTED,
        )
    )
    ax.text(0.43, 0.855, r"$P_{2\rightarrow30}$", ha="center", fontsize=8.2, color=BASE.COLORS["assembly"])

    assembly_nodes = {
        "a": (0.54, 0.79), "b": (0.62, 0.89), "c": (0.70, 0.79),
        "d": (0.62, 0.69), "e": (0.62, 0.79),
    }
    assembly_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
    ]
    BASE.draw_framework(ax, assembly_nodes, assembly_edges, scale=0.90)
    ax.text(0.62, 0.615, r"connected assembly, $N=30$", ha="center", fontsize=8.0)
    ax.add_patch(
        FancyArrowPatch(
            (0.73, 0.79),
            (0.81, 0.79),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.05,
            color=MUTED,
        )
    )
    peak_x = np.array([0.04, 0.18, 0.35, 0.56, 0.76, 0.91])
    peak_h = np.array([0.28, 0.78, 0.42, 0.62, 0.33, 0.48])
    x0, y0, width, height = 0.84, 0.755, 0.14, 0.105
    ax.plot([x0, x0 + width], [y0, y0], color="#67727A", lw=0.85)
    for px, ph in zip(peak_x, peak_h):
        ax.plot([x0 + px * width] * 2, [y0, y0 + ph * height], color="#67727A", lw=1.15)
    ax.text(0.91, 0.685, "PXRD check", ha="center", fontsize=8.0, color="#4F5B64")

    ax.plot([0.02, 0.98], [0.535, 0.535], color="#D5DADE", lw=0.8)
    ax.text(
        0.02,
        0.495,
        "Growth of an established nuclei",
        fontweight="bold",
        fontsize=8.5,
        va="top",
    )

    seed_nodes = {
        "a": (0.075, 0.265), "b": (0.16, 0.365), "c": (0.245, 0.265),
        "d": (0.16, 0.165), "e": (0.16, 0.265),
    }
    seed_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
    ]
    BASE.draw_framework(ax, seed_nodes, seed_edges, scale=0.88)
    ax.text(0.14, 0.025, r"seed, $N=76$", ha="center", fontsize=8.0)
    ax.add_patch(
        FancyArrowPatch(
            (0.285, 0.265),
            (0.405, 0.265),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.1,
            color=MUTED,
        )
    )
    ax.text(0.345, 0.325, r"$P_{76\rightarrow126}$", ha="center", fontsize=8.2, color=BASE.COLORS["growth_probability"])

    grown_nodes = {
        "a": (0.47, 0.265), "b": (0.55, 0.365), "c": (0.63, 0.265),
        "d": (0.55, 0.165), "e": (0.55, 0.265), "f": (0.73, 0.365),
        "g": (0.81, 0.265), "h": (0.73, 0.165), "i": (0.73, 0.265),
    }
    grown_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
        ("c", "f"), ("f", "g"), ("g", "h"), ("h", "c"),
        ("c", "i"), ("f", "i"), ("g", "i"), ("h", "i"),
    ]
    BASE.draw_framework(ax, grown_nodes, grown_edges, new_nodes={"f", "g", "h", "i"}, scale=0.85)
    ax.text(0.64, 0.025, r"grown, $N=126$", ha="center", fontsize=8.0)
    ax.text(0.925, 0.265, r"$G_{76\rightarrow126}$", ha="center", fontsize=8.2, color=BASE.COLORS["growth_rate"])
    ax.text(0.925, 0.19, "conditional\ngrowth rate", ha="center", va="top", fontsize=7.8, color="#56616A")


def stacked_response(
    ax,
    cax,
    source: pd.DataFrame,
    value: str,
    *,
    cmap_name: str | mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    title: str,
    colorbar_label: str,
    show_y_label: bool,
    show_z_label: bool,
) -> None:
    cmap = (
        mpl.colormaps[cmap_name].copy()
        if isinstance(cmap_name, str)
        else cmap_name.copy()
    )
    cmap.set_bad(MISSING)
    x, y = np.meshgrid(np.log10(BASE.RHO_VALUES), BASE.Q_VALUES)
    for z_index, sc_value in enumerate(BASE.SC_VALUES):
        subset = source[np.isclose(source.S_close, sc_value)]
        values = (
            subset.pivot(
                index="atomic_Zr_to_BDC_ratio",
                columns="link_removal_to_addition_ratio",
                values=value,
            )
            .reindex(index=BASE.Q_VALUES, columns=BASE.RHO_VALUES)
            .to_numpy(float)
        )
        z = np.full_like(x, float(z_index))
        facecolors = cmap(norm(values))
        facecolors[..., 3] = np.where(np.isfinite(values), 0.90, 0.58)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0.24,
            edgecolor="white",
            antialiased=True,
        )
    ax.set_title(title, pad=1.5, fontsize=9.0, fontweight="bold")
    # Native 3D axis labels rotate and drift when the figure is scaled in Word.
    # Keep the ticks attached to the axes, but place the three coordinate names
    # in fixed 2D positions so they cannot enter neighboring panels.
    # Keep the native 3D labels in the SVG.  The earlier export replaced them
    # with floating annotations, which some SVG viewers positioned outside the
    # visible panel.  Explicit axis labels make the standalone C-E panels
    # self-contained in Inkscape and Word.
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$", labelpad=10, fontsize=8.7)
    ax.set_ylabel("Zr/BDC", labelpad=10, fontsize=8.7)
    ax.set_zlabel(r"$S_{\mathrm{close}}$", labelpad=8, fontsize=8.7)
    # Match the canonical Figure 3 axes exactly.  rho is displayed on a
    # logarithmic coordinate but labelled with all six sampled values;
    # S_close is an indexed stack labelled with all five sampled values.
    ax.set_xticks(
        np.log10(BASE.RHO_VALUES),
        [f"{value:g}" for value in BASE.RHO_VALUES],
    )
    ax.set_yticks([0.1, 1, 3, 10], ["0.1", "1", "3", "10"])
    ax.set_zticks(
        np.arange(len(BASE.SC_VALUES)),
        [f"{value:g}" for value in BASE.SC_VALUES],
    )
    if not show_y_label:
        ax.set_yticklabels([])
    if not show_z_label:
        ax.set_zticklabels([])
    ax.set_xlim(np.log10(BASE.RHO_VALUES[0]), np.log10(BASE.RHO_VALUES[-1]))
    ax.set_ylim(0.1, 10)
    ax.set_zlim(-0.15, len(BASE.SC_VALUES) - 0.85)
    ax.set_box_aspect((1.28, 1.00, 0.82), zoom=0.91)
    BASE.clean_3d_axes(ax)
    ax.tick_params(labelsize=7.5, pad=0.1)
    colorbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(colorbar_label, labelpad=1.5, fontsize=7.8)
    colorbar.ax.tick_params(labelsize=7.0, pad=1, length=2)


def anchor_comparison(ax, response: pd.DataFrame) -> None:
    BASE.anchor_comparison(ax, response)
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$", labelpad=3)
    ax.set_ylabel("Probability", labelpad=4)
    ax.set_title("Nucleation is suppressed before established growth", fontweight="bold", pad=14)
    # Keep the fixed-condition note in the title-to-plot gutter.  It must not
    # cover the probability curves or any of their markers.
    ax.text(
        0.98,
        1.015,
        r"Zr/BDC = 1; $S_{\mathrm{close}}=1$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
        clip_on=False,
    )
    ax.tick_params(labelsize=7.5)
    legend = ax.get_legend()
    if legend is not None:
        if hasattr(legend, "set_loc"):
            legend.set_loc("lower left")
        else:
            legend._loc = 3  # Matplotlib < 3.8 compatibility: lower left.
        plt.setp(legend.get_texts(), fontsize=8.0)


def sensitivity_heatmap(
    ax,
    sensitivity: pd.DataFrame,
    process: str,
    sc_value: float,
    *,
    norm,
    cmap,
    show_y: bool,
    show_x: bool,
    row_label: str | None,
):
    image = BASE.sensitivity_heatmap(
        ax,
        sensitivity,
        process,
        sc_value,
        norm=norm,
        cmap=cmap,
        show_y=show_y,
        show_x=show_x,
        row_label=None,
    )
    ax.set_xlabel("")
    if show_y:
        ax.set_ylabel(
            (row_label + "\nZr/BDC") if row_label else "Zr/BDC",
            labelpad=4,
            fontsize=8.0,
            fontweight="bold" if row_label else "normal",
        )
    ax.tick_params(labelsize=7.2, length=2.2, pad=1.5)
    return image


def pxrd_panel(ax) -> None:
    rhos = list(PXRD_SOURCES)
    colors = [mpl.colormaps["viridis"](v) for v in np.linspace(0.12, 0.82, len(rhos))]
    offsets = np.arange(len(rhos), 0, -1, dtype=float)
    for rho, color, offset in zip(rhos, colors, offsets):
        data = pd.read_csv(PXRD_SOURCES[rho])
        ax.plot(
            data.two_theta_deg,
            data.intensity_smoothed_normalized + offset,
            color=color,
            lw=1.35,
        )
        ax.text(
            29.65,
            offset + 0.18,
            rf"$\rho_{{\mathrm{{off}}}}={rho:g}$, $N=10\,000$",
            color=color,
            ha="right",
            va="bottom",
            fontsize=8.2,
        )

    peaks = pd.read_csv(IDEAL_PEAKS)
    baseline = 0.12
    ax.hlines(baseline, 3, 30, color="#3F474D", lw=0.8)
    for row in peaks.itertuples():
        ax.vlines(
            row.two_theta_deg,
            baseline,
            baseline + 0.78 * row.intensity_normalized,
            color="#30383E",
            lw=1.9,
        )
    ax.text(
        29.65,
        baseline + 0.12,
        "ideal periodic UiO-66",
        color="#30383E",
        ha="right",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
    )
    ax.set_xlim(3, 30)
    ax.set_ylim(0, len(rhos) + 1.18)
    ax.set_xlabel(r"$2\theta$ (degrees, Cu K$\alpha$)", labelpad=4)
    ax.set_ylabel("Normalized Debye intensity (offset)", labelpad=4)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(
        r"PXRD of $N=10{,}000$ assemblies (Zr/BDC = 1, $S_{\mathrm{close}}=1$)",
        loc="left",
        fontweight="bold",
        pad=6,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.8)


def add_panel_letters(fig, panels) -> None:
    for letter, ax, dx in panels:
        box = ax.get_position()
        fig.text(
            box.x0 - dx,
            box.y1 + 0.028,
            letter,
            ha="right",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=TEXT,
        )


def render_figure(response, sensitivity, rate_max):
    # The composite is deliberately taller than v7.  This gives the natural-
    # aspect Panel A enough physical height; no panel is stretched to consume
    # a rectangular slot.
    fig = plt.figure(figsize=(7.48, 11.30), facecolor="white")
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.80, 1.18, 1.24, 0.84],
        left=0.080,
        right=0.975,
        top=0.976,
        bottom=0.055,
        hspace=0.34,
    )

    # Panel A contains the horizontal molecular pathway and needs the larger
    # share of the top row. Panel B remains readable at the narrower width,
    # while the reduced gutter keeps the paired panels visually connected.
    top_grid = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12)
    # Keep an undistorted, natural-aspect Panel A inside its layout slot.  The
    # invisible slot axis anchors the composite-level letter so the A position
    # does not move when Matplotlib centers the equal-aspect content axis.
    ax_a_letter_anchor = fig.add_subplot(top_grid[0, 0])
    ax_a_letter_anchor.axis("off")
    ax_a = fig.add_axes(ax_a_letter_anchor.get_position())
    definition_panel(ax_a)
    ax_b = fig.add_subplot(top_grid[0, 1])
    anchor_comparison(ax_b, response)
    # Preserve Panel B's native 4.8:3.5 plotting aspect rather than stretching
    # it vertically to match Panel A's square molecular canvas.
    ax_b.set_box_aspect(3.5 / 4.8)
    ax_b.set_anchor("N")
    # Reduce Panel B uniformly, anchor it to the right, and lower it slightly.
    # This creates clear space below the subtitle and preserves a clean gutter
    # from Panel A without changing Panel B's native aspect ratio.
    panel_b_box = ax_b.get_position()
    panel_b_scale = 0.86
    panel_b_vertical_drop = 0.012
    panel_b_width = panel_b_box.width * panel_b_scale
    panel_b_height = panel_b_box.height * panel_b_scale
    ax_b.set_position(
        [
            panel_b_box.x1 - panel_b_width,
            panel_b_box.y1 - panel_b_height - panel_b_vertical_drop,
            panel_b_width,
            panel_b_height,
        ]
    )

    response_grid = outer[1, 0].subgridspec(
        3,
        3,
        height_ratios=[1.0, 0.10, 0.10],
        wspace=0.15,
        hspace=0.02,
    )
    ax_c = fig.add_subplot(response_grid[0, 0], projection="3d")
    ax_d = fig.add_subplot(response_grid[0, 1], projection="3d")
    ax_e = fig.add_subplot(response_grid[0, 2], projection="3d")
    response_x = fig.add_subplot(response_grid[1, :])
    response_x.axis("off")
    response_x.text(0.5, 0.05, r"$\rho_{\mathrm{off}}$ (shared x coordinate)", ha="center", va="bottom", fontsize=8.2)
    cax_c = fig.add_subplot(response_grid[2, 0])
    cax_d = fig.add_subplot(response_grid[2, 1])
    cax_e = fig.add_subplot(response_grid[2, 2])
    stacked_response(
        ax_c,
        cax_c,
        response,
        "P_2_to_30",
        cmap_name=RESPONSE_BLUE,
        norm=Normalize(0, 1),
        title="Nucleation probability\n" + r"$P_{2\rightarrow30}$",
        colorbar_label="Probability",
        show_y_label=True,
        show_z_label=False,
    )
    stacked_response(
        ax_d,
        cax_d,
        response,
        "P_76_to_126",
        cmap_name=RESPONSE_BLUE,
        norm=Normalize(0, 1),
        title="Growth probability\n" + r"$P_{76\rightarrow126}$",
        colorbar_label="Probability",
        show_y_label=False,
        show_z_label=False,
    )
    stacked_response(
        ax_e,
        cax_e,
        response,
        "G_76_to_126_entities_per_model_time",
        cmap_name=RESPONSE_TEAL,
        norm=Normalize(0, rate_max),
        title="Growth rate\n" + r"$G_{76\rightarrow126}$",
        colorbar_label="Entities per model time",
        show_y_label=False,
        show_z_label=True,
    )

    f_grid = outer[2, 0].subgridspec(
        5,
        3,
        height_ratios=[1.0, 1.0, 0.48, 0.42, 0.22],
        wspace=0.055,
        hspace=0.17,
    )
    sc_display = [0.1, 1.0, 2.0]
    processes = [
        ("assembly_J_1_to_30", "Nucleation"),
        ("seed_growth_J_76_to_126", "Growth"),
    ]
    sensitivity_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    sensitivity_cmap = mpl.colormaps["RdBu_r"].copy()
    sensitivity_cmap.set_bad(MISSING)
    f_axes = []
    for row_index, (process, row_label) in enumerate(processes):
        for col_index, sc_value in enumerate(sc_display):
            axis = fig.add_subplot(f_grid[row_index, col_index])
            sensitivity_heatmap(
                axis,
                sensitivity,
                process,
                sc_value,
                norm=sensitivity_norm,
                cmap=sensitivity_cmap,
                show_y=col_index == 0,
                show_x=row_index == 1,
                row_label=row_label if col_index == 0 else None,
            )
            if row_index == 0:
                axis.set_title(rf"$S_{{\mathrm{{close}}}}={sc_value:g}$", fontweight="bold", pad=4)
            f_axes.append(axis)

    f_x = fig.add_subplot(f_grid[2, :])
    f_x.axis("off")
    f_x.text(
        0.5,
        0.02,
        r"Adjacent $\rho_{\mathrm{off}}$ interval",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    f_key = fig.add_subplot(f_grid[3, :])
    f_key.axis("off")
    cax = f_key.inset_axes([0.20, 0.20, 0.60, 0.28])
    colorbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=sensitivity_cmap,
        norm=sensitivity_norm,
        orientation="horizontal",
        extend="min",
    )
    colorbar.set_label(
        r"$\beta=\Delta\ln J/\Delta\ln\rho_{\mathrm{off}}$",
        labelpad=2.5,
        fontsize=8.3,
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=7.2, pad=1.5, length=2)
    f_note = fig.add_subplot(f_grid[4, :])
    f_note.axis("off")
    f_note.text(
        0.5,
        0.55,
        r"Gray: unresolved zero-event interval; values below $-3$ are clipped only for display.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#5F6971",
    )

    ax_g = fig.add_subplot(outer[3, 0])
    pxrd_panel(ax_g)

    add_panel_letters(
        fig,
        [
            ("A", ax_a_letter_anchor, 0.012),
            ("B", ax_b, 0.010),
            ("C", ax_c, 0.022),
            ("D", ax_d, 0.022),
            ("E", ax_e, 0.022),
            ("F", f_axes[0], 0.012),
            ("G", ax_g, 0.010),
        ],
    )
    return fig


def save_figure(fig, stem: Path, dpi: int = 450):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.035)


def export_panels(response, sensitivity, rate_max):
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 4.1))
    definition_panel(ax)
    STYLE.panel_letter(ax, "A", case="upper", dx=-0.06, dy=1.03, fontsize=11)
    save_figure(fig, PANEL_DIR / "figure3A_definition")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    anchor_comparison(ax, response)
    STYLE.panel_letter(ax, "B", case="upper", dx=-0.10, dy=1.05, fontsize=11)
    save_figure(fig, PANEL_DIR / "figure3B_reference_slice")
    plt.close(fig)

    specs = [
        ("C", "P_2_to_30", RESPONSE_BLUE, Normalize(0, 1), "Nucleation probability\n" + r"$P_{2\rightarrow30}$", "Probability"),
        ("D", "P_76_to_126", RESPONSE_BLUE_TEAL, Normalize(0, 1), "Crystal growth probability\n" + r"$P_{76\rightarrow126}$", "Probability"),
        ("E", "G_76_to_126_entities_per_model_time", RESPONSE_TEAL, Normalize(0, rate_max), "Growth rate\n" + r"$G_{76\rightarrow126}$", "Entities per model time"),
    ]
    for letter, value, cmap, norm, title, colorbar_label in specs:
        fig = plt.figure(figsize=(4.8, 4.45))
        # Keep the native 3D axis labels and remove the former auxiliary
        # coordinate row, which duplicated the labels and overlapped the key.
        grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.13], hspace=0.015)
        ax = fig.add_subplot(grid[0, 0], projection="3d")
        cax = fig.add_subplot(grid[1, 0])
        # Use a shorter and thinner centered color band so it remains a key
        # rather than becoming a dominant visual element.
        cax.set_position([0.20, 0.075, 0.60, 0.018])
        stacked_response(
            ax,
            cax,
            response,
            value,
            cmap_name=cmap,
            norm=norm,
            title=title,
            colorbar_label=colorbar_label,
            show_y_label=True,
            show_z_label=True,
        )
        ax.text2D(
            -0.08,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
        save_figure(fig, PANEL_DIR / f"figure3{letter}_response")
        plt.close(fig)

    # Export Panel F as the same two-by-three matrix with a dedicated key row.
    fig = plt.figure(figsize=(8.0, 4.8))
    grid = fig.add_gridspec(5, 3, height_ratios=[1, 1, 0.48, 0.42, 0.22], wspace=0.055, hspace=0.17)
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(MISSING)
    axes = []
    for row, (process, row_label) in enumerate([
        ("assembly_J_1_to_30", "Nucleation"),
        ("seed_growth_J_76_to_126", "Growth"),
    ]):
        for col, sc_value in enumerate([0.1, 1.0, 2.0]):
            ax = fig.add_subplot(grid[row, col])
            sensitivity_heatmap(
                ax,
                sensitivity,
                process,
                sc_value,
                norm=norm,
                cmap=cmap,
                show_y=col == 0,
                show_x=row == 1,
                row_label=row_label if col == 0 else None,
            )
            if row == 0:
                ax.set_title(rf"$S_{{\mathrm{{close}}}}={sc_value:g}$", fontweight="bold")
            axes.append(ax)
    shared_x = fig.add_subplot(grid[2, :])
    shared_x.axis("off")
    shared_x.text(0.5, 0.02, r"Adjacent $\rho_{\mathrm{off}}$ interval", ha="center", va="bottom", fontsize=8.0)
    key = fig.add_subplot(grid[3, :])
    key.axis("off")
    cax = key.inset_axes([0.20, 0.20, 0.60, 0.28])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal", extend="min")
    cb.set_label(r"$\beta=\Delta\ln J/\Delta\ln\rho_{\mathrm{off}}$", fontsize=8.3, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7.2)
    note = fig.add_subplot(grid[4, :])
    note.axis("off")
    note.text(0.5, 0.55, r"Gray: unresolved zero-event interval; values below $-3$ are clipped only for display.", ha="center", va="center", fontsize=8.0)
    STYLE.panel_letter(axes[0], "F", case="upper", dx=-0.13, dy=1.05, fontsize=11)
    save_figure(fig, PANEL_DIR / "figure3F_sensitivity")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    pxrd_panel(ax)
    STYLE.panel_letter(ax, "G", case="upper", dx=-0.06, dy=1.05, fontsize=11)
    save_figure(fig, PANEL_DIR / "figure3G_N10000_pxrd")
    plt.close(fig)


def main() -> None:
    STYLE.apply_figure_style(frame="open", font="Arial", sizes=(8.6, 7.6, 6.8), grid=False)
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    response, sensitivity, rate_max = prepare_data()
    shutil.copy2(RESPONSE_SOURCE, HERE / "figure3_response_surface_source_v11.csv")
    shutil.copy2(SENSITIVITY_SOURCE, HERE / "figure3_local_sensitivity_source_v11.csv")

    for path in [*PXRD_SOURCES.values(), IDEAL_PEAKS]:
        if not path.exists():
            raise FileNotFoundError(path)

    fig = render_figure(response, sensitivity, rate_max)
    save_figure(fig, OUTPUT_PREFIX)
    plt.close(fig)
    export_panels(response, sensitivity, rate_max)

    source_rows = [
        {"panel": "A", "source": str(BASE_SCRIPT), "status": "proposed explanatory schematic"},
        {"panel": "B", "source": str(HERE / "figure3_response_surface_source_v11.csv"), "status": "calculated reference slice"},
        {"panel": "C", "source": str(HERE / "figure3_response_surface_source_v11.csv"), "status": "calculated response surface; original blue palette"},
        {"panel": "D", "source": str(HERE / "figure3_response_surface_source_v11.csv"), "status": "calculated response surface; revised blue-teal palette"},
        {"panel": "E", "source": str(HERE / "figure3_response_surface_source_v11.csv"), "status": "calculated response surface; original teal palette"},
        {"panel": "F", "source": str(HERE / "figure3_local_sensitivity_source_v11.csv"), "status": "calculated"},
    ]
    for rho, path in PXRD_SOURCES.items():
        source_rows.append(
            {
                "panel": "G",
                "source": str(path),
                "status": f"calculated finite-cluster Debye PXRD; rho_off={rho:g}; N=10000",
                "sha256": sha256(path),
            }
        )
    source_rows.append(
        {
            "panel": "G",
            "source": str(IDEAL_PEAKS),
            "status": "calculated periodic UiO-66 reference reflections",
            "sha256": sha256(IDEAL_PEAKS),
        }
    )
    pd.DataFrame(source_rows).to_csv(HERE / "panel_source_manifest_blue_teal_CDE_v3.csv", index=False)

    metadata = {
        "canonical_figure": str(OUTPUT_PREFIX.with_suffix(".png")),
        "final_width_mm": 190,
        "panel_g_entity_count": 10000,
        "panel_g_rho_off_values": list(PXRD_SOURCES),
        "panel_g_excluded_peer": "rho_off=30: trajectory was censored at N=2922 after the finite 1-billion-event/24-hour attempt and is not shown as an N=10000 peer",
        "panel_g_reviewed_seed_shown": False,
        "panel_b_to_e_x_label": "rho_off",
        "panel_a_aspect_policy": "equal x/y display scaling; natural molecular geometry; no anisotropic stretch",
        "panel_b_native_box_aspect_height_over_width": 3.5 / 4.8,
        "panel_b_uniform_scale_in_composite": 0.86,
        "panel_b_vertical_drop_fraction": 0.012,
        "panel_b_annotation_policy": "fixed-condition note in title-to-plot gutter; no data overlap",
        "panel_order": ["definitions", "reference slice", "nucleation response", "growth-probability response", "growth-rate response", "local sensitivity", "N=10000 PXRD"],
        "presentation_revision_only": True,
        "response_palette_revision": {
            "panels": ["C", "D", "E"],
            "C_palette_unchanged": ["#EEF6FB", "#C9E0F0", "#91C1DC", "#5B9FC4", "#2F78A5"],
            "D_palette_revised": ["#EEF8FA", "#C6E5EC", "#8FC8D3", "#58A9B8", "#2D788D"],
            "E_palette_unchanged": ["#EEF9F7", "#C7E9E3", "#8BD1C6", "#4DAFA4", "#237F78"],
            "data_norms_unchanged": True,
        },
    }
    (HERE / "figure3_revision_blue_teal_CDE_v3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTPUT_PREFIX.with_suffix(".png"))


if __name__ == "__main__":
    main()
