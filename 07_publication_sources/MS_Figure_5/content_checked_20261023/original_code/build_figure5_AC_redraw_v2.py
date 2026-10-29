"""Redraw the current manuscript Figure 5 panels A-C as standalone SVGs.

The current manuscript numbering places the trajectory/response-surface figure
under Figure 5, while the provenance-preserving source files are named
``figure6_panel_A_v4.svg`` and ``figure6_panel_C_v4.svg`` in the historical
workspace. Panel A keeps the archived trajectory image, replaces the baked-in
one-unit y tick labels with two-unit labels, and uses a fixed 227:135
width:height canvas. Panels B and C are regenerated from their archived source
tables using the original variance-analysis and trilinear-boundary definitions.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent
A_SOURCE = SOURCE_DIR / "figure6_panel_A_v4.svg"
C_SOURCE = SOURCE_DIR / "figure6_panel_C_v4.svg"
SOURCE_TABLE = SOURCE_DIR.parent / "common_response_surface_v2_final_condition_summary.csv"
SENSITIVITY_TABLE = (
    HERE.parents[5]
    / "08_historical_source_library"
    / "02_main_figure_sources"
    / "Figure_06"
    / "source_data"
    / "figure6bc_current_parameter_phase_v1_sensitivity.csv"
)
A_OUTPUT = HERE / "Figure5_panel_A_redraw.svg"
C_OUTPUT = HERE / "Figure5_panel_C_redraw.svg"
B_OUTPUT = HERE / "Figure5_panel_B_redraw.svg"
A_PREVIEW = HERE / "Figure5_panel_A_redraw.png"
B_PREVIEW = HERE / "Figure5_panel_B_redraw.png"


def extract_panel_a_image() -> Image.Image:
    """Extract and orient the embedded trajectory raster from the source SVG."""
    text = A_SOURCE.read_text(encoding="utf-8")
    match = re.search(r"data:image/png;base64,([^\"]+)", text)
    if match is None:
        raise RuntimeError(f"Embedded panel-A PNG not found in {A_SOURCE}")
    image_bytes = base64.b64decode(match.group(1))
    image_path = HERE / "_panel_A_source.png"
    image_path.write_bytes(image_bytes)
    image = Image.open(image_path).convert("RGB")
    # Matplotlib's SVG backend writes the raster with an inverted y display.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image_path.unlink()
    return image


def panel_a() -> None:
    image = extract_panel_a_image()
    width, height = image.size

    # Remove the original baked-in one-unit y labels and tick marks. The
    # trajectory starts at the vertical axis around x=153 px; stopping at 170
    # px removes labels/ticks without removing the first data column.
    image = image.copy()
    ImageDraw.Draw(image).rectangle((0, 0, 170, height), fill="white")

    # The source crop was plotted with extent=(0, 1, 0.17, 1). These pixel
    # bounds are the original y=20 and y=1 axis positions in that crop.
    y_extent_bottom = 0.15
    y_extent_top = 1.0
    plot_top_px = 47.0
    plot_bottom_px = float(height - 1)
    original_axis_x = 153.0 / float(width - 1)
    image_x0, image_x1 = 0.10, 0.99
    image_axis_x = image_x0 + (image_x1 - image_x0) * original_axis_x

    # A higher figure DPI prevents Matplotlib's SVG backend from reducing the
    # archived 1783 px trajectory layer to a roughly 600 px embedded raster.
    fig = plt.figure(figsize=(6.8, 6.8 * 135.0 / 227.0), dpi=300, facecolor="white")
    # Reserve a narrow top band for the panel title so it does not overlap the
    # embedded "Intermediates" annotation in the archived trajectory image.
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.96])
    ax.imshow(
        image,
        extent=(image_x0, image_x1, y_extent_bottom, y_extent_top),
        aspect="auto",
        interpolation="lanczos",
        origin="upper",
        zorder=1,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Rebuild the y axis with labels every two entities.
    def y_position(value: float) -> float:
        py = plot_bottom_px - (value - 1.0) / 19.0 * (plot_bottom_px - plot_top_px)
        return y_extent_bottom + (1.0 - py / (height - 1.0)) * (y_extent_top - y_extent_bottom)

    axis_y0 = y_position(1.0)
    axis_y1 = y_position(20.0)
    ax.plot([image_axis_x, image_axis_x], [axis_y0, axis_y1], color="#202124", lw=0.8, zorder=3)
    for value in range(2, 21, 2):
        y = y_position(float(value))
        ax.plot([image_axis_x - 0.009, image_axis_x], [y, y], color="#202124", lw=0.8, zorder=3)
        ax.text(
            image_axis_x - 0.014,
            y,
            str(value),
            ha="right",
            va="center",
            fontsize=11.5,
            color="#202124",
            zorder=4,
        )
    ax.text(
        image_axis_x - 0.060,
        0.56,
        "Number of Entities",
        rotation=90,
        ha="center",
        va="center",
        fontsize=12.5,
        color="#202124",
        zorder=4,
    )

    # Original x ticks correspond to 0, 10, ..., 60 ms.
    tick_x = np.array([420, 780, 1135, 1495, 1855, 2210, 2570], dtype=float)
    tick_x = (tick_x - 180.0) / (2920.0 - 180.0)
    tick_x = image_x0 + (image_x1 - image_x0) * tick_x
    for x, label in zip(tick_x, ["0", "10", "20", "30", "40", "50", "60"]):
        ax.plot([x, x], [axis_y0 - 0.012, axis_y0], color="#202124", lw=0.8, zorder=3)
        ax.text(x, axis_y0 - 0.026, label, ha="center", va="top", fontsize=11.5, color="#202124", zorder=4)
    ax.text(
        0.55,
        0.020,
        "Elapsed growth time (ms)",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#202124",
        zorder=4,
    )
    ax.text(0.018, 1.035, "A", ha="left", va="top", fontsize=15, fontweight="bold", color="#202124", zorder=5, clip_on=False)
    ax.text(0.115, 1.035, "Representative stochastic trajectory", ha="left", va="top", fontsize=13, color="#202124", zorder=5, clip_on=False)

    fig.savefig(A_OUTPUT, format="svg", dpi=300, facecolor="white")
    fig.savefig(A_PREVIEW, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def panel_c() -> None:
    """Keep the approved C geometry and enlarge only its axis text layers."""
    text = C_SOURCE.read_text(encoding="utf-8")
    tick_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    label_ids = [7, 12, 18]

    def scale_text_group(svg: str, text_id: int, factor: float) -> str:
        pattern = re.compile(
            rf'(<g id="text_{text_id}">)(.*?<g transform="translate\(([-0-9.]+) ([-0-9.]+)\))',
            re.DOTALL,
        )
        match = pattern.search(svg)
        if match is None:
            raise RuntimeError(f"Cannot locate text_{text_id} in {C_SOURCE}")
        x, y = match.group(3), match.group(4)
        opening = (
            f'<g id="text_{text_id}" '
            f'transform="translate({x} {y}) scale({factor}) translate(-{x} -{y})">'
        )
        return svg[: match.start(1)] + opening + svg[match.end(1) :]

    for text_id in tick_ids:
        text = scale_text_group(text, text_id, 1.28)
    for text_id in label_ids:
        text = scale_text_group(text, text_id, 1.32)

    # The data geometry remains at its original coordinates. Extra right and
    # bottom canvas space prevents the enlarged z and x labels from clipping.
    text = re.sub(
        r'width="[0-9.]+pt" height="[0-9.]+pt" viewBox="0 0 [0-9.]+ [0-9.]+"',
        'width="350pt" height="340pt" viewBox="0 0 350 340"',
        text,
        count=1,
    )
    C_OUTPUT.write_text(text, encoding="utf-8")


def panel_b() -> None:
    """Redraw panel B natively from the response-surface data."""
    sensitivity = pd.read_csv(SENSITIVITY_TABLE).set_index("parameter")
    order = ["rho_off", "atomic_metal_linker_ratio", "s_close"]
    first = sensitivity.loc[order, "first_order"].to_numpy(float)
    total = sensitivity.loc[order, "total_effect"].to_numpy(float)
    first_low = sensitivity.loc[order, "first_order_ci_low"].to_numpy(float)
    first_high = sensitivity.loc[order, "first_order_ci_high"].to_numpy(float)
    total_low = sensitivity.loc[order, "total_effect_ci_low"].to_numpy(float)
    total_high = sensitivity.loc[order, "total_effect_ci_high"].to_numpy(float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.7), facecolor="white")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.25)

    y = np.arange(3, dtype=float)
    height = 0.32
    error_style = {"ecolor": "#202124", "elinewidth": 1.2, "capsize": 3, "capthick": 1.2}
    ax.barh(
        y + height / 2,
        first,
        height=height,
        color="#5185C0",
        edgecolor="none",
        xerr=np.vstack((first - first_low, first_high - first)),
        error_kw=error_style,
        label="First-order effect",
        zorder=3,
    )
    ax.barh(
        y - height / 2,
        total,
        height=height,
        color="#E99D4E",
        edgecolor="none",
        xerr=np.vstack((total - total_low, total_high - total)),
        error_kw=error_style,
        label="Total effect including interactions",
        zorder=3,
    )

    ax.set_yticks(y, [r"$\rho_{\mathrm{off}}$", "M/BTB", r"$s_{\mathrm{close}}$"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.12)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Fraction of variance in persistent-assembly probability", fontsize=13, labelpad=9)
    ax.set_title("Sensitivity of persistent-assembly probability", loc="left", fontsize=14, pad=9)
    ax.tick_params(axis="x", labelsize=12, direction="out", length=4, width=0.8)
    ax.tick_params(axis="y", labelsize=13, direction="out", length=4, width=0.8, pad=7)
    ax.grid(axis="x", color="#B0B0B0", alpha=0.22, linewidth=0.8, zorder=0)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(
        frameon=False,
        fontsize=11.5,
        loc="lower right",
        handlelength=1.8,
        labelspacing=0.35,
    )
    fig.text(0.035, 0.94, "B", ha="left", va="top", fontsize=16, fontweight="bold", color="#202124")

    fig.savefig(B_OUTPUT, format="svg", facecolor="white")
    fig.savefig(B_PREVIEW, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    panel_a()
    panel_b()
    panel_c()
    print(A_OUTPUT)
    print(B_OUTPUT)
    print(C_OUTPUT)


if __name__ == "__main__":
    main()
