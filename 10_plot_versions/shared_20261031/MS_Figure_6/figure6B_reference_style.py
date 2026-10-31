"""Reproduce the manually adjusted Figure 6B layout using the numeric renderer.

Reference: W:/光催化小组/0-papers-in-progress/2025-宋宇航-MOL_growth_mechanism/
Figures/Figures/Figure-261009/Figure6.svg (read-only style reference).
Only layout, font sizes, and label locations were measured in that SVG.
All 182 scatter coordinates and grouped medians come from source_data CSVs
through original_code/build_figure7_kinetic_window_v2.py:panel_b.
No SVG artwork or raster crop is used to generate this panel.
The source SVG is not needed at runtime.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt


# The reference uses this uniform mm-to-source-point transform for panel B.
MM_PER_POINT = 0.2262056
REFERENCE_AXES = (76.527301, 32.180425, 125.299760, 71.684065)
PLOT_WIDTH_PT = (REFERENCE_AXES[2] - REFERENCE_AXES[0]) / MM_PER_POINT
PLOT_HEIGHT_PT = (REFERENCE_AXES[3] - REFERENCE_AXES[1]) / MM_PER_POINT
FIGURE_WIDTH_PT, FIGURE_HEIGHT_PT = 302.4, 226.8
PLOT_LEFT_PT = 45.0
PLOT_BOTTOM_PT = FIGURE_HEIGHT_PT - 20.0 - PLOT_HEIGHT_PT


def reference_position(x_mm, y_mm):
    """Reference page coordinates -> independent panel figure coordinates."""
    left, top, right, bottom = REFERENCE_AXES
    x = (PLOT_LEFT_PT + (x_mm - left) / MM_PER_POINT) / FIGURE_WIDTH_PT
    y = (PLOT_BOTTOM_PT + (bottom - y_mm) / MM_PER_POINT) / FIGURE_HEIGHT_PT
    return x, y


def label(fig, text, x_mm, y_mm, size_mm, **kwargs):
    return fig.text(
        *reference_position(x_mm, y_mm), text,
        fontsize=size_mm / MM_PER_POINT,
        fontfamily='Arial', color='black', va='baseline', **kwargs,
    )


def new_figure():
    fig = plt.figure(figsize=(FIGURE_WIDTH_PT / 72, FIGURE_HEIGHT_PT / 72))
    ax = fig.add_axes([
        PLOT_LEFT_PT / FIGURE_WIDTH_PT,
        PLOT_BOTTOM_PT / FIGURE_HEIGHT_PT,
        PLOT_WIDTH_PT / FIGURE_WIDTH_PT,
        PLOT_HEIGHT_PT / FIGURE_HEIGHT_PT,
    ])
    return fig, ax


def finish(fig, ax, artist, q1):
    """Keep native data/annotations and apply the reference's final typography."""
    left, top, right, bottom = REFERENCE_AXES
    width, height = right - left, bottom - top
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['left'].set_linewidth(1.01235)
    ax.spines['bottom'].set_linewidth(1.01235)
    ax.tick_params(labelbottom=False, labelleft=False, length=3, width=0.6)

    # Explicit text positions preserve edits made to the original Inkscape
    # composition after its Matplotlib export (independent of font auto-layout).
    tick_font_mm = 2.01083
    for value in range(0, 13, 2):
        x_mm = left + value / 12.6 * width
        label(fig, str(value), x_mm, 74.287796, tick_font_mm, ha='center')
    for value in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y_mm = bottom - (value - 0.44) / (1.025 - 0.44) * height
        label(fig, f'{value:.1f}', 75.056961, y_mm + 0.566702,
              tick_font_mm, ha='right')
    label(fig, r'RMS thickness ($\mathrm{\AA}$)  $\leftarrow$ thinner',
          (left + right) / 2, 77.137001, 2.28136, ha='center')
    label(fig, 'Primary-layer metal fraction', 71.003029, (top + bottom) / 2,
          2.29839, ha='center', rotation=90, rotation_mode='anchor')
    label(fig, 'B', 67.626465, 33.119190, 4.6144, ha='left', fontweight='bold')

    cbar_left_mm, cbar_width_mm = 126.62510, 1.97518
    cbar_x, cbar_y = reference_position(cbar_left_mm, bottom)
    cax = fig.add_axes([
        cbar_x, cbar_y,
        cbar_width_mm / MM_PER_POINT / FIGURE_WIDTH_PT,
        height / MM_PER_POINT / FIGURE_HEIGHT_PT,
    ])
    cbar = fig.colorbar(artist, cax=cax)
    cbar.set_ticks(q1.rho)
    cbar.ax.tick_params(labelright=False, length=3, width=0.6)
    cbar.ax.tick_params(which='minor', length=2.2, width=0.6)
    cbar.outline.set_linewidth(0.6)
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    for value in q1.rho:
        fraction = math.log10(value / 0.1) / math.log10(30 / 0.1)
        y_mm = bottom - fraction * height + 0.566702
        label(fig, f'{value:g}', 130.07062, y_mm, tick_font_mm, ha='left')
    label(fig, r'$\rho_{\mathrm{off}}$', 134.49484, (top + bottom) / 2,
          2.74503, ha='center', rotation=90, rotation_mode='anchor')
    return cbar
