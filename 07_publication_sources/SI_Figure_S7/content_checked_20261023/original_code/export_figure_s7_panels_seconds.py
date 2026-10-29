"""Rebuild Figure S7 A-D as standalone, fully vector SVGs in seconds.

Usage:
    python export_figure_s7_panels_seconds.py --data-dir PATH --output-dir PATH

Requires numpy, pandas and matplotlib. The source CSV columns ending in _s
already contain t_s = 0.175 * u, applied by export_long_growth_data.py using
the 25 degC ligand-exchange reference (1/5.72 s^-1, rounded to 0.175 s).
Plot these seconds directly: do not divide by 0.175 or multiply again.
This is a room-temperature reference scaling, not an independently validated
absolute clock for framework growth. The 120 degC factor 0.2405 ms is not used.
The supplied representative CSV contains 30,000 sampled points for each of
I11, I12 and I14, including the subsequent rise. All supplied points are drawn;
the final SI's entity-axis window (0 to 35) is preserved. Exact first-passage
statistics come from first_passage_24.csv, not the sampled representative lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Optional task-local plotting dependency directory; never needed on a normal
# Python installation with numpy, pandas and matplotlib already installed.
if os.environ.get("FIGURE_S7_PLOT_DEPS"):
    sys.path.insert(0, os.environ["FIGURE_S7_PLOT_DEPS"])

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter


SECONDS_PER_ALGORITHM_UNIT = 0.175
THRESHOLDS = (5, 6, 9, 13, 20, 30)
INTERVALS = ((3, 5), (5, 6), (6, 9), (9, 13), (13, 20), (20, 30))
REPRESENTATIVES = (11, 12, 14)
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
SOURCE_FILES = (
    "representative_pre30_trajectories.csv",
    "first_passage_24.csv",
    "residence_pre30_by_case_entity.csv",
)


def load_data(data_dir: Path):
    representative, first, residence = [pd.read_csv(data_dir / f) for f in SOURCE_FILES]
    assert len(first) == 24 and first.case_index.is_unique
    assert set(first.case_index) == set(range(24))
    assert len(residence) == 24 * 29
    assert not residence.duplicated(["case_index", "entity"]).any()
    assert set(representative.case_index) == set(REPRESENTATIVES)
    assert not any(d.isna().any().any() for d in (representative, first, residence))
    first = first.set_index("case_index").sort_index()
    grid = residence.pivot(index="case_index", columns="entity", values="residence_time_s")
    assert list(grid.columns) == list(range(1, 30))
    assert list(grid.index) == list(first.index)
    assert np.isfinite(grid.to_numpy()).all() and (grid.to_numpy() > 0).all()
    np.testing.assert_allclose(grid.sum(axis=1), first.t_entity_30_s, rtol=1e-12, atol=1e-9)
    for a, b in INTERVALS:
        delta = first[f"t_entity_{b}_s"] - first[f"t_entity_{a}_s"]
        assert (delta > 0).all()
        np.testing.assert_allclose(delta, first[f"t_{a}_to_{b}_s"], rtol=1e-10, atol=1e-10)
    for case in REPRESENTATIVES:
        frame = representative.loc[representative.case_index.eq(case)]
        assert len(frame) == 30000
        assert frame.time_s.is_monotonic_increasing
    order = first.t_entity_30_s.sort_values(kind="stable").index
    assert [int(order[0]), int(order[len(order) // 2]), int(order[-1])] == list(REPRESENTATIVES)
    return representative, first, grid, order


def figure(title: str, heatmap: bool = False):
    # Every standalone SVG is 16 cm wide and 13 cm high. White background,
    # source palette and final-SI plot definitions are retained.
    fig = plt.figure(figsize=(16 / 2.54, 13 / 2.54), facecolor="white")
    bounds = [0.145, 0.205, 0.815, 0.685]
    if heatmap:
        bounds = [0.145, 0.205, 0.690, 0.685]
    ax = fig.add_axes(bounds)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return fig, ax


def save(fig, output_dir: Path, qa_dir: Path, name: str, source: str):
    svg = output_dir / (name + ".svg")
    fig.savefig(svg, format="svg", metadata={
        "Title": name,
        "Description": source + "; t_s = 0.175 * u (25 degC reference); CSV already in seconds, no additional conversion. All marks and text are vector elements.",
        "Creator": "export_figure_s7_panels_seconds.py",
        "Date": None,
    })
    fig.savefig(qa_dir / (name + ".png"), dpi=200, facecolor="white")
    root = ET.parse(svg).getroot()
    assert not root.findall(".//{http://www.w3.org/2000/svg}image"), "Raster image in SVG"
    assert root.findall(".//{http://www.w3.org/2000/svg}text"), "Editable text missing"
    assert root.findall(".//{http://www.w3.org/2000/svg}path"), "Vector paths missing"
    # Validate labels/legends stay on the canvas after the final render.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    legends = [*fig.legends, *(a.get_legend() for a in fig.axes if a.get_legend() is not None)]
    for legend in legends:
        box = legend.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= w and box.y1 <= h, (name, "Legend outside canvas")
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or (hasattr(obj, "get_text") and not obj.get_text()):
            continue
        if isinstance(obj, matplotlib.text.Annotation):
            continue
        box = obj.get_window_extent(renderer)
        # Tick locations outside the limits can still have invisible Text
        # children. Only explicit axis labels, titles and legends are checked.
        if any(obj is a.xaxis.label or obj is a.yaxis.label or obj is a.title for a in fig.axes):
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= w and box.y1 <= h, (name, obj.get_text())
    plt.close(fig)
    return {"file": svg.name, "bytes": svg.stat().st_size, "vector_only": True, "width_cm": 16, "height_cm": 13}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = Path(__file__).resolve().parent.parent
    parser.add_argument("--data-dir", type=Path, default=group / "source_data")
    parser.add_argument("--output-dir", type=Path, default=group / "time_seconds_20261027")
    parser.add_argument("--qa-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = args.qa_dir or args.output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    representative, first, residence, order = load_data(args.data_dir)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "FigureS7_seconds_20261027",
        "path.simplify": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "savefig.transparent": False,
    })
    outputs = []

    fig, ax = figure("A  Repeated waiting before growth")
    for case, color in zip(REPRESENTATIVES, COLORS):
        frame = representative.loc[representative.case_index.eq(case)]
        t30_s = float(first.loc[case, "t_entity_30_s"])
        ax.plot(frame.time_s.to_numpy(), frame.entity.to_numpy(),
                color=color, linewidth=0.65, label=rf"I{case}, $t_{{30}}$ = {t30_s:.0f} s", rasterized=False)
    ax.set_ylim(0, 35)
    # Rescale the original u-axis window and ticks to preserve the same view.
    ax.set_xlim(np.array([-7500, 161000]) * SECONDS_PER_ALGORITHM_UNIT)
    ax.set_yticks(range(0, 36, 5))
    ax.set_xticks(np.arange(0, 160001, 40000) * SECONDS_PER_ALGORITHM_UNIT)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Entity number")
    ax.legend(loc="upper center", fontsize=9, frameon=True, framealpha=0.92)
    outputs.append(save(fig, args.output_dir, qa_dir, "SI_Figure_S7_A", SOURCE_FILES[0]))

    fig, ax = figure("B  First passage to each size")
    # Empirical survival S(t) = number of trajectories with T > t divided by
    # 24. All selected trajectories reach all thresholds; none are censored.
    min_s = min(first[f"t_entity_{n}_s"].min() for n in THRESHOLDS)
    max_s = first.t_entity_30_s.max()
    left = min_s / 1.9
    right = max_s * 1.9
    for n, color in zip(THRESHOLDS, COLORS):
        times = np.sort(first[f"t_entity_{n}_s"].to_numpy())
        x = np.r_[times[0] / 1.02, times]
        y = np.r_[1.0, 1.0 - np.arange(1, len(times) + 1) / len(times)]
        ax.step(x, y, where="post", color=color, linewidth=1.35, label=f"N = {n}", rasterized=False)
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("First-passage time (s)")
    ax.set_ylabel("Fraction not yet reached")
    ax.legend(loc="lower center", bbox_to_anchor=(0.55, 0.018),
              bbox_transform=fig.transFigure, borderaxespad=0, ncols=3,
              fontsize=9, frameon=False, handlelength=2.3, columnspacing=2.1)
    outputs.append(save(fig, args.output_dir, qa_dir, "SI_Figure_S7_B", SOURCE_FILES[1]))

    fig, ax = figure("C  Paired transition intervals")
    intervals = [(first[f"t_entity_{b}_s"] - first[f"t_entity_{a}_s"]).to_numpy()
                 for a, b in INTERVALS]
    # Match the final SI's linear-value violin KDE, displayed on a log axis.
    # No log-space KDE, outlier removal or imputation is applied.
    violin = ax.violinplot(intervals, positions=np.arange(1, 7), widths=0.5,
                          showmedians=True, showextrema=False, points=100, bw_method="scott")
    for body in violin["bodies"]:
        body.set_facecolor(COLORS[0])
        body.set_edgecolor("none")
        body.set_alpha(0.30)
        body.set_rasterized(False)
    violin["cmedians"].set_color(COLORS[0])
    violin["cmedians"].set_linewidth(1.6)
    ax.set_yscale("log")
    ax.set_ylim(np.array([1e-4, 4e5]) * SECONDS_PER_ALGORITHM_UNIT)
    ax.set_yticks(10.0 ** np.arange(-4, 5, 2))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(np.arange(1, 7), [f"{a}\N{EN DASH}{b}" for a, b in INTERVALS])
    ax.set_xlabel("Entity-number interval")
    ax.set_ylabel("Transition interval (s)")
    outputs.append(save(fig, args.output_dir, qa_dir, "SI_Figure_S7_C", SOURCE_FILES[1]))

    fig, ax = figure("D  Residence below 30 entities", heatmap=True)
    # Convert to seconds BEFORE taking log10: compared with the old log10(u)
    # map, every value shifts by log10(0.175), not by a factor of 0.175.
    values = residence.loc[order, list(range(1, 30))].to_numpy()
    logged = np.log10(values)
    # pcolormesh keeps all 696 heatmap cells as SVG paths. imshow would embed
    # a bitmap, so it is deliberately avoided for the requested vector output.
    mesh = ax.pcolormesh(np.arange(0.5, 30.5), np.arange(-0.5, 24.5), logged,
                         shading="flat", cmap="YlOrRd", rasterized=False,
                         edgecolors="face", linewidth=0.15)
    ax.set_ylim(23.5, -0.5)
    ax.set_xlim(0.5, 29.5)
    ax.set_xticks([1, 5, 9, 13, 17, 21, 25, 29])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Entity number")
    ax.set_ylabel(r"Trajectory ordered by $t_{30}$")
    color_ax = fig.add_axes([0.855, 0.205, 0.030, 0.685])
    cb_ticks = np.arange(np.ceil(logged.min()), np.floor(logged.max()) + 1)
    cb = fig.colorbar(mesh, cax=color_ax, ticks=cb_ticks)
    cb.set_label(r"$\log_{10}(\mathrm{Residence\ time}/\mathrm{s})$", labelpad=8)
    cb.ax.tick_params(labelsize=9)
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.solids.set_linewidth(0.15)
    outputs.append(save(fig, args.output_dir, qa_dir, "SI_Figure_S7_D", SOURCE_FILES[2]))

    report = {
        "figure": "SI Figure S7",
        "seconds_per_algorithm_unit": SECONDS_PER_ALGORITHM_UNIT,
        "reference_temperature_C": 25,
        "conversion_equation": "t_s = 0.175 * u",
        "input_time_coordinate": "seconds already exported as u * 0.175; CSV _s columns",
        "output_time_coordinate": "seconds (s); room-temperature ligand-exchange reference scaling",
        "additional_multiplier_applied_to_csv_times": 1.0,
        "number_of_trajectories": len(first),
        "number_of_residence_cells": int(values.size),
        "t30_exact_s": {f"I{case}": float(first.loc[case, "t_entity_30_s"])
                      for case in REPRESENTATIVES},
        "paired_interval_medians_s": {f"{a}-{b}": float(np.median(v))
                                      for (a, b), v in zip(INTERVALS, intervals)},
        "residence_case_indices_top_to_bottom": [int(i) for i in order],
        "residence_log10_seconds_range": [float(logged.min()), float(logged.max())],
        "residence_total_max_absolute_error_s": float(
            (residence.sum(axis=1) - first.t_entity_30_s).abs().max()),
        "source_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        "outputs": outputs,
    }
    (qa_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
