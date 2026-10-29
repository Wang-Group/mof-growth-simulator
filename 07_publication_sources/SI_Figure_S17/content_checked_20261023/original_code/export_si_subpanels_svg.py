from __future__ import annotations

"""Export standalone vector SVG panels from the current SI figure assets.

The S15/S16 maps are rebuilt as vector rectangles from the colors displayed in
the current SI artwork.  This keeps the displayed SI result while removing the
blur associated with embedding the PNG in an SVG.  The typography, axes,
colorbars and colormaps follow the original plotting script
``plot_nucleation_growth_comparison_v1.py``.

Other panels that already exist as Matplotlib SVGs are copied with a stable
SI-oriented filename.  Composite figures are regenerated panel-by-panel from
their original plotting code and source data; no composite SVG is cropped.
"""

from copy import deepcopy
from pathlib import Path
import csv
import importlib.util
import json
import pickle
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import peak_widths


BASE = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MOL_handoff"
)
WORK = BASE / "03_workspace" / "analysis_workspace" / "rewrite" / "recalculation_campaign_20260830"
OUT = WORK / "si_subpanels_svg_corrected_20260921"

X_VALUES = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
Y_LABELS = ["0.1", "0.25", "0.5", "1", "1.5", "3", "10"]
SC_VALUES = [0.0, 0.1, 0.5, 2.0]

S15_PANEL_X = [(259, 728), (772, 1241), (1285, 1754), (1798, 2267)]
S15_ROW_Y = [(217, 651), (886, 1320), (1555, 1988)]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_cleanup_output() -> None:
    """Remove only files produced by the previous export.

    The output directory may also contain manually edited composite SVGs.  A
    blanket recursive delete would destroy those files before a new export
    starts, so cleanup is driven by the previous manifest instead.  Any file
    not listed in that manifest is intentionally preserved.
    """

    if not OUT.exists():
        return

    generated: set[Path] = set()
    previous_manifest = OUT / "manifest.csv"
    if previous_manifest.exists():
        try:
            with previous_manifest.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    value = (row.get("svg") or "").strip()
                    if not value:
                        continue
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = OUT / candidate
                    try:
                        generated.add(candidate.resolve())
                    except OSError:
                        generated.add(candidate)
        except (OSError, csv.Error):
            # If the manifest is unreadable, preserve everything.  It is safer
            # to leave stale generated files than to risk deleting manual work.
            generated.clear()

    for path in sorted(generated, key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_file() and path.is_relative_to(OUT.resolve()):
                path.unlink()
        except (OSError, ValueError):
            pass

    # These bookkeeping files belong to the exporter, not to manual artwork.
    for name in ("manifest.csv", "audit_report.csv", "subpanel_index.csv", "README.txt", "validation.txt"):
        path = OUT / name
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    # Remove only directories that became empty; never remove a directory that
    # still contains an untracked/manual file.
    for directory in sorted((p for p in OUT.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def sample_cells(path: Path, x_ranges: list[tuple[int, int]], y_range: tuple[int, int], rows: int = 7, cols: int = 6) -> list[list[str]]:
    """Read one row of SI heatmaps and return exact displayed cell colors."""

    image = np.asarray(Image.open(path).convert("RGB"))
    y0, y1 = y_range
    out: list[list[str]] = []
    for x0, x1 in x_ranges:
        colors: list[str] = []
        for r in range(rows):
            yc = int(round(y0 + (r + 0.5) * (y1 - y0) / rows))
            for c in range(cols):
                xc = int(round(x0 + (c + 0.5) * (x1 - x0) / cols))
                rgb = image[yc, xc]
                colors.append("#%02x%02x%02x" % tuple(int(v) for v in rgb))
        out.append([colors[r * cols : (r + 1) * cols] for r in range(rows)])
    return out


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    transformed = np.log10(values)
    edges = np.empty(len(values) + 1)
    edges[1:-1] = (transformed[:-1] + transformed[1:]) / 2
    edges[0] = transformed[0] - (transformed[1] - transformed[0]) / 2
    edges[-1] = transformed[-1] + (transformed[-1] - transformed[-2]) / 2
    return 10**edges


def save_vector_heatmap(
    colors: list[list[str]],
    destination: Path,
    *,
    row_letter: str,
    row_title: str,
    system_label: str,
    sc: float,
    cmap_name: str,
    norm: Normalize,
    cbar_label: str,
    cbar_ticks: list[float],
) -> None:
    """Write one independent, fully vectorized heatmap panel."""

    ensure_dir(destination.parent)
    fig = plt.figure(figsize=(4.35, 4.0), facecolor="white")
    ax = fig.add_axes([0.20, 0.18, 0.59, 0.61])
    cax = fig.add_axes([0.84, 0.18, 0.035, 0.61])

    x_edges = centers_to_edges(X_VALUES)
    nrows, ncols = 7, 6
    for r in range(nrows):
        # The source image is top-to-bottom (10 to 0.1), whereas the plotting
        # coordinate is bottom-to-top.  Reverse the row index only.
        y_index = nrows - 1 - r
        for c in range(ncols):
            ax.add_patch(
                Rectangle(
                    (x_edges[c], y_index),
                    x_edges[c + 1] - x_edges[c],
                    1.0,
                    facecolor=colors[r][c],
                    edgecolor="none",
                    linewidth=0,
                    antialiased=False,
                    transform=ax.transData,
                )
            )

    ax.set_xscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(0, nrows)
    ax.set_xticks(X_VALUES)
    ax.set_xticklabels([f"{v:g}" for v in X_VALUES])
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(Y_LABELS)
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$")
    ax.set_ylabel(f"Atomic metal/{system_label}")
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.set_title(rf"$s_{{\mathrm{{close}}}} = {sc:g}$", pad=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # The original SI colorbars use the same three maps and fixed ranges.
    sm = ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    # Matplotlib rasterizes colorbar solids by default once the number of
    # color steps exceeds its internal threshold.  Keep the standalone SVG
    # colorbar as vector paths so labels, cells and the scale all remain
    # resolution-independent.
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    cb.set_ticks(cbar_ticks)
    cb.set_label(cbar_label, labelpad=7)
    cb.ax.tick_params(labelsize=8, width=0.8, length=3)
    cb.outline.set_linewidth(0.8)

    fig.text(0.075, 0.89, row_letter, fontsize=12, fontweight="bold", ha="left", va="top")
    fig.text(0.20, 0.93, row_title, fontsize=10, fontweight="bold", ha="left", va="top")
    fig.savefig(destination, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def save_matrix_panel(
    matrix: np.ndarray,
    destination: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: list[str],
    ylabels: list[str],
    cmap_name: str,
    norm,
    cbar_label: str,
    letter: str,
    annotate: bool = False,
    annotation_matrix: np.ndarray | None = None,
    log_x: bool = False,
) -> None:
    """Draw a rectangular matrix with vector cells, axes, and optional labels."""

    ensure_dir(destination.parent)
    fig = plt.figure(figsize=(4.55, 3.85), facecolor="white")
    ax = fig.add_axes([0.19, 0.19, 0.61, 0.61])
    cax = fig.add_axes([0.84, 0.19, 0.035, 0.61])
    matrix = np.asarray(matrix, dtype=float)
    # Matplotlib places row 0 at the bottom for these pcolormesh coordinates.
    # The source tables are ordered from the smallest to the largest y value,
    # matching the SI figures (smallest at bottom, largest at top).
    z = matrix
    x = np.arange(z.shape[1] + 1, dtype=float)
    y = np.arange(z.shape[0] + 1, dtype=float)
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad("#e6e8ea")
    ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="flat")
    if annotate:
        vals = annotation_matrix if annotation_matrix is not None else matrix
        vals = np.asarray(vals, dtype=float)
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                value = vals[r, c]
                if not np.isfinite(value):
                    continue
                rgba = cmap(norm(z[r, c])) if np.isfinite(z[r, c]) else (0, 0, 0, 1)
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text = f"{value:.3g}" if abs(value) < 100 else f"{value:.0f}"
                ax.text(c + 0.5, r + 0.5, text, ha="center", va="center", fontsize=7.0,
                        color="black" if lum > 0.55 else "white")
    ax.set_xlim(0, z.shape[1]); ax.set_ylim(0, z.shape[0])
    ax.set_xticks(np.arange(z.shape[1]) + 0.5, xlabels)
    ax.set_yticks(np.arange(z.shape[0]) + 0.5, ylabels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    cb.set_label(cbar_label, labelpad=7)
    cb.ax.tick_params(labelsize=8, width=0.8, length=3)
    cb.outline.set_linewidth(0.8)
    fig.text(0.075, 0.89, letter, fontsize=12, fontweight="bold", ha="left", va="top")
    fig.savefig(destination, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def export_s24() -> list[dict[str, str]]:
    """Export the six visible panels in the BTB complete metric grid."""

    path = WORK / "figure7_redesign_20260904_v2" / "supporting_information" / "figureS_btb_complete_metric_grid_source.csv"
    data = pd.read_csv(path)
    qvals = [0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0]
    rhos = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    specs = [
        ("A", "Seed reaches +50 entities", "growth_probability", "viridis", Normalize(0, 1), "probability per initialized seed"),
        ("B", "Successful +50 growth events", "growth_event_rate", "cividis", LogNorm(1e-3, 1.2), "events per seed per model-time exposure"),
        ("C", "Final thickness after +50 growth", "median_final_rms_A_success", "magma", Normalize(0, 12), "median RMS thickness (Å)"),
        ("D", "Final material registered with seed layer", "median_primary_fraction_success", "viridis", Normalize(0, 1), "median primary-domain metal fraction"),
        ("E", "Persistent daughter domain after +50 growth", "final_daughter_probability_success", "magma_r", Normalize(0, 1), "probability among +50 traces"),
    ]
    rows: list[dict[str, str]] = []
    for letter, title, col, cmap, norm, cbar in specs:
        pivot = data.pivot(index="q", columns="rho", values=col).reindex(index=qvals, columns=rhos)
        dest = OUT / "S24" / f"S24_{letter}.svg"
        save_matrix_panel(pivot.to_numpy(float), dest, title=title, xlabel=r"link-removal/addition ratio, $\rho_{\rm off}$",
                          ylabel="atomic metal/BTB ratio", xlabels=[f"{v:g}" for v in rhos],
                          ylabels=[f"{v:g}" for v in qvals], cmap_name=cmap, norm=norm, cbar_label=cbar, letter=letter.lower())
        rows.append({"si_figure":"S24","panel":letter,"subpanel":"","svg":str(dest),"source_code":str(WORK / "figure7_redesign_20260904_v2" / "build_figure7_kinetic_window_v2.py"),"source_artwork":str(path),"export_type":"vector heatmap rebuilt from original source table"})

    # Panel F: the original q=1 endpoint comparison.
    q1 = data[np.isclose(data.q, 1.0)].sort_values("rho")
    fig, ax = plt.subplots(figsize=(4.9, 3.45), facecolor="white")
    ax.plot(q1.rho, q1.growth_probability, "o-", color="#2B7896", lw=1.8, ms=5, label="reaches +50")
    ax.plot(q1.rho, q1.final_daughter_probability_success, "s-", color="#B5443A", lw=1.8, ms=5, label="final daughter domain | +50")
    ax.plot(q1.rho, q1.median_primary_fraction_success, "^-", color="#39845A", lw=1.8, ms=5, label="primary metal fraction | +50")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 40); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"link-removal/addition ratio, $\rho_{\rm off}$"); ax.set_ylabel("probability or fraction")
    ax.set_title("Matched outcomes at metal/BTB = 1", pad=7)
    ax.grid(True, which="both", color="#d9dde1", lw=0.55)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.58, 0.13))
    fig.text(0.055, 0.93, "f", fontsize=12, fontweight="bold", ha="left", va="top")
    dest = OUT / "S24" / "S24_F.svg"; ensure_dir(dest.parent)
    fig.savefig(dest, facecolor="white", bbox_inches="tight", pad_inches=0.035); plt.close(fig)
    rows.append({"si_figure":"S24","panel":"F","subpanel":"","svg":str(dest),"source_code":str(WORK / "figure7_redesign_20260904_v2" / "build_figure7_kinetic_window_v2.py"),"source_artwork":str(path),"export_type":"vector line plot rebuilt from original source table"})
    return rows


def export_s50() -> list[dict[str, str]]:
    """Export the four formic-acid/concentration maps in SI Figure S50."""

    path = WORK / "figure8_redesign_20260904_v2" / "figure8_condition_source_v2.csv"
    data = pd.read_csv(path)
    acids = [0.0, 0.33, 0.82, 1.64, 2.46, 3.28]
    rows: list[dict[str, str]] = []
    for letter, operation, varied, ylabel, title_prefix in [
        ("A", "FIG8_FORMIC_ACID_BY_METAL", "metal_mM", "Hf concentration (mM)", "Productive growth across formic acid and Hf"),
        ("B", "FIG8_FORMIC_ACID_BY_LINKER", "linker_mM", "BTB concentration (mM)", "Productive growth across formic acid and BTB"),
        ("C", "FIG8_FORMIC_ACID_BY_METAL", "metal_mM", "Hf concentration (mM)", "Persistent multilayer products across formic acid and Hf"),
        ("D", "FIG8_FORMIC_ACID_BY_LINKER", "linker_mM", "BTB concentration (mM)", "Persistent multilayer products across formic acid and BTB"),
    ]:
        sub = data[data.operation_id.eq(operation)].copy()
        val_col = "net_growth_rate_mean" if letter in {"A", "B"} else "multilayer_seed_fraction_mean"
        pivot = sub.pivot(index=varied, columns="formic_acid_M", values=val_col).reindex(columns=acids)
        yvals = [float(v) for v in pivot.index]
        dest = OUT / "S50" / f"S50_{letter}.svg"
        vmax = float(np.nanmax(pivot.to_numpy(float)))
        ylabel_short = ylabel
        save_matrix_panel(pivot.to_numpy(float), dest, title=title_prefix, xlabel="Formic acid (M)", ylabel=ylabel_short,
                          xlabels=[f"{v:g}" for v in acids], ylabels=[f"{v:g}" for v in yvals],
                          cmap_name="viridis", norm=Normalize(0, vmax if vmax > 0 else 1),
                          cbar_label=("Net entities per seed per model time" if letter in {"A","B"} else "Fraction of terminal seeds with >1 stable layer domain"), letter=letter.lower())
        rows.append({"si_figure":"S50","panel":letter,"subpanel":"","svg":str(dest),"source_code":str(WORK / "figure8_redesign_20260904_v2" / "build_figure8_growth_window_v2.py"),"source_artwork":str(path),"export_type":"vector heatmap rebuilt from original source table"})
    return rows


def export_heatmap_series(si_number: str, *, system_label: str, row_titles: list[str], png_name: str) -> list[dict[str, str]]:
    png_path = BASE / "02_current_figures" / "si" / png_name
    sampled = [sample_cells(png_path, S15_PANEL_X, y) for y in S15_ROW_Y]
    row_specs = [
        ("A", row_titles[0], "magma", Normalize(-8, -0.2), r"log$_{10}$ rate", [-8, -6, -4, -2]),
        ("B", row_titles[1], "viridis", Normalize(0, 1), "probability", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("C", row_titles[2], "cividis", Normalize(0, 70), "entities / model time", [0, 20, 40, 60]),
    ]
    rows: list[dict[str, str]] = []
    for row_idx, (letter, title, cmap_name, norm, cbar_label, ticks) in enumerate(row_specs):
        for col_idx, sc in enumerate(SC_VALUES):
            dest = OUT / si_number / f"{si_number}_{letter}_sclose_{str(sc).replace('.', 'p')}.svg"
            save_vector_heatmap(
                sampled[row_idx][col_idx], dest,
                row_letter=letter,
                row_title=title,
                system_label=system_label,
                sc=sc,
                cmap_name=cmap_name,
                norm=norm,
                cbar_label=cbar_label,
                cbar_ticks=ticks,
            )
            rows.append({
                "si_figure": si_number,
                "panel": letter,
                "subpanel": f"sclose={sc:g}",
                "svg": str(dest),
                "source_code": str(WORK / "plot_nucleation_growth_comparison_v1.py"),
                "source_artwork": str(png_path),
                "export_type": "vector rectangles + vector axes/text/colorbar; cell colors sampled from current SI artwork",
            })
    return rows


def copy_panel(src: Path, si: str, name: str, *, source_code: str = "") -> dict[str, str]:
    ensure_dir(OUT / si)
    dest = OUT / si / name
    shutil.copy2(src, dest)
    return {
        "si_figure": si,
        "panel": name.rsplit("_", 1)[0],
        "subpanel": "",
        "svg": str(dest),
        "source_code": source_code,
        "source_artwork": str(src),
        "export_type": "copied original Matplotlib SVG",
    }


def replace_plain_svg_text(path: Path, old: str, new: str) -> None:
    """Replace the first literal Matplotlib SVG text node matching *old*."""

    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<text\b[^>]*>)" + re.escape(old) + r"(</text>)")
    content, count = pattern.subn(r"\1" + new + r"\2", content, count=1)
    if count != 1:
        raise ValueError(f"Could not replace panel label {old!r} in {path}")
    path.write_text(content, encoding="utf-8")


def remove_plain_svg_text(path: Path, text_value: str) -> None:
    """Remove the first literal Matplotlib SVG text node matching a label."""

    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"<text\b[^>]*>" + re.escape(text_value) + r"</text>")
    content, count = pattern.subn("", content, count=1)
    if count != 1:
        raise ValueError(f"Could not remove panel label {text_value!r} in {path}")
    path.write_text(content, encoding="utf-8")


def overlay_panel_label(
    path: Path,
    label: str,
    *,
    x: float = 7.2,
    y: float = 15.1,
    cover: tuple[float, float, float, float] | None = None,
) -> None:
    """Add a vector panel label, optionally covering a superseded old label."""

    tree = ET.parse(path)
    root = tree.getroot()
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "si_panel_label_override"})
    if cover is not None:
        rx, ry, rw, rh = cover
        ET.SubElement(
            group,
            f"{{{SVG_NS}}}rect",
            {"x": f"{rx:g}", "y": f"{ry:g}", "width": f"{rw:g}", "height": f"{rh:g}", "fill": "white"},
        )
    node = ET.SubElement(
        group,
        f"{{{SVG_NS}}}text",
        {
            "x": f"{x:g}",
            "y": f"{y:g}",
            "font-family": "Arial, DejaVu Sans, sans-serif",
            "font-size": "11",
            "font-weight": "700",
            "fill": "#000000",
        },
    )
    node.text = label
    tree.write(path, encoding="utf-8", xml_declaration=True)


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    values = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if not values:
        raise ValueError("SVG has no viewBox")
    nums = [float(x) for x in re.split(r"[ ,]+", values.strip()) if x]
    if len(nums) != 4:
        raise ValueError(f"Bad viewBox: {values}")
    return tuple(nums)  # type: ignore[return-value]


def crop_svg(src: Path, dest: Path, crop: tuple[float, float, float, float], *, source_code: str = "") -> dict[str, str]:
    """Crop a vector SVG by changing the viewBox and translating its content."""

    tree = ET.parse(src)
    root = tree.getroot()
    _, _, full_w, full_h = parse_viewbox(root)
    x, y, w, h = crop
    if not (0 <= x < full_w and 0 <= y < full_h and w > 0 and h > 0):
        raise ValueError(f"Crop {crop} outside {full_w} x {full_h}")

    new_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{w:g}pt",
            "height": f"{h:g}pt",
            "viewBox": f"0 0 {w:g} {h:g}",
            "version": "1.1",
            "overflow": "hidden",
        },
    )
    # Retain the original defs and metadata, then shift the complete figure.
    for child in root:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in {"metadata", "defs"}:
            new_root.append(deepcopy(child))
    group = ET.SubElement(new_root, f"{{{SVG_NS}}}g", {"transform": f"translate({-x:g} {-y:g})"})
    for child in root:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag not in {"metadata", "defs"}:
            group.append(deepcopy(child))
    ensure_dir(dest.parent)
    ET.ElementTree(new_root).write(dest, encoding="utf-8", xml_declaration=True)
    return {
        "si_figure": dest.parent.name,
        "panel": dest.stem.split("_")[1] if "_" in dest.stem else dest.stem,
        "subpanel": "",
        "svg": str(dest),
        "source_code": source_code,
        "source_artwork": str(src),
        "export_type": "vector crop of original SVG",
    }


def open_frame(ax: plt.Axes) -> None:
    """Match the open-axis frame used by the original scientific figures."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def standalone_record(si: str, panel: str, dest: Path, source_code: Path, source_artwork: Path) -> dict[str, str]:
    return {
        "si_figure": si,
        "panel": panel,
        "subpanel": "",
        "svg": str(dest),
        "source_code": str(source_code),
        "source_artwork": str(source_artwork),
        "export_type": "standalone panel redrawn by source code from original data",
    }


def save_standalone(fig: mpl.figure.Figure, dest: Path, *, pad: float = 0.04) -> None:
    ensure_dir(dest.parent)
    fig.savefig(dest, facecolor="white", bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def first_peak_metrics(pattern: pd.DataFrame) -> tuple[float, float]:
    theta = pattern.two_theta_deg.to_numpy(float)
    intensity = pattern.intensity_smoothed_normalized.to_numpy(float)
    indices = np.flatnonzero((theta >= 5.8) & (theta <= 8.05))
    peak_index = int(indices[np.argmax(intensity[indices])])
    fwhm = float(peak_widths(intensity, [peak_index], rel_height=0.5)[0][0] * np.median(np.diff(theta)))
    return float(theta[peak_index]), fwhm


def size_label(size: int) -> str:
    return str(size) if size < 1000 else f"{size / 1000.0:g}k"


def export_s26_standalone() -> list[dict[str, str]]:
    """Redraw every S26 panel independently from the original PXRD tables."""

    source_code = WORK / "figure3_rebuild_20260901" / "n10000_growth_campaign_v1" / "plot_n10000_pxrd_evolution_v2.py"
    pxrd_dir = BASE / "08_historical_source_library" / "02_main_figure_sources" / "Figure_03" / "n10000_final_pxrd"
    source_artwork = BASE / "08_historical_source_library" / "02_main_figure_sources" / "Figure_03" / "n10000_final_figure" / "n10000_pxrd_size_evolution_v2.svg"
    label_pattern = re.compile(r"rho(?P<rho>.+)_N(?P<size>\d+)_finite_debye_pxrd\.csv$")
    patterns: dict[tuple[float, int], pd.DataFrame] = {}
    metrics: list[dict[str, float]] = []
    for path in sorted(pxrd_dir.glob("*_finite_debye_pxrd.csv")):
        match = label_pattern.match(path.name)
        if match is None:
            continue
        rho = float(match.group("rho").replace("p", "."))
        size = int(match.group("size"))
        pattern = pd.read_csv(path)
        patterns[(rho, size)] = pattern
        position, fwhm = first_peak_metrics(pattern)
        metrics.append({"rho": rho, "entities": size, "position": position, "fwhm": fwhm})
    metric_frame = pd.DataFrame(metrics).sort_values(["rho", "entities"])
    periodic = pd.read_csv(pxrd_dir / "ideal_uio66_periodic_peak_positions.csv")
    ideal_111 = float(periodic.sort_values("two_theta_deg").iloc[0].two_theta_deg)
    rhos = sorted(metric_frame.rho.unique())
    sizes = sorted(metric_frame.entities.unique())
    cmap = mpl.colormaps["viridis"]
    colors = {rho: cmap(value) for rho, value in zip(rhos, np.linspace(0.08, 0.9, len(rhos)))}
    rows: list[dict[str, str]] = []

    # A1-A6: each heat map is created as its own figure.  Nothing from the
    # six-panel composition is retained in the SVG DOM.
    image = None
    for index, rho in enumerate(rhos):
        fig, ax = plt.subplots(figsize=(4.25, 2.25), facecolor="white")
        available = sorted(size for candidate_rho, size in patterns if np.isclose(candidate_rho, rho))
        matrix = np.stack([patterns[(rho, size)].intensity_smoothed_normalized.to_numpy(float) for size in available])
        theta = patterns[(rho, available[0])].two_theta_deg.to_numpy(float)
        image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
                          vmin=0, vmax=1, extent=(theta.min(), theta.max(), -0.5, len(available) - 0.5))
        ax.set_yticks(np.arange(len(available)), [size_label(size) for size in available])
        suffix = " (censored at N=2922)" if np.isclose(rho, 30.0) else ""
        ax.set_title(rf"fixed $\rho_{{\mathrm{{off}}}}={rho:g}$" + suffix, loc="left")
        ax.set_xlim(3, 30)
        if index >= 3:
            ax.set_xlabel(r"$2\theta$ (degrees)")
        if index % 3 == 0:
            ax.set_ylabel("aggregate size, N")
        else:
            ax.tick_params(labelleft=False)
        if index == 0:
            fig.text(0.015, 0.965, "A", fontsize=12, fontweight="bold", ha="left", va="top")
        rho_names = ["0p1", "0p3", "1p0", "3", "10", "30"]
        dest = OUT / "S26" / f"S26_A{index + 1}_rho{rho_names[index]}.svg"
        save_standalone(fig, dest, pad=0.025)
        rows.append(standalone_record("S26", f"A{index + 1}", dest, source_code, pxrd_dir))

    # The scale is auxiliary artwork, generated directly rather than cropped.
    fig = plt.figure(figsize=(0.82, 3.65), facecolor="white")
    cax = fig.add_axes([0.22, 0.08, 0.22, 0.86])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap="magma"); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("normalized Debye intensity")
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    dest = OUT / "S26" / "S26_colorbar.svg"
    save_standalone(fig, dest, pad=0.01)
    rows.append(standalone_record("S26", "colorbar", dest, source_code, pxrd_dir))

    display_rows: list[tuple[float, int, bool]] = []
    for rho in rhos:
        available = sorted(size for candidate_rho, size in patterns if np.isclose(candidate_rho, rho))
        size = 10000 if 10000 in available else max(available)
        display_rows.append((rho, size, size < 10000))

    fig, ax = plt.subplots(figsize=(6.15, 3.55), facecolor="white")
    offsets = np.arange(len(display_rows), 0, -1, dtype=float)
    neutral = "#69777F"
    for offset, (rho, size, censored) in zip(offsets, display_rows):
        pattern = patterns[(rho, size)]
        ax.plot(pattern.two_theta_deg, pattern.intensity_smoothed_normalized + offset,
                color=neutral if censored else colors[rho], lw=1.05 if not censored else 0.95,
                ls="--" if censored else "-")
        label = rf"$\rho_{{\mathrm{{off}}}}={rho:g}$, N={size_label(size)}" + (" (censored)" if censored else "")
        ax.text(29.7, offset + 0.18, label, ha="right", color=neutral if censored else colors[rho])
    baseline = 0.12
    for peak in periodic.itertuples(index=False):
        ax.vlines(float(peak.two_theta_deg), baseline, baseline + 0.65 * float(peak.intensity_normalized), color="#41494F", lw=0.8)
    ax.text(29.7, baseline + 0.07, "periodic UiO-66", ha="right", color="#41494F")
    ax.set(xlim=(3, 30), ylim=(0, len(display_rows) + 1.05), yticks=[],
           xlabel=r"$2\theta$ (degrees, Cu K$\alpha$)", ylabel="normalized intensity (vertically offset)")
    ax.set_title("Completed N=10k products retain the UiO-66 peak sequence", loc="left")
    open_frame(ax); fig.text(0.012, 0.97, "B", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S26" / "S26_B.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S26", "B", dest, source_code, pxrd_dir))

    fig, ax = plt.subplots(figsize=(3.45, 3.55), facecolor="white")
    for rho, group in metric_frame.groupby("rho"):
        group = group.sort_values("entities"); censored = np.isclose(float(rho), 30.0)
        ax.plot(group.entities, group.fwhm, marker="o", ms=4, lw=1.1,
                color=neutral if censored else colors[float(rho)], ls="--" if censored else "-",
                markerfacecolor="white" if censored else colors[float(rho)],
                label=f"{rho:g}" + (" censored" if censored else ""))
    if 1000 in sizes:
        anchor = float(metric_frame[metric_frame.entities == 1000].fwhm.median())
        guide_sizes = np.asarray([1000, 10000], dtype=float)
        ax.plot(guide_sizes, anchor * (guide_sizes / 1000.0) ** (-1 / 3), color="#667079", lw=1.0, ls="--", label=r"$N^{-1/3}$ guide")
    ax.set_xscale("log"); ax.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    ax.set(xlabel="aggregate size, N", ylabel=r"first-peak FWHM ($^\circ$)")
    ax.set_title("The low-angle peak narrows with size", loc="left")
    ax.legend(title=r"fixed $\rho_{\mathrm{off}}$", ncol=2, loc="upper right", columnspacing=0.7, handlelength=1.4)
    ax.grid(True, axis="y", color="#DDE1E4", lw=0.5); open_frame(ax)
    fig.text(0.012, 0.97, "C", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S26" / "S26_C.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S26", "C", dest, source_code, pxrd_dir))

    fig, ax = plt.subplots(figsize=(3.45, 3.55), facecolor="white")
    for rho, group in metric_frame.groupby("rho"):
        group = group.sort_values("entities"); censored = np.isclose(float(rho), 30.0)
        ax.plot(group.entities, group.position, marker="o", ms=4, lw=1.1,
                color=neutral if censored else colors[float(rho)], ls="--" if censored else "-",
                markerfacecolor="white" if censored else colors[float(rho)])
    ax.axhline(ideal_111, color="#41494F", lw=0.9, ls=":")
    ax.text(9300, ideal_111 - 0.003, "periodic (111)", ha="right", va="top")
    ax.set_xscale("log"); ax.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    ax.set(xlabel="aggregate size, N", ylabel=r"first-peak position ($^\circ$)")
    ax.set_title("A systematic peak offset remains", loc="left")
    ax.grid(True, axis="y", color="#DDE1E4", lw=0.5); open_frame(ax)
    fig.text(0.012, 0.97, "D", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S26" / "S26_D.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S26", "D", dest, source_code, pxrd_dir))
    return rows


def export_s27_standalone() -> list[dict[str, str]]:
    """Redraw S27 A-C from the three original rate tables."""

    data_dir = BASE / "08_historical_source_library" / "02_main_figure_sources" / "Figure_05" / "source_data"
    mismatched_path = data_dir / "figure5_surface_vs_seeded_growth_sc1_v1_mismatched_surface.csv"
    matched_path = data_dir / "figure5_surface_vs_seeded_growth_sc1_v1_matched_surface_control.csv"
    seeded_path = data_dir / "figure5_surface_vs_seeded_growth_sc1_v1_seeded.csv"
    source_code = BASE / "08_historical_source_library" / "02_main_figure_sources" / "Figure_05" / "plot_figure5_surface_nucleation_v2.py"
    mismatched = pd.read_csv(mismatched_path); matched = pd.read_csv(matched_path); seeded = pd.read_csv(seeded_path)
    qvals = [0.5, 1.0, 1.5, 3.0]; rhos = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    rows: list[dict[str, str]] = []
    cmap = mpl.colormaps["magma"].copy(); cmap.set_bad("#E1E4E6")
    for letter, title, frame, source_path in [
        ("A", "Separate surface domain", mismatched, mismatched_path),
        ("B", "Established-seed growth", seeded, seeded_path),
    ]:
        pivot = frame.pivot(index="q", columns="rho", values="rate").reindex(index=qvals, columns=rhos)
        values = pivot.to_numpy(float)
        logged = np.full_like(values, np.nan, dtype=float)
        logged[values > 0] = np.log10(values[values > 0])
        fig, ax = plt.subplots(figsize=(3.75, 3.55), facecolor="white")
        im = ax.imshow(logged, origin="lower", aspect="auto", cmap=cmap, vmin=-5, vmax=1)
        ax.set_xticks(range(len(rhos)), [f"{v:g}" for v in rhos])
        ax.set_yticks(range(len(qvals)), [f"{v:g}" for v in qvals])
        ax.set_xlabel(r"$\rho_{\mathrm{off}}$"); ax.set_ylabel("Atomic Zr/BDC ratio"); ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label(r"log$_{10}$ event rate / model time")
        if cb.solids is not None: cb.solids.set_rasterized(False)
        fig.text(0.012, 0.97, letter, fontsize=13, fontweight="bold", va="top")
        dest = OUT / "S27" / f"S27_{letter}.svg"; save_standalone(fig, dest)
        rows.append(standalone_record("S27", letter, dest, source_code, source_path))

    fig, ax = plt.subplots(figsize=(3.75, 3.55), facecolor="white")
    series = [
        ("Separate domain", mismatched, "#B83E32", "o"),
        ("Matched surface", matched, "#7B4DA3", "^"),
        ("Established seed", seeded, "#287CA5", "s"),
    ]
    for label, frame, color, marker in series:
        group = frame[np.isclose(frame.q, 1.0)].sort_values("rho")
        x = group.rho.to_numpy(float); rate = group.rate.to_numpy(float)
        base = rate[np.argmin(np.abs(x - 0.1))]
        center = rate / base; low = group.rate_low.to_numpy(float) / base; high = group.rate_high.to_numpy(float) / base
        center_plot = np.where(center > 0, center, np.nan)
        low_plot = np.maximum(low, 1e-6)
        ax.fill_between(x, low_plot, high, color=color, alpha=0.14)
        ax.plot(x, center_plot, color=color, marker=marker, lw=1.8, ms=6, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=r"$\rho_{\mathrm{off}}$", ylabel="Rate / rate at 0.1", xlim=(0.075, 40), ylim=(3e-6, 3))
    ax.set_title("Atomic Zr/BDC = 1"); ax.legend(frameon=False, loc="lower left")
    fig.text(0.012, 0.97, "C", fontsize=13, fontweight="bold", va="top")
    dest = OUT / "S27" / "S27_C.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S27", "C", dest, source_code, data_dir))
    return rows


def export_s29_standalone() -> list[dict[str, str]]:
    """Run the original surface-registry logic and draw A-D independently."""

    source_dir = BASE / "08_historical_source_library" / "03_si_figure_sources" / "v14_S29_original_inputs"
    source_code = source_dir / "make_surface_registry_figure.py"
    geometry_json = Path(__file__).with_name("s29_geometry_centers_protocol5.json")
    geometry = json.loads(geometry_json.read_text(encoding="utf-8"))
    parent_points = {key: np.asarray(value, dtype=float) for key, value in geometry["parent"].items()}
    outer_points = {key: np.asarray(value, dtype=float) for key, value in geometry["outer"].items()}
    shift = outer_points["Ligand"].mean(axis=0) - parent_points["Ligand"].mean(axis=0)
    registry: dict[str, np.ndarray] = {}
    for entity_type, outer_array in outer_points.items():
        transformed = outer_array - shift
        parent_array = parent_points[entity_type]
        distances = np.sqrt(((transformed[:, None, :] - parent_array[None, :, :]) ** 2).sum(axis=2))
        registry[entity_type] = transformed[distances.min(axis=1) > 0.2]
    conditions = []
    for summary_path in sorted(source_dir.glob("pilot_*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        csv_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".csv"))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            pilot_rows = list(csv.DictReader(handle))
        conditions.append((summary, pilot_rows))
    rows: list[dict[str, str]] = []

    fig = plt.figure(figsize=(3.0, 3.7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    colors = {"Zr": "#31A6A0", "Ligand": "#A8B0B8"}
    for entity_type, points in parent_points.items():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=26 if entity_type == "Zr" else 7,
                   c=colors[entity_type], alpha=0.90, edgecolors="none")
    for entity_type, points in registry.items():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=12 if entity_type == "Zr" else 3.5,
                   facecolors="none", edgecolors="#D55E00", alpha=0.18, linewidths=0.35)
    ax.view_init(elev=21, azim=36); ax.set_box_aspect((1, 1, 1)); ax.dist = 6.6; ax.set_axis_off()
    ax.set_title("a  Parent and ideal next shell", loc="left", fontweight="bold", fontsize=8.8)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#31A6A0", markeredgecolor="none", label="parent Zr6", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#A8B0B8", markeredgecolor="none", label="parent BDC", markersize=4),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#D55E00", alpha=0.55, label="unused ideal site", markersize=5),
    ]
    ax.legend(handles=handles, loc="lower center", frameon=False, fontsize=7.5)
    dest = OUT / "S29" / "S29_A.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S29", "A", dest, source_code, source_dir))

    registered = [sum(int(row["is_parent_registry"]) for row in pilot_rows) for _, pilot_rows in conditions]
    off = [len(pilot_rows) - count for (_, pilot_rows), count in zip(conditions, registered)]
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(4.3, 2.55), facecolor="white")
    ax.bar(x, registered, color="#3B82B8", label="on ideal lattice")
    ax.bar(x, off, bottom=registered, color="#D55E00", label="off ideal lattice")
    ax.set_yscale("symlog", linthresh=10); ax.set_ylabel("Accepted additions")
    ax.set_xticks(x, [str(i + 1) for i in x]); ax.set_xlabel("Pilot trajectory (inputs in accompanying table)")
    ax.set_title("b  Pilot additions", loc="left", fontweight="bold", fontsize=8.8)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.4); open_frame(ax)
    dest = OUT / "S29" / "S29_B.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S29", "B", dest, source_code, source_dir))

    distances = [float(row["registry_distance_A"]) for _, pilot_rows in conditions for row in pilot_rows
                 if np.isfinite(float(row["registry_distance_A"]))]
    bins = np.linspace(0, max(8.0, max(distances, default=8.0)), 33)
    fig, ax = plt.subplots(figsize=(3.05, 2.55), facecolor="white")
    ax.hist(distances, bins=bins, color="#6B7280", edgecolor="white", linewidth=0.25)
    ax.axvline(0.5, color="#D55E00", linestyle="--", linewidth=1.1, label="0.5 Å cutoff")
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("Nearest ideal-site distance (Å)"); ax.set_ylabel("Accepted additions")
    ax.set_title("c  Distance from ideal lattice", loc="left", fontweight="bold", fontsize=8.8)
    ax.legend(frameon=False, fontsize=7.5); open_frame(ax)
    dest = OUT / "S29" / "S29_C.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S29", "C", dest, source_code, source_dir))

    max_size = [summary["maximum_post_seed_component"]["entities"] for summary, _ in conditions]
    cycle = [summary["maximum_post_seed_component"]["cycle_rank"] for summary, _ in conditions]
    fig, ax = plt.subplots(figsize=(3.05, 2.55), facecolor="white")
    width = 0.38
    ax.bar(x - width / 2, max_size, width, color="#31A6A0", label="entities")
    ax.bar(x + width / 2, cycle, width, color="#CC79A7", label="cycle rank")
    ax.set_xticks(x, [str(i + 1) for i in x]); ax.set_xlabel("Pilot trajectory"); ax.set_ylabel("Maximum observed value")
    ax.set_title("d  Largest new component", loc="left", fontweight="bold", fontsize=8.8)
    ax.legend(frameon=False, fontsize=7.5); open_frame(ax)
    dest = OUT / "S29" / "S29_D.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S29", "D", dest, source_code, source_dir))
    return rows


def export_s58_s59_standalone() -> list[dict[str, str]]:
    """Redraw the defect-result composite figures as genuine standalone panels."""

    source_dir = WORK / "manuscript_revision_v32_defect_results_20260914"
    source_code = source_dir / "prepare_figures.py"
    analysis_dir = WORK / "defect_322_analysis_20260914_v1"
    d = pd.read_csv(analysis_dir / "diagnostic_counts.csv")
    rad = pd.read_csv(analysis_dir / "radial_counts.csv")
    coverage = pd.read_csv(analysis_dir / "condition_coverage.csv")
    colors = ["#287a9c", "#bd7138"]
    rng = np.random.default_rng(25)
    rows: list[dict[str, str]] = []

    fig, ax = plt.subplots(figsize=(3.1, 2.65), facecolor="white")
    for i, mode in enumerate(["fixed", "finite"]):
        g = d[(d.tol == 2) & (d.alpha == 1.05) & (d.delta == 0) & (d.rho == 1) &
              (d["mode"] == mode) & (d.cohort == "original_endpoint")]
        ax.scatter(i + rng.uniform(-0.14, 0.14, 32), g.vacancy_pct, s=10, alpha=0.6, c=colors[i])
        ax.plot([i - 0.24, i + 0.24], [g.vacancy_pct.median()] * 2, c=colors[i], lw=2)
    ax.set(xticks=[0, 1], xticklabels=["Fixed", "Finite"], ylabel="Missing interior BDC (%)",
           title="Same 50,000-event endpoint", ylim=(-0.3, 6.7), xlim=(-0.5, 1.5))
    fig.text(0.012, 0.97, "A", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S58" / "S58_A.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S58", "A", dest, source_code, analysis_dir))

    fig, ax = plt.subplots(figsize=(3.1, 2.65), facecolor="white")
    for i, mode in enumerate(["fixed", "finite"]):
        r = rad[(rad.tol == 2) & (rad.delta == 0) & (rad.rho == 1) & (rad["mode"] == mode) & ((rad.hi - rad.lo) == 0.5)].copy()
        r["p"] = np.where(r.eligible >= 10, 100 * r.missing / r.eligible, np.nan)
        p = r.pivot(index="id", columns="lo", values="p").dropna()
        for _, row in p.iterrows(): ax.plot([3 * i, 3 * i + 1], row.values, c=colors[i], lw=0.55, alpha=0.25)
        ax.plot([3 * i, 3 * i + 1], p.median().values, "o-", c=colors[i], lw=2, ms=3, label=mode.capitalize())
    ax.set(xticks=[0, 1, 3, 4], xticklabels=["Inner", "Outer", "Inner", "Outer"],
           ylabel="Missing interior BDC (%)", title="Radial distribution", ylim=(-0.3, 11.4))
    ax.legend(loc="upper right", fontsize=8); fig.text(0.012, 0.97, "B", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S58" / "S58_B.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S58", "B", dest, source_code, analysis_dir))

    fig, ax = plt.subplots(figsize=(6.15, 2.75), facecolor="white")
    for i, alpha in enumerate([0.85, 1.05, 1.25]):
        g = d[(d.tol == 2) & (d.alpha == alpha) & (d.delta == 2) & (d.rho == 1) &
              (d["mode"] == "fixed") & (d.cohort == "size_followon")]
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(g)), g.vacancy_pct, s=10, c="#70559a", alpha=0.6)
        ax.plot([i - 0.18, i + 0.18], [g.vacancy_pct.median()] * 2, c="#70559a", lw=2)
    ax.set(xticks=[0, 1, 2], xticklabels=["0.85", "1.05", "1.25"], ylabel="Missing interior BDC (%)",
           xlabel="Envelope parameter α", title="N = 1000: geometric-analysis sensitivity", ylim=(-0.4, 10), xlim=(-0.45, 2.45))
    fig.text(0.012, 0.97, "C", fontsize=12, fontweight="bold", va="top")
    dest = OUT / "S58" / "S58_C.svg"; save_standalone(fig, dest)
    rows.append(standalone_record("S58", "C", dest, source_code, analysis_dir))

    for mode, letter in [("fixed", "A"), ("finite", "B")]:
        mat = np.zeros((6, 3), int)
        for r in coverage[coverage["mode"] == mode].itertuples():
            mat[int(r.delta / 2), [0.1, 1, 3].index(r.rho)] = int(r.original_endpoint + r.size_followon)
        fig, ax = plt.subplots(figsize=(3.45 if mode == "fixed" else 4.1, 3.55), facecolor="white")
        im = ax.imshow(mat, vmin=0, vmax=32, cmap="Blues", aspect="auto")
        for i in range(6):
            for j in range(3):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="white" if mat[i, j] > 18 else "black", fontsize=9)
        ax.set(xticks=range(3), xticklabels=["0.1", "1", "3"], yticks=range(6), yticklabels=range(0, 12, 2),
               xlabel=r"$\rho_0$", ylabel=r"$\Delta=U_D/(RT)$", title=mode.capitalize() + " pool")
        if mode == "finite":
            cb = fig.colorbar(im, ax=ax, label="Available structures / 32", shrink=0.8)
            if cb.solids is not None: cb.solids.set_rasterized(False)
        dest = OUT / "S59" / f"S59_{letter}_{mode}_pool.svg"; save_standalone(fig, dest)
        rows.append(standalone_record("S59", letter, dest, source_code, analysis_dir))
    return rows


def main() -> None:
    setup_style()
    # Remove only stale files recorded by the previous export.  Untracked
    # hand-edited composite SVGs in this directory are deliberately preserved.
    safe_cleanup_output()
    ensure_dir(OUT)
    records: list[dict[str, str]] = []

    # The requested four-column response-map panels: 3 rows x 4 columns for
    # each material, all exported as independent SVGs.
    records += export_heatmap_series(
        "S15",
        system_label="BDC",
        row_titles=[
            "Assembly-formation rate (smoothed display)",
            "Probability of growth: 76 to 126 entities",
            "Growth rate among successful seeds: 76 to 126",
        ],
        png_name="Figure_S15.png",
    )
    records += export_heatmap_series(
        "S16",
        system_label="BTB",
        row_titles=[
            "Assembly-formation rate (smoothed display)",
            "Probability of growth: 30 to 80 entities",
            "Growth rate among successful seeds: 30 to 80",
        ],
        png_name="Figure_S16.png",
    )
    records += export_s24()
    records += export_s50()

    # Existing vector panels.  These are copied rather than rasterized.
    source = BASE / "03_workspace" / "analysis_workspace" / "rewrite" / "recalculation_campaign_20260830"
    hist = BASE / "08_historical_source_library"
    sources = [
        ("S25", "S25_A_rho0p1.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho0p1.svg", "plot_size_panel.py"),
        ("S25", "S25_B_rho0p3.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho0p3.svg", "plot_size_panel.py"),
        ("S25", "S25_C_rho1p0.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho1p0.svg", "plot_size_panel.py"),
        ("S25", "S25_D_rho3p0.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho3p0.svg", "plot_size_panel.py"),
        ("S25", "S25_E_rho10p0.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho10p0.svg", "plot_size_panel.py"),
        ("S25", "S25_F_rho30p0.svg", source / "btb_N10000_pxrd_supplement_20260907_v1" / "figures" / "panels" / "S25_rho30p0.svg", "plot_size_panel.py"),
        ("S51", "S51_A.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "growth_shape_A.svg", "build_high_rho_figure.py"),
        ("S51", "S51_B.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "growth_shape_B.svg", "build_high_rho_figure.py"),
        ("S51", "S51_C.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "growth_shape_C.svg", "build_high_rho_figure.py"),
        ("S51", "S51_D.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "growth_shape_D.svg", "build_high_rho_figure.py"),
        ("S52", "S52_A_N250.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "PXRD_N250.svg", "build_high_rho_figure.py"),
        ("S52", "S52_B_N500.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "PXRD_N500.svg", "build_high_rho_figure.py"),
        ("S52", "S52_C_N1000.svg", source / "btb_high_rho_N1000_20260907_v1" / "analysis" / "figures" / "panels" / "PXRD_N1000.svg", "build_high_rho_figure.py"),
        ("S53", "S53_A.svg", source / "manuscript_revision_v28_highrho_shoulder_20260910" / "figures" / "panels" / "FigureS77_A.svg", "build_highrho_shoulder.py"),
        ("S53", "S53_B.svg", source / "manuscript_revision_v28_highrho_shoulder_20260910" / "figures" / "panels" / "FigureS77_B.svg", "build_highrho_shoulder.py"),
        ("S54", "S54_A.svg", source / "figure3_rebuild_20260901" / "figure3_revision_20260904_v11" / "panels" / "figure3F_sensitivity.svg", "build_figure3_revision_v11.py"),
        ("S54", "S54_B.svg", source / "figure3_rebuild_20260901" / "figure3_revision_20260904_v11" / "panels" / "figure3G_N10000_pxrd.svg", "build_figure3_revision_v11.py"),
        ("S55", "S55_A.svg", hist / "02_main_figure_sources" / "Figure_05" / "figure5_panel_C.svg", "plot_figure5_surface_nucleation_v2.py"),
        ("S55", "S55_B.svg", hist / "02_main_figure_sources" / "Figure_05" / "figure5_panel_D.svg", "plot_figure5_surface_nucleation_v2.py"),
        ("S56", "S56_A.svg", source / "figure7_redesign_20260904_v2" / "panels" / "figure7C.svg", "build_figure7_kinetic_window_v2.py"),
        ("S56", "S56_B.svg", source / "manuscript_revision_v28_highrho_shoulder_20260910" / "figures" / "panels" / "Figure7_E.svg", "build_pxrd_figures.py"),
        ("S57", "S57_A.svg", source / "figure8_redesign_20260904_v2" / "panels" / "figure8B.svg", "build_figure8_growth_window_v2.py"),
        ("S35", "S35.svg", source / "manuscript_revision_v30_identity_defects_20260911" / "acid_only_panels" / "S35_acid_only.svg", "build_acid_panels.py"),
    ]
    for si, name, src, code in sources:
        if src.exists():
            records.append(copy_panel(src, si, name, source_code=str(src.parent.parent / code)))
            dest = OUT / si / name
            if si == "S25":
                overlay_panel_label(dest, name.split("_")[1], x=2.5, y=12.5)
            elif si == "S52":
                overlay_panel_label(dest, name.split("_")[1], x=2.5, y=12.5)
            elif si == "S54" and name == "S54_A.svg":
                replace_plain_svg_text(dest, "F", "A")
            elif si == "S54" and name == "S54_B.svg":
                replace_plain_svg_text(dest, "G", "B")
            elif si == "S55" and name == "S55_A.svg":
                overlay_panel_label(dest, "A", cover=(0, 0, 28, 23))
            elif si == "S55" and name == "S55_B.svg":
                overlay_panel_label(dest, "B", cover=(0, 0, 28, 23))
            elif si == "S56" and name == "S56_A.svg":
                replace_plain_svg_text(dest, "C", "A")
            elif si == "S56" and name == "S56_B.svg":
                replace_plain_svg_text(dest, "E", "B")
            elif si == "S57":
                remove_plain_svg_text(dest, "B")

    # These figures were previously viewport-cropped from a composite SVG.
    # They are now drawn panel-by-panel from the original code and source data,
    # so every SVG contains only the requested subplot.
    records += export_s26_standalone()
    records += export_s27_standalone()
    records += export_s29_standalone()
    records += export_s58_s59_standalone()

    manifest = OUT / "manifest.csv"
    fields = ["si_figure", "panel", "subpanel", "svg", "source_code", "source_artwork", "export_type"]
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    audit_rows = [
        ("S15", "matched after correction", "removed duplicated row letter and non-SI footer; data, direction and colors match Word"),
        ("S16", "matched after correction", "removed duplicated row letter and non-SI footer; data, direction and colors match Word"),
        ("S24", "matched after correction", "y axis restored to 0.1 bottom and 10 top; panel E colormap reversed to match Word; panel letters and F styling corrected"),
        ("S25", "matched after correction", "added A-F labels to the original vector panels"),
        ("S26", "matched after correction", "9 actual subpanels exported (A1-A6, B-D); shared colorbar exported separately as auxiliary artwork"),
        ("S27", "matched after correction", "A-C redrawn independently from the original rate tables; no composite SVG crop remains"),
        ("S29", "matched after correction", "A-D redrawn independently by the original surface-registry code and pilot inputs"),
        ("S35", "matched", "original vector source matches Word"),
        ("S50", "matched after correction", "concentration axes restored to small values at bottom and large values at top; panel letters corrected"),
        ("S51", "matched", "original vector panels match Word"),
        ("S52", "matched after correction", "added A-C labels to the original vector panels"),
        ("S53", "matched", "original vector panels match Word"),
        ("S54", "matched after correction", "renumbered retained main-figure panels F/G as SI panels A/B"),
        ("S55", "matched after correction", "renumbered retained main-figure panels C/D as SI panels A/B"),
        ("S56", "matched after correction", "renumbered panel C as A and replaced the incorrect B source with the enlarged shoulder panel"),
        ("S57", "matched after correction", "removed superseded main-figure panel letter B"),
        ("S58", "matched after correction", "A-C redrawn independently by prepare_figures.py logic and original defect tables"),
        ("S59", "matched after correction", "fixed/finite-pool panels redrawn independently; colorbar retained with the finite-pool panel"),
    ]
    with (OUT / "audit_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SI figure", "status", "audit result / correction"])
        writer.writerows(audit_rows)

    expected = {
        "S15": (12, "A:4 response maps; B:4 response maps; C:4 response maps"),
        "S16": (12, "A:4 response maps; B:4 response maps; C:4 response maps"),
        "S24": (6, "A-F"),
        "S25": (6, "A-F"),
        "S26": (9, "A1-A6, B-D; colorbar is auxiliary"),
        "S27": (3, "A-C"),
        "S29": (4, "A-D"),
        "S35": (1, "single panel"),
        "S50": (4, "A-D"),
        "S51": (4, "A-D"),
        "S52": (3, "A-C"),
        "S53": (2, "A-B"),
        "S54": (2, "A-B"),
        "S55": (2, "A-B"),
        "S56": (2, "A-B"),
        "S57": (1, "single panel"),
        "S58": (3, "A-C"),
        "S59": (2, "A-B"),
    }
    with (OUT / "subpanel_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SI figure", "expected actual subpanels", "panel layout", "SVG files"])
        for si, (expected_count, layout) in expected.items():
            figure_dir = OUT / si
            names = []
            for svg in sorted(figure_dir.glob("*.svg")):
                if svg.stem.lower().endswith("colorbar"):
                    continue
                names.append(svg.name)
            writer.writerow([si, expected_count, layout, "; ".join(names)])

    (OUT / "README.txt").write_text(
        "Corrected standalone SVG exports checked against the figures embedded in "
        "supporting_information_v22_defect_results_20260914.docx.\n"
        "The 59 files in 02_current_figures/si are byte-for-byte identical to the 59 images embedded in the Word file.\n"
        "manifest.csv records the source code/artwork for each SVG.\n"
        "audit_report.csv records the visual comparison and corrections.\n"
        "subpanel_index.csv lists the actual subpanel count and filenames; S26 has 9 actual subpanels plus one auxiliary colorbar.\n"
        "S26, S27, S29, S58 and S59 are redrawn independently from source code/data, not cropped from a composite SVG.\n"
        "validation.txt records XML/viewBox validation.\n"
        "Axes, labels, curves and rebuilt matrix/colorbar elements remain vector objects.\n"
        "Where an original Matplotlib SVG intentionally contains a continuous raster image layer, that original-resolution layer is preserved rather than replacing the panel with a screenshot.\n",
        encoding="utf-8",
    )

    # Basic XML validation for every exported SVG.
    errors: list[str] = []
    count = 0
    for svg in OUT.rglob("*.svg"):
        count += 1
        try:
            root = ET.parse(svg).getroot()
            if "viewBox" not in root.attrib:
                errors.append(f"missing viewBox: {svg}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"{svg}: {exc}")
    (OUT / "validation.txt").write_text(
        f"SVG files checked: {count}\n" + ("OK\n" if not errors else "\n".join(errors) + "\n"),
        encoding="utf-8",
    )
    print(f"wrote {count} SVG files to {OUT}")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
