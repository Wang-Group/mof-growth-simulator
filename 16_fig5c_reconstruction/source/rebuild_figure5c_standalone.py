"""Rebuild manuscript Figure 5C from archived condition-level BTB-MOL results.

This is an independently written, self-contained numerical and plotting route.
It does not claim to reproduce the missing historical analysis module's mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "input_data" / "common_response_surface_v2_final_condition_summary.csv"
DEFAULT_OUTPUT = HERE / "Figure5C_standalone_transparent_outline_larger_text"
GRID_SHAPE = (10, 14, 9)  # rho_off, M/BTB, s_close
FINE_GRID = (34, 38, 22)
LEVEL = 0.5
DISPLAY_SEED = 20260901
DISPLAY_FRACTION = 0.20


def load_conditions(path: Path):
    frame = pd.read_csv(path)
    frame = frame.loc[frame["system"] == "btb_mol"].copy()
    columns = ["off_weight", "q", "SC", "trace_count", "target_count",
               "persistent_assembly_probability"]
    if len(frame) != np.prod(GRID_SHAPE) or frame[columns].isna().any().any():
        raise ValueError("Expected 1260 complete BTB-MOL condition rows")
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if (frame[["off_weight", "q"]] <= 0).any().any():
        raise ValueError("Both logarithmic coordinates must be positive")
    if frame.duplicated(["off_weight", "q", "SC"]).any():
        raise ValueError("Repeated condition in the input table")
    if not (frame.trace_count == 32).all():
        raise ValueError("The archived campaign has 32 traces per condition")
    if ((frame.target_count < 0) | (frame.target_count > frame.trace_count)).any():
        raise ValueError("Target counts must be between zero and trace count")

    frame["P"] = frame.target_count / frame.trace_count
    if not np.allclose(frame.P, frame.persistent_assembly_probability, atol=1e-12):
        raise ValueError("Archived probabilities differ from target_count / trace_count")

    off = np.sort(frame.off_weight.unique())
    ratio = np.sort(frame.q.unique())
    close = np.sort(frame.SC.unique())
    if (len(off), len(ratio), len(close)) != GRID_SHAPE:
        raise ValueError("The input axes do not match the complete 10 x 14 x 9 grid")
    indexed = frame.set_index(["off_weight", "q", "SC"])["P"]
    keys = pd.MultiIndex.from_product([off, ratio, close], names=indexed.index.names)
    cube = indexed.reindex(keys).to_numpy().reshape(GRID_SHAPE)
    if not np.isfinite(cube).all():
        raise ValueError("Missing or nonfinite grid value; no imputation is allowed")
    axes = (np.log10(off), np.log10(ratio), close)
    return frame, axes, cube


def boundary_triangles(axes, cube):
    """Trilinear interpolation on a fine grid followed by marching tetrahedra."""
    fine = tuple(np.linspace(axis[0], axis[-1], n)
                 for axis, n in zip(axes, FINE_GRID))
    points = np.stack(np.meshgrid(*fine, indexing="ij"), axis=-1).reshape(-1, 3)
    interpolator = RegularGridInterpolator(axes, cube, method="linear",
                                           bounds_error=True)
    probability = interpolator(points)
    shape = tuple(len(axis) for axis in fine)
    origins = np.stack(np.meshgrid(*(np.arange(n - 1) for n in shape),
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    positions = origins[:, None, :] + corners[None, :, :]
    vertex_ids = np.ravel_multi_index(positions.reshape(-1, 3).T, shape).reshape(-1, 8)
    tetrahedra = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                           [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    tetra_ids = vertex_ids[:, tetrahedra].reshape(-1, 4)
    high = probability[tetra_ids] >= LEVEL
    patterns = (high * np.array([1, 2, 4, 8])).sum(axis=1)

    def crossing(ids, a, b):
        va, vb = probability[ids[:, a]], probability[ids[:, b]]
        start, end = points[ids[:, a]], points[ids[:, b]]
        left = np.zeros(len(ids))
        right = np.ones(len(ids))
        start_high = va >= LEVEL
        for _ in range(24):
            mid = (left + right) / 2
            mid_high = interpolator(start + mid[:, None] * (end - start)) >= LEVEL
            left = np.where(mid_high == start_high, mid, left)
            right = np.where(mid_high == start_high, right, mid)
        weight = np.where(va == LEVEL, 0, (left + right) / 2)
        return start + weight[:, None] * (end - start)

    faces = []
    for pattern in range(1, 15):
        ids = tetra_ids[patterns == pattern]
        if not len(ids):
            continue
        positive = [i for i in range(4) if pattern & (1 << i)]
        negative = [i for i in range(4) if not pattern & (1 << i)]
        if len(positive) == 1:
            polygon = [crossing(ids, positive[0], n) for n in negative]
        elif len(negative) == 1:
            polygon = [crossing(ids, negative[0], p) for p in positive]
        else:
            a, b = positive
            c, d = negative
            polygon = [crossing(ids, a, c), crossing(ids, a, d),
                       crossing(ids, b, d), crossing(ids, b, c)]
        polygon = np.stack(polygon, axis=1)
        faces.append(polygon[:, [0, 1, 2]])
        if polygon.shape[1] == 4:
            faces.append(polygon[:, [0, 2, 3]])
    if not faces:
        raise ValueError("No P=0.5 boundary occurs inside the sampled domain")
    triangles = np.concatenate(faces)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    return triangles[area > 1e-12]


def displayed_conditions(frame):
    rng = np.random.default_rng(DISPLAY_SEED)
    low = frame.loc[frame.P <= 0.1]
    transition = frame.loc[(frame.P > 0.1) & (frame.P < 0.9)]
    high = frame.loc[frame.P >= 0.9]

    def sample(part):
        ids = np.sort(rng.choice(len(part), round(len(part) * DISPLAY_FRACTION),
                                 replace=False))
        return part.iloc[ids]

    return sample(low), transition, sample(high)


def plot_panel(frame, triangles, output, point_style, font_scale):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                         "svg.fonttype": "none", "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(6.6, 5.5), facecolor="white")
    ax = fig.add_axes([0.06, 0.07, 0.91, 0.80], projection="3d")
    ax.view_init(elev=23, azim=-61)
    ax.set_box_aspect((1.0, 0.95, 0.72), zoom=1.16)
    blue, red, peach = "#5185c0", "#d26d4f", "#ecc2af"
    ax.add_collection3d(Poly3DCollection(
        triangles, facecolor=blue, edgecolor="none", alpha=0.48,
        linewidths=0, zsort="average"))

    low, transition, high = displayed_conditions(frame)
    point_classes = [(transition, peach, "#d79778", 0.80),
                     (low, blue, "#356f9f", 0.76),
                     (high, red, "#a9513a", 0.76)]
    for part, color, edge, opacity in point_classes:
        style = ({"facecolors": [to_rgba(color, 0.38)],
                  "edgecolors": [to_rgba(edge, 0.82)], "linewidths": 0.65}
                 if point_style == "outlined" else
                 {"c": color, "alpha": opacity, "edgecolors": "none"})
        ax.scatter(np.log10(part.off_weight), np.log10(part.q), part.SC,
                   s=22.0, depthshade=False, rasterized=False, **style)
    ax.set_xlim(np.log10(0.01), np.log10(1500))
    ax.set_ylim(np.log10(0.01), np.log10(100))
    ax.set_zlim(0, 2)
    ax.set_xticks(np.log10([0.01, 0.1, 1, 10, 100, 1500]))
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1500"])
    ax.set_yticks(np.log10([0.1, 1, 10, 100]))
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_zticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel(r"$\rho_{\mathrm{off}}$", labelpad=5)
    ax.set_ylabel("M/BTB", labelpad=7)
    ax.set_zlabel(r"$s_{\mathrm{close}}$", labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_size(10 * font_scale)
    ax.tick_params(labelsize=8 * font_scale * 1.18, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.92, 0.92, 0.92, 0.5))
        axis.pane.set_edgecolor("#a8a8a8")
        axis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 0.65)
        axis._axinfo["grid"]["linewidth"] = 0.65

    handles = [Patch(facecolor=blue, edgecolor="none",
                     label=r"Interpolated $P = 0.5$ boundary")]
    for color, edge, label in [(peach, "#d79778", r"Calculated: $0.1 \leq P \leq 0.9$"),
                               (blue, "#356f9f", r"Calculated: $P \leq 0.1$"),
                               (red, "#a9513a", r"Calculated: $P \geq 0.9$")]:
        handles.append(Line2D([], [], marker="o", ls="none", markersize=5.5,
                              markerfacecolor=(to_rgba(color, 0.38) if point_style == "outlined" else color),
                              markeredgecolor=(to_rgba(edge, 0.82) if point_style == "outlined" else "none"),
                              markeredgewidth=(0.65 if point_style == "outlined" else 0),
                              label=label))
    fig.text(0.015, 0.96, "C", ha="left", va="top", fontsize=17 * font_scale, weight="bold")
    fig.text(0.135, 0.96, "Calculated kinetic formation boundary",
             ha="left", va="top", fontsize=12 * font_scale)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.895),
               frameon=False, fontsize=8.3 * font_scale, handletextpad=0.55,
               labelspacing=0.27, borderaxespad=0)
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    return {"low": len(low), "transition": len(transition), "high": len(high)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point-style", choices=["outlined", "solid"], default="outlined")
    parser.add_argument("--font-scale", type=float, default=1.2)
    args = parser.parse_args()
    if not (0.75 <= args.font_scale <= 1.5):
        parser.error("--font-scale must be between 0.75 and 1.5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame, axes, cube = load_conditions(args.input)
    triangles = boundary_triangles(axes, cube)
    vertex_probability = RegularGridInterpolator(axes, cube, bounds_error=True)(
        triangles.reshape(-1, 3))
    displayed = plot_panel(frame, triangles, args.output, args.point_style, args.font_scale)
    record = {
        "implementation": "Independent self-contained reconstruction; not the missing historical module",
        "run_command": ("python rebuild_figure5c_standalone.py --point-style "
                        + args.point_style + " --font-scale " + str(args.font_scale)),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "system_filter": "btb_mol",
        "input_conditions": len(frame),
        "launched_traces_per_condition": 32,
        "probability": "target_count / trace_count",
        "probability_classes": {"P <= 0.1": int((frame.P <= 0.1).sum()),
                                "0.1 < P < 0.9": int(((frame.P > 0.1) & (frame.P < 0.9)).sum()),
                                "P >= 0.9": int((frame.P >= 0.9).sum())},
        "interpolation": "trilinear in log10(rho_off), log10(M/BTB), linear s_close",
        "missing_values": "fail; no imputation",
        "out_of_domain": "no extrapolation",
        "boundary": "P=0.5 marching tetrahedra on fine grid; intersections refined by bisection against the original trilinear interpolator",
        "fine_grid": FINE_GRID,
        "triangles": len(triangles),
        "max_abs_vertex_probability_error": float(np.max(np.abs(vertex_probability - LEVEL))),
        "mean_abs_vertex_probability_error": float(np.mean(np.abs(vertex_probability - LEVEL))),
        "display": {"sample_fraction_low_high": DISPLAY_FRACTION,
                    "seed": DISPLAY_SEED, "transition_all": True,
                    "plotted": displayed, "point_style": args.point_style,
                    "font_scale": args.font_scale,
                    "tick_label_size_pt": 8 * args.font_scale * 1.18,
                    "point_area_pt2": 22.0,
                    "fill_alpha": (0.38 if args.point_style == "outlined" else "class-specific 0.76-0.80"),
                    "edge_alpha": (0.82 if args.point_style == "outlined" else 0)},
        "outputs": [str(args.output.with_suffix(ext).resolve()) for ext in (".svg", ".png")],
    }
    args.output.with_name(args.output.name + "_method.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
