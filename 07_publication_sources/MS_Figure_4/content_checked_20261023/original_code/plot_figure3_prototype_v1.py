"""Prototype Figure 3 with explicit kinetic definitions and structural checks.

This script intentionally avoids the former unqualified terms "nucleation
event" and "seed reaches +50".  Every plotted transition names its initial and
target size.  The current seeded dataset contains S_close = 0, 0.1, 0.5, 1,
and 2; S_close = 1.5 is not interpolated or invented.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REWRITE = CAMPAIGN.parent

COMMITTOR_PATH = CAMPAIGN / "common_response_surface_v2_final_pooled_committor.csv"
ASSEMBLY_SUMMARY_PATH = CAMPAIGN / "common_response_surface_v2_final_condition_summary.csv"
SEEDED_BASE_SUMMARY = (
    REWRITE
    / "simulation_review_gate_20260831"
    / "production_results_20260901"
    / "seeded_analysis_v3"
    / "seeded_v3_condition_summary.csv"
)
SEEDED_BASE_TRACES = (
    REWRITE
    / "simulation_review_gate_20260831"
    / "production_results_20260901"
    / "seeded_analysis_v3"
    / "seeded_v3_trace_metrics.csv"
)
SEEDED_SC1_SUMMARY = (
    CAMPAIGN
    / "figure_rebuild_20260901"
    / "sc1_seeded_analysis_v1"
    / "seeded_v3_condition_summary.csv"
)
SEEDED_SC1_TRACES = (
    CAMPAIGN
    / "figure_rebuild_20260901"
    / "sc1_seeded_analysis_v1"
    / "seeded_v3_trace_metrics.csv"
)
PXRD_DIR = HERE / "pxrd_ordered_sc1_q1_v2"

Q_VALUES = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0])
RHO_VALUES = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
SC_VALUES = np.array([0.0, 0.1, 0.5, 1.0, 2.0])

COLORS = {
    "assembly": "#C95F30",
    "growth_probability": "#2878A5",
    "growth_rate": "#2E7D5B",
    "text": "#20262B",
    "grid": "#D8DDE2",
    "missing": "#D9DDE1",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def combined_seeded_summary() -> pd.DataFrame:
    data = pd.concat(
        [pd.read_csv(SEEDED_BASE_SUMMARY), pd.read_csv(SEEDED_SC1_SUMMARY)],
        ignore_index=True,
    )
    keys = ["system", "q_atom", "off_weight", "SC"]
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicated seeded summary coordinates:\n{data.loc[duplicates, keys]}")
    return data


def combined_seeded_traces() -> pd.DataFrame:
    data = pd.concat(
        [pd.read_csv(SEEDED_BASE_TRACES), pd.read_csv(SEEDED_SC1_TRACES)],
        ignore_index=True,
    )
    keys = ["run_id"]
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicated seeded run IDs:\n{data.loc[duplicates, keys].head()}")
    return data


def interpolate_committor_to_seed_grid(committor: pd.DataFrame, sc_value: float) -> np.ndarray:
    subset = committor[
        (committor.system == "uio66")
        & (committor.entity_count == 2)
        & np.isclose(committor.SC, sc_value)
    ].copy()
    output = np.full((len(Q_VALUES), len(RHO_VALUES)), np.nan)
    for row_index, q_value in enumerate(Q_VALUES):
        row = subset[np.isclose(subset.q, q_value)].sort_values("off_weight")
        if len(row) < 2:
            continue
        output[row_index] = np.interp(
            np.log10(RHO_VALUES),
            np.log10(row.off_weight.to_numpy(float)),
            row.persistent_assembly_committor.to_numpy(float),
        )
    return output


def seeded_matrix(seed: pd.DataFrame, sc_value: float, value: str) -> np.ndarray:
    subset = seed[
        (seed.system == "uio66") & np.isclose(seed.SC, sc_value)
    ].copy()
    pivot = subset.pivot(index="q_atom", columns="off_weight", values=value)
    return pivot.reindex(index=Q_VALUES, columns=RHO_VALUES).to_numpy(float)


def response_source_table(committor: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for sc_value in SC_VALUES:
        assembly = interpolate_committor_to_seed_grid(committor, sc_value)
        reach = seeded_matrix(seed, sc_value, "target_reached_fraction")
        rate = seeded_matrix(seed, sc_value, "median_conditional_net_growth_rate")
        for iq, q_value in enumerate(Q_VALUES):
            for ir, rho_value in enumerate(RHO_VALUES):
                rows.append(
                    {
                        "system": "uio66",
                        "atomic_Zr_to_BDC_ratio": q_value,
                        "link_removal_to_addition_ratio": rho_value,
                        "S_close": sc_value,
                        "P_2_to_30": assembly[iq, ir],
                        "P_76_to_126": reach[iq, ir],
                        "G_76_to_126_entities_per_model_time": rate[iq, ir],
                        "P_2_to_30_rho_mapping": "linear interpolation on log10(rho) from de novo grid",
                    }
                )
    return pd.DataFrame(rows)


def condition_rates(assembly_summary: pd.DataFrame, seed_traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assembly = assembly_summary[assembly_summary.system == "uio66"].copy()
    assembly["rate"] = assembly.nucleation_event_count / assembly.candidate_model_time_exposure
    assembly = assembly.rename(columns={"off_weight": "rho", "q": "q_value"})
    assembly["process"] = "assembly_J_1_to_30"

    seed_rows = []
    seed_uio = seed_traces[seed_traces.system == "uio66"].copy()
    for keys, group in seed_uio.groupby(["q_atom", "off_weight", "SC"]):
        q_value, rho_value, sc_value = keys
        successes = int(group.target_reached.astype(bool).sum())
        exposure = float(group.model_time.sum())
        seed_rows.append(
            {
                "q_value": float(q_value),
                "rho": float(rho_value),
                "SC": float(sc_value),
                "rate": successes / exposure if exposure > 0 else math.nan,
                "successes": successes,
                "exposure": exposure,
                "process": "seed_growth_J_76_to_126",
            }
        )
    return assembly, pd.DataFrame(seed_rows)


def matched_assembly_rates(assembly_rates: pd.DataFrame) -> pd.DataFrame:
    """Interpolate positive assembly rates onto the seeded rho grid.

    Interpolation is linear in log(rate) versus log(rho). A target point is
    unresolved when either bracketing native rate is zero; zero-event cells are
    not replaced by pseudo-events.
    """
    rows = []
    subset = assembly_rates[
        assembly_rates.q_value.isin(Q_VALUES) & assembly_rates.SC.isin(SC_VALUES)
    ]
    for (q_value, sc_value), group in subset.groupby(["q_value", "SC"]):
        group = group.sort_values("rho")
        rho_source = group.rho.to_numpy(float)
        rate_source = group.rate.to_numpy(float)
        for rho_target in RHO_VALUES:
            exact = np.flatnonzero(np.isclose(rho_source, rho_target))
            if len(exact):
                rate = rate_source[exact[0]]
            else:
                upper = int(np.searchsorted(rho_source, rho_target, side="right"))
                lower = upper - 1
                if lower < 0 or upper >= len(rho_source):
                    rate = math.nan
                elif rate_source[lower] <= 0 or rate_source[upper] <= 0:
                    rate = math.nan
                else:
                    rate = float(
                        np.exp(
                            np.interp(
                                math.log(rho_target),
                                np.log(rho_source[[lower, upper]]),
                                np.log(rate_source[[lower, upper]]),
                            )
                        )
                    )
            rows.append(
                {
                    "q_value": float(q_value),
                    "rho": float(rho_target),
                    "SC": float(sc_value),
                    "rate": rate,
                    "process": "assembly_J_1_to_30",
                    "rho_mapping": "log(rate)-log(rho) interpolation; zero-bracket points unresolved",
                }
            )
    return pd.DataFrame(rows)


def local_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (process, q_value, sc_value), group in data.groupby(["process", "q_value", "SC"]):
        group = group.sort_values("rho")
        pairs = zip(group.to_dict("records")[:-1], group.to_dict("records")[1:])
        for lower, upper in pairs:
            rho_low = float(lower["rho"])
            rho_high = float(upper["rho"])
            rate_low = float(lower["rate"])
            rate_high = float(upper["rate"])
            resolved = (
                rho_low > 0
                and rho_high > 0
                and np.isfinite(rate_low)
                and np.isfinite(rate_high)
                and rate_low > 0
                and rate_high > 0
            )
            records.append(
                {
                    "process": process,
                    "q_value": float(q_value),
                    "S_close": float(sc_value),
                    "rho_low": rho_low,
                    "rho_high": rho_high,
                    "rho_geometric_midpoint": math.sqrt(rho_low * rho_high),
                    "rate_low": rate_low,
                    "rate_high": rate_high,
                    "resolved": int(resolved),
                    "beta_dlnJ_dlnrho": (
                        math.log(rate_high / rate_low) / math.log(rho_high / rho_low)
                        if resolved
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def clean_3d_axes(ax) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#BFC5CA")
    ax.tick_params(pad=0.5)
    ax.view_init(elev=27, azim=-56)


def stacked_response(
    ax,
    source: pd.DataFrame,
    value: str,
    *,
    cmap_name: str,
    norm: mpl.colors.Normalize,
    title: str,
    colorbar_label: str,
) -> None:
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(COLORS["missing"])
    x, y = np.meshgrid(np.log10(RHO_VALUES), Q_VALUES)
    for z_index, sc_value in enumerate(SC_VALUES):
        subset = source[np.isclose(source.S_close, sc_value)]
        values = (
            subset.pivot(
                index="atomic_Zr_to_BDC_ratio",
                columns="link_removal_to_addition_ratio",
                values=value,
            )
            .reindex(index=Q_VALUES, columns=RHO_VALUES)
            .to_numpy(float)
        )
        z = np.full_like(x, float(z_index))
        facecolors = cmap(norm(values))
        facecolors[..., 3] = np.where(np.isfinite(values), 0.88, 0.55)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0.18,
            edgecolor="white",
            antialiased=True,
        )
    ax.set_title(title, pad=2.0, fontweight="bold")
    ax.set_xlabel(r"link-removal/addition ratio, $\rho_{\rm off}$", labelpad=4)
    ax.set_ylabel("Zr/BDC", labelpad=4)
    ax.set_zlabel(r"sampled $S_{\rm close}$", labelpad=3)
    ax.set_xticks(np.log10(RHO_VALUES), [f"{value:g}" for value in RHO_VALUES])
    ax.set_yticks([0.1, 1, 3, 10], ["0.1", "1", "3", "10"])
    ax.set_zticks(np.arange(len(SC_VALUES)), [f"{value:g}" for value in SC_VALUES])
    ax.set_xlim(np.log10(RHO_VALUES[0]), np.log10(RHO_VALUES[-1]))
    ax.set_ylim(0.1, 10)
    ax.set_zlim(-0.15, len(SC_VALUES) - 0.85)
    ax.set_box_aspect((1.28, 1.18, 0.92))
    clean_3d_axes(ax)
    cax = inset_axes(
        ax,
        width="82%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0, -0.18, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    colorbar.set_label(colorbar_label, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.8, pad=1)


def draw_zr6_node(ax, center: tuple[float, float], scale: float = 1.0, color: str = "#2A7FA8") -> None:
    """Draw a recognizable but explicitly schematic Zr6 oxo-cluster glyph."""
    x0, y0 = center
    radius = 0.024 * scale
    angles = np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 6, endpoint=False)
    points = np.column_stack([x0 + radius * np.cos(angles), y0 + radius * np.sin(angles)])
    ax.add_patch(Polygon(points, closed=True, fc="#DCECF3", ec=color, lw=0.8, zorder=4))
    for x, y in points:
        ax.add_patch(Circle((x, y), 0.0062 * scale, fc=color, ec="white", lw=0.35, zorder=6))
    ax.add_patch(Circle((x0, y0), 0.0054 * scale, fc="#D55B45", ec="white", lw=0.3, zorder=6))


def draw_bdc_linker(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    scale: float = 1.0,
    color: str = "#59636B",
    zorder: int = 2,
) -> None:
    """Draw BDC as a benzene ring with carboxylate oxygen termini."""
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    vector = p1 - p0
    length = np.linalg.norm(vector)
    if length == 0:
        return
    unit = vector / length
    normal = np.array([-unit[1], unit[0]])
    midpoint = (p0 + p1) / 2
    ring_radius = min(0.018 * scale, 0.16 * length)
    ring_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + math.atan2(unit[1], unit[0])
    ring = np.array(
        [midpoint + ring_radius * (np.cos(angle) * unit + np.sin(angle) * normal) for angle in ring_angles]
    )
    left_ring = midpoint - ring_radius * unit
    right_ring = midpoint + ring_radius * unit
    left_o = p0 + 0.18 * vector
    right_o = p1 - 0.18 * vector
    ax.plot([p0[0], left_o[0], left_ring[0]], [p0[1], left_o[1], left_ring[1]], color=color, lw=0.9, zorder=zorder)
    ax.plot([right_ring[0], right_o[0], p1[0]], [right_ring[1], right_o[1], p1[1]], color=color, lw=0.9, zorder=zorder)
    ax.add_patch(Polygon(ring, closed=True, fill=False, ec=color, lw=0.85, zorder=zorder + 1))
    for oxygen in (left_o, right_o):
        ax.add_patch(Circle(tuple(oxygen), 0.0046 * scale, fc="#D55B45", ec="white", lw=0.25, zorder=zorder + 2))


def draw_framework(
    ax,
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    *,
    new_nodes: set[str] | None = None,
    scale: float = 1.0,
) -> None:
    new_nodes = new_nodes or set()
    for first, second in edges:
        draw_bdc_linker(ax, nodes[first], nodes[second], scale=scale)
    for name, center in nodes.items():
        color = "#53A9B8" if name in new_nodes else "#2A7FA8"
        draw_zr6_node(ax, center, scale=scale, color=color)


def draw_pxrd_gate(ax, x0: float, y0: float, width: float = 0.12, height: float = 0.09) -> None:
    peak_x = np.array([0.04, 0.18, 0.35, 0.56, 0.76, 0.91])
    peak_h = np.array([0.28, 0.78, 0.42, 0.62, 0.33, 0.48])
    ax.plot([x0, x0 + width], [y0, y0], color="#67727A", lw=0.7)
    for px, ph in zip(peak_x, peak_h):
        ax.plot([x0 + px * width] * 2, [y0, y0 + ph * height], color="#67727A", lw=1.0)
    ax.text(x0 + width / 2, y0 - 0.035, "PXRD", ha="center", va="top", fontsize=6.8, color="#4F5B64")


def definition_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top: a chemically recognizable two-entity precursor commits to a larger
    # connected assembly, followed by a distinct structural gate.
    ax.text(0.02, 0.95, "Nucleation pathway", fontweight="bold", fontsize=8.9)
    draw_zr6_node(ax, (0.085, 0.74), scale=1.05)
    draw_bdc_linker(ax, (0.105, 0.74), (0.245, 0.74), scale=1.05)
    ax.text(0.16, 0.64, r"Zr$_6$ + BDC", ha="center", fontsize=7.1)
    ax.text(0.16, 0.59, r"nascent pair, $N=2$", ha="center", fontsize=7.1, color="#56616A")
    ax.add_patch(FancyArrowPatch((0.285, 0.74), (0.405, 0.74), arrowstyle="-|>", mutation_scale=9, lw=1.05, color="#66717A"))
    ax.text(0.345, 0.78, r"$P_{2\rightarrow30}$", ha="center", fontsize=7.2, color=COLORS["assembly"])

    assembly_nodes = {
        "a": (0.47, 0.73), "b": (0.56, 0.82), "c": (0.65, 0.73),
        "d": (0.56, 0.64), "e": (0.56, 0.73),
    }
    assembly_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
    ]
    draw_framework(ax, assembly_nodes, assembly_edges, scale=0.78)
    ax.text(0.56, 0.56, r"connected assembly, $N=30$", ha="center", fontsize=7.0)
    ax.add_patch(FancyArrowPatch((0.70, 0.73), (0.79, 0.73), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#66717A"))
    draw_pxrd_gate(ax, 0.82, 0.70, width=0.15, height=0.12)

    ax.plot([0.02, 0.98], [0.49, 0.49], color="#D5DADE", lw=0.75)

    # Bottom: show that the growth observable starts from an established
    # framework, not from an abstract circle or an unspecified seed.
    ax.text(0.02, 0.44, "Growth from an established UiO-66 seed", fontweight="bold", fontsize=8.9)
    seed_nodes = {
        "a": (0.09, 0.23), "b": (0.18, 0.32), "c": (0.27, 0.23),
        "d": (0.18, 0.14), "e": (0.18, 0.23),
    }
    seed_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
    ]
    draw_framework(ax, seed_nodes, seed_edges, scale=0.72)
    ax.text(0.18, 0.055, r"reviewed seed, $N=76$", ha="center", fontsize=7.0)
    ax.add_patch(FancyArrowPatch((0.32, 0.23), (0.45, 0.23), arrowstyle="-|>", mutation_scale=9, lw=1.05, color="#66717A"))
    ax.text(0.385, 0.275, r"$P_{76\rightarrow126}$", ha="center", fontsize=7.2, color=COLORS["growth_probability"])

    grown_nodes = {
        "a": (0.52, 0.23), "b": (0.61, 0.32), "c": (0.70, 0.23),
        "d": (0.61, 0.14), "e": (0.61, 0.23), "f": (0.79, 0.32),
        "g": (0.88, 0.23), "h": (0.79, 0.14), "i": (0.79, 0.23),
    }
    grown_edges = [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "e"), ("b", "e"), ("c", "e"), ("d", "e"),
        ("c", "f"), ("f", "g"), ("g", "h"), ("h", "c"),
        ("c", "i"), ("f", "i"), ("g", "i"), ("h", "i"),
    ]
    draw_framework(ax, grown_nodes, grown_edges, new_nodes={"f", "g", "h", "i"}, scale=0.70)
    ax.text(0.70, 0.055, r"grown framework, $N=126$", ha="center", fontsize=7.0)

    # Minimal chemistry key.
    draw_zr6_node(ax, (0.91, 0.40), scale=0.55)
    ax.text(0.94, 0.40, r"Zr$_6$ node", va="center", fontsize=6.6)
    draw_bdc_linker(ax, (0.875, 0.34), (0.945, 0.34), scale=0.62)
    ax.text(0.94, 0.30, "BDC linker", ha="center", fontsize=6.6)
    ax.text(
        0.98,
        0.012,
        "schematic; glyph count not to scale",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#747D84",
    )


def anchor_comparison(ax, source: pd.DataFrame) -> None:
    subset = source[np.isclose(source.S_close, 1.0) & np.isclose(source.atomic_Zr_to_BDC_ratio, 1.0)].sort_values(
        "link_removal_to_addition_ratio"
    )
    x = subset.link_removal_to_addition_ratio.to_numpy(float)
    ax.plot(x, subset.P_2_to_30, "o-", color=COLORS["assembly"], lw=1.7, ms=4, label=r"$P_{2\rightarrow30}$")
    ax.plot(x, subset.P_76_to_126, "s-", color=COLORS["growth_probability"], lw=1.7, ms=4, label=r"$P_{76\rightarrow126}$")
    ax.set_xscale("log")
    ax.set_xlim(0.085, 36)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel(r"link-removal/addition ratio, $\rho_{\rm off}$")
    ax.set_ylabel("threshold-crossing probability")
    ax.grid(True, which="major", color=COLORS["grid"], lw=0.55)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    ax.set_title(r"Reference slice: Zr/BDC = 1, $S_{\rm close}=1$", fontweight="bold", pad=5)


def sensitivity_cloud(ax, sensitivity: pd.DataFrame, process: str, title: str, norm, cmap) -> None:
    subset = sensitivity[(sensitivity.process == process) & sensitivity.S_close.isin(SC_VALUES)].copy()
    z_lookup = {value: index for index, value in enumerate(SC_VALUES)}
    resolved = subset[subset.resolved == 1]
    unresolved = subset[subset.resolved == 0]
    ax.scatter(
        np.log10(unresolved.rho_geometric_midpoint),
        unresolved.q_value,
        unresolved.S_close.map(z_lookup),
        c=COLORS["missing"],
        s=5,
        marker="x",
        linewidths=0.45,
        depthshade=False,
    )
    ax.scatter(
        np.log10(resolved.rho_geometric_midpoint),
        resolved.q_value,
        resolved.S_close.map(z_lookup),
        c=resolved.beta_dlnJ_dlnrho,
        cmap=cmap,
        norm=norm,
        s=10,
        marker="s",
        linewidths=0,
        alpha=0.9,
        depthshade=False,
    )
    ax.set_title(title, fontweight="bold", pad=2)
    ax.set_xlabel(r"$\rho_{\rm off}$ interval midpoint", labelpad=4)
    ax.set_ylabel("Zr/BDC", labelpad=4)
    ax.set_zlabel(r"sampled $S_{\rm close}$", labelpad=3)
    xticks = np.log10([0.1, 1, 10, 100, 1000])
    ax.set_xticks(xticks, ["0.1", "1", "10", "100", "1000"])
    ax.set_yticks([0.1, 1, 3, 10], ["0.1", "1", "3", "10"])
    ax.set_zticks(np.arange(len(SC_VALUES)), [f"{value:g}" for value in SC_VALUES])
    ax.set_xlim(np.log10(0.01), np.log10(1500))
    ax.set_ylim(0.1, 10)
    ax.set_zlim(-0.15, len(SC_VALUES) - 0.85)
    ax.set_box_aspect((1.45, 1.12, 0.88))
    clean_3d_axes(ax)


def sensitivity_matrix(
    ax,
    sensitivity: pd.DataFrame,
    process: str,
    title: str,
    norm,
    cmap,
    *,
    show_y_labels: bool,
) -> mpl.image.AxesImage:
    subset = sensitivity[sensitivity.process == process].copy()
    intervals = [f"{RHO_VALUES[i]:g}-{RHO_VALUES[i + 1]:g}" for i in range(len(RHO_VALUES) - 1)]
    values = np.full((len(SC_VALUES) * len(Q_VALUES), len(intervals)), np.nan)
    for sc_index, sc_value in enumerate(SC_VALUES):
        for q_index, q_value in enumerate(Q_VALUES):
            row_index = sc_index * len(Q_VALUES) + q_index
            rows = subset[
                np.isclose(subset.S_close, sc_value)
                & np.isclose(subset.q_value, q_value)
            ].copy()
            for interval_index in range(len(intervals)):
                hit = rows[
                    np.isclose(rows.rho_low, RHO_VALUES[interval_index])
                    & np.isclose(rows.rho_high, RHO_VALUES[interval_index + 1])
                ]
                if len(hit) == 1 and int(hit.iloc[0].resolved) == 1:
                    values[row_index, interval_index] = float(hit.iloc[0].beta_dlnJ_dlnrho)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, origin="upper", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xticks(range(len(intervals)), intervals, rotation=25, ha="right")
    y_positions = np.arange(len(SC_VALUES) * len(Q_VALUES))
    y_labels = []
    for sc_value in SC_VALUES:
        for q_index, q_value in enumerate(Q_VALUES):
            y_labels.append(
                (rf"$S_{{close}}={sc_value:g}$   {q_value:g}" if q_index == 0 else f"{q_value:g}")
                if show_y_labels
                else ""
            )
    ax.set_yticks(y_positions, y_labels)
    ax.set_xlabel(r"adjacent $\rho_{\rm off}$ interval")
    if show_y_labels:
        ax.set_ylabel(r"sampled $S_{close}$ block and Zr/BDC ratio")
    for sc_index, sc_value in enumerate(SC_VALUES):
        start = sc_index * len(Q_VALUES)
        center = start + (len(Q_VALUES) - 1) / 2
        if sc_index:
            ax.axhline(start - 0.5, color="white", lw=1.6)
    ax.tick_params(axis="y", labelsize=6.4, pad=2)
    ax.tick_params(axis="x", labelsize=7.0, pad=1)
    for spine in ax.spines.values():
        spine.set_color("#AEB6BC")
    return image


def sensitivity_profiles(ax, sensitivity: pd.DataFrame, process: str, title: str, interval_colors: list) -> None:
    """Show beta(q) at every sampled S_close and rho interval.

    Each horizontal S_close plane contains five colored beta-versus-q curves,
    one for every adjacent rho interval. Missing segments are unresolved because
    at least one endpoint contained zero successful events.
    """
    subset = sensitivity[sensitivity.process == process].copy()
    for sc_index, sc_value in enumerate(SC_VALUES):
        sc_rows = subset[np.isclose(subset.S_close, sc_value)]
        for interval_index in range(len(RHO_VALUES) - 1):
            rows = sc_rows[
                np.isclose(sc_rows.rho_low, RHO_VALUES[interval_index])
                & np.isclose(sc_rows.rho_high, RHO_VALUES[interval_index + 1])
            ].set_index("q_value").reindex(Q_VALUES)
            beta = rows.beta_dlnJ_dlnrho.to_numpy(float)
            resolved = rows.resolved.fillna(0).to_numpy(int) == 1
            beta[~resolved] = np.nan
            beta_display = np.maximum(beta, -3.0)
            ax.plot(
                Q_VALUES,
                np.full(len(Q_VALUES), sc_index, dtype=float),
                beta_display,
                color=interval_colors[interval_index],
                lw=1.35,
                marker="o",
                ms=2.8,
                mfc=interval_colors[interval_index],
                mec="white",
                mew=0.35,
            )
    ax.set_title(title, fontweight="bold", pad=3)
    ax.set_xlabel("Zr/BDC ratio", labelpad=5)
    ax.set_ylabel(r"sampled $S_{close}$", labelpad=5)
    ax.set_zlabel(r"$\beta=\Delta\ln J/\Delta\ln\rho_{off}$", labelpad=5)
    ax.set_xticks([0.1, 1, 3, 10], ["0.1", "1", "3", "10"])
    ax.set_yticks(np.arange(len(SC_VALUES)), [f"{value:g}" for value in SC_VALUES])
    ax.set_zticks([-3, -2, -1, 0], [r"$\leq-3$", "-2", "-1", "0"])
    ax.set_xlim(0.1, 10)
    ax.set_ylim(-0.2, len(SC_VALUES) - 0.8)
    ax.set_zlim(-3.0, 0.5)
    ax.set_box_aspect((1.45, 1.0, 0.92))
    clean_3d_axes(ax)


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
    row_label: str | None = None,
) -> mpl.image.AxesImage:
    intervals = [(RHO_VALUES[index], RHO_VALUES[index + 1]) for index in range(len(RHO_VALUES) - 1)]
    matrix = np.full((len(Q_VALUES), len(intervals)), np.nan)
    subset = sensitivity[
        (sensitivity.process == process) & np.isclose(sensitivity.S_close, sc_value)
    ]
    for q_index, q_value in enumerate(Q_VALUES):
        for interval_index, (rho_low, rho_high) in enumerate(intervals):
            hit = subset[
                np.isclose(subset.q_value, q_value)
                & np.isclose(subset.rho_low, rho_low)
                & np.isclose(subset.rho_high, rho_high)
            ]
            if len(hit) == 1 and int(hit.iloc[0].resolved) == 1:
                matrix[q_index, interval_index] = max(
                    -3.0, float(hit.iloc[0].beta_dlnJ_dlnrho)
                )
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(intervals)))
    ax.set_yticks(range(len(Q_VALUES)))
    if show_x:
        ax.set_xticklabels(
            [f"{low:g}-{high:g}" for low, high in intervals],
            rotation=28,
            ha="right",
        )
    if show_y:
        ax.set_yticklabels([f"{value:g}" for value in Q_VALUES])
    if show_x:
        ax.set_xlabel("link-removal/addition ratio interval", labelpad=2)
    if show_y:
        ax.set_ylabel(
            (row_label + "\nZr/BDC") if row_label else "Zr/BDC",
            labelpad=3,
            fontsize=7.0,
            fontweight="bold" if row_label else "normal",
        )
    ax.tick_params(
        labelsize=6.4,
        length=2,
        pad=1,
        labelbottom=show_x,
        labelleft=show_y,
    )
    for spine in ax.spines.values():
        spine.set_color("#AEB6BC")
        spine.set_linewidth(0.6)
    return image


def pxrd_panel(ax_pattern, ax_similarity, response: pd.DataFrame) -> None:
    labels = [
        "reviewed_seed",
        "grown_rho0p1",
        "grown_rho0p3",
        "grown_rho1",
        "grown_rho3",
        "grown_rho10",
    ]
    rho_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    sequence_colors = [mpl.colormaps["viridis"](value) for value in np.linspace(0.08, 0.9, len(rho_values))]
    colors = ["#525F69", *sequence_colors]
    offsets = [6.15, 5.0, 4.0, 3.0, 2.0, 1.0]
    metadata = json.loads((PXRD_DIR / "pxrd_audit_metadata.json").read_text(encoding="utf-8"))
    similarity = {
        row["label"]: float(row["cosine_similarity_to_reviewed_seed"])
        for row in metadata["patterns"]
    }

    for label, color, offset in zip(labels, colors, offsets):
        data = pd.read_csv(PXRD_DIR / f"{label}_finite_debye_pxrd.csv")
        ax_pattern.plot(
            data.two_theta_deg,
            data.intensity_smoothed_normalized + offset,
            color=color,
            lw=1.05,
        )
        direct_label = (
            "reviewed seed, N = 76"
            if label == "reviewed_seed"
            else rf"$\rho_{{off}}={label.replace('grown_rho', '').replace('p', '.')}$, N = 126"
        )
        ax_pattern.text(29.7, offset + 0.24, direct_label, color=color, ha="right", fontsize=7.0)

    peaks = pd.read_csv(PXRD_DIR / "ideal_uio66_periodic_peak_positions.csv")
    baseline = 0.18
    for row in peaks.itertuples():
        ax_pattern.vlines(
            row.two_theta_deg,
            baseline,
            baseline + 0.62 * row.intensity_normalized,
            color="#444C52",
            lw=0.85,
        )
    ax_pattern.text(29.7, baseline + 0.08, "ideal periodic UiO-66", ha="right", color="#525C64", fontsize=7.0)
    ax_pattern.set_xlim(3, 30)
    ax_pattern.set_ylim(0.02, 7.18)
    ax_pattern.set_xlabel(r"$2\theta$ (degrees, Cu K$\alpha$)")
    ax_pattern.set_ylabel("normalized Debye intensity (vertically offset)")
    ax_pattern.set_yticks([])
    ax_pattern.grid(False)
    ax_pattern.set_title(
        r"Ordered PXRD series for saved growth structures ($q=1$, $S_{close}=1$)",
        fontweight="bold",
        pad=5,
    )
    ax_pattern.annotate(
        "",
        xy=(0.022, 0.17),
        xytext=(0.022, 0.72),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#68737B"},
    )
    ax_pattern.text(
        0.032,
        0.445,
        r"increasing $\rho_{off}$",
        transform=ax_pattern.transAxes,
        rotation=90,
        va="center",
        ha="left",
        fontsize=6.8,
        color="#5C6770",
    )

    rho_tokens = ["0p1", "0p3", "1", "3", "10"]
    similarity_values = np.array([similarity[f"grown_rho{token}"] for token in rho_tokens])
    ax_similarity.plot(rho_values, similarity_values, color="#90989E", lw=1.0, zorder=1)
    for rho, value, color in zip(rho_values, similarity_values, sequence_colors):
        ax_similarity.scatter(rho, value, s=30, color=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax_similarity.axhline(1.0, color="#525F69", lw=0.8, ls="--")
    ax_similarity.text(0.11, 0.999, "reviewed seed", fontsize=6.8, color="#525F69", va="top")
    no_growth = response[
        np.isclose(response.S_close, 1.0)
        & np.isclose(response.atomic_Zr_to_BDC_ratio, 1.0)
        & np.isclose(response.link_removal_to_addition_ratio, 30.0)
    ].iloc[0]
    if float(no_growth.P_76_to_126) == 0.0:
        ax_similarity.scatter([30.0], [0.943], marker="x", s=32, color="#8B9399", linewidth=1.0)
        ax_similarity.text(27.0, 0.945, "no N = 126\ntrajectory", ha="right", va="bottom", fontsize=6.6, color="#6B747B")
    ax_similarity.set_xscale("log")
    ax_similarity.set_xlim(0.085, 36)
    ax_similarity.set_ylim(0.94, 1.003)
    ax_similarity.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax_similarity.set_xlabel(r"link-removal/addition ratio, $\rho_{off}$")
    ax_similarity.set_ylabel("cosine similarity to reviewed-seed PXRD")
    ax_similarity.grid(True, axis="y", color="#DEE2E5", lw=0.55)
    ax_similarity.set_title("Whole-pattern similarity", fontweight="bold", pad=5)
    ax_similarity.text(
        0.03,
        0.035,
        "one saved structure per condition; line guides the eye",
        transform=ax_similarity.transAxes,
        fontsize=6.5,
        color="#69737B",
    )


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    method = ax.text2D if hasattr(ax, "text2D") else ax.text
    method(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def save_formats(fig, stem: Path, dpi: int = 360) -> None:
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
        )


def export_panel_bundle(response: pd.DataFrame, sensitivity: pd.DataFrame, rate_max: float) -> None:
    """Export independent editable panels for 1:1 review and manual assembly."""
    bundle = HERE / "figure3_panel_bundle_v2"
    bundle.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.75))
    definition_panel(ax)
    save_formats(fig, bundle / "figure3a_molecular_schematic")
    plt.close(fig)

    response_specs = [
        ("b", "P_2_to_30", "magma", Normalize(0, 1), r"Nucleation probability, $P_{2\rightarrow30}$", "probability"),
        ("c", "P_76_to_126", "viridis", Normalize(0, 1), r"Crystal growth probability, $P_{76\rightarrow126}$", "probability"),
        ("d", "G_76_to_126_entities_per_model_time", "cividis", Normalize(0, rate_max), r"Growth rate of successful trajectories, $G_{76\rightarrow126}$", "entities per model time"),
    ]
    for letter, value, cmap, norm, title, colorbar_label in response_specs:
        fig = plt.figure(figsize=(4.25, 3.65))
        ax = fig.add_subplot(111, projection="3d")
        stacked_response(
            ax,
            response,
            value,
            cmap_name=cmap,
            norm=norm,
            title=title,
            colorbar_label=colorbar_label,
        )
        save_formats(fig, bundle / f"figure3{letter}_response_surface")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.65, 3.0))
    anchor_comparison(ax, response)
    save_formats(fig, bundle / "figure3e_reference_comparison")
    plt.close(fig)

    sc_display = [0.1, 1.0, 2.0]
    processes = [
        ("assembly_J_1_to_30", "Nucleation"),
        ("seed_growth_J_76_to_126", "Crystal growth"),
    ]
    sensitivity_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    sensitivity_cmap = mpl.colormaps["RdBu_r"].copy()
    sensitivity_cmap.set_bad(COLORS["missing"])
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.45), sharex=True, sharey=True)
    for row_index, (process, row_label) in enumerate(processes):
        for col_index, sc_value in enumerate(sc_display):
            sensitivity_heatmap(
                axes[row_index, col_index],
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
                axes[row_index, col_index].set_title(rf"$S_{{close}}={sc_value:g}$", fontweight="bold")
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=sensitivity_norm, cmap=sensitivity_cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.16,
        aspect=45,
        extend="min",
    )
    colorbar.set_label(r"$\beta=\Delta\ln J/\Delta\ln\rho_{off}$")
    fig.text(
        0.5,
        0.012,
        r"gray: unresolved zero-event interval; values below $-3$ are clipped only for display",
        fontsize=6.5,
        color="#69737B",
        ha="center",
    )
    save_formats(fig, bundle / "figure3f_local_sensitivity")
    plt.close(fig)

    fig = plt.figure(figsize=(10.8, 3.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.0], wspace=0.19)
    ax_pattern = fig.add_subplot(grid[0, 0])
    ax_similarity = fig.add_subplot(grid[0, 1])
    pxrd_panel(ax_pattern, ax_similarity, response)
    save_formats(fig, bundle / "figure3g_pxrd_series")
    plt.close(fig)

    pd.DataFrame(
        [
            {"panel": "A", "role": "molecular and operational schematic", "evidence_status": "illustrative proposed schematic"},
            {"panel": "B", "role": "operational nucleation probability", "evidence_status": "calculated"},
            {"panel": "C", "role": "seeded-growth probability", "evidence_status": "calculated"},
            {"panel": "D", "role": "conditional crystal-growth rate", "evidence_status": "calculated"},
            {"panel": "E", "role": "reference-slice comparison", "evidence_status": "calculated"},
            {"panel": "F", "role": "local sensitivity", "evidence_status": "calculated"},
            {"panel": "G", "role": "finite-cluster PXRD validation", "evidence_status": "calculated; one saved structure per condition"},
        ]
    ).to_csv(bundle / "panel_source_manifest_v2.csv", index=False)


def main() -> None:
    setup_style()
    committor = pd.read_csv(COMMITTOR_PATH)
    seed_summary = combined_seeded_summary()
    seed_traces = combined_seeded_traces()
    response = response_source_table(committor, seed_summary)
    response.to_csv(HERE / "figure3_response_surface_source_v1.csv", index=False)

    assembly_summary = pd.read_csv(ASSEMBLY_SUMMARY_PATH)
    assembly_rates_native, seed_rates = condition_rates(assembly_summary, seed_traces)
    assembly_rates = matched_assembly_rates(assembly_rates_native)
    sensitivity = local_sensitivity(pd.concat([assembly_rates, seed_rates], ignore_index=True))
    sensitivity.to_csv(HERE / "figure3_local_sensitivity_source_v1.csv", index=False)

    fig = plt.figure(figsize=(15.4, 10.9), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        left=0.045,
        right=0.985,
        bottom=0.06,
        top=0.975,
        wspace=0.24,
        hspace=0.52,
        height_ratios=[1.0, 0.84, 0.78],
        width_ratios=[1.0, 1.12, 1.12, 1.12],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    definition_panel(ax_a)

    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    ax_d = fig.add_subplot(gs[0, 3], projection="3d")
    stacked_response(
        ax_b,
        response,
        "P_2_to_30",
        cmap_name="magma",
        norm=Normalize(0, 1),
        title=r"Nucleation probability, $P_{2\rightarrow30}$",
        colorbar_label="probability",
    )
    stacked_response(
        ax_c,
        response,
        "P_76_to_126",
        cmap_name="viridis",
        norm=Normalize(0, 1),
        title=r"Crystal growth probability, $P_{76\rightarrow126}$",
        colorbar_label="probability",
    )
    finite_rates = response.G_76_to_126_entities_per_model_time.replace([np.inf, -np.inf], np.nan)
    rate_max = float(np.nanpercentile(finite_rates, 98))
    stacked_response(
        ax_d,
        response,
        "G_76_to_126_entities_per_model_time",
        cmap_name="cividis",
        norm=Normalize(0, rate_max),
        title=r"Growth rate of successful trajectories, $G_{76\rightarrow126}$",
        colorbar_label="entities per model time",
    )

    ax_e = fig.add_subplot(gs[1, 0])
    anchor_comparison(ax_e, response)

    f_grid = gs[1, 1:].subgridspec(2, 3, wspace=0.10, hspace=0.13)
    sc_display = [0.1, 1.0, 2.0]
    processes = [
        ("assembly_J_1_to_30", "Nucleation"),
        ("seed_growth_J_76_to_126", "Crystal growth"),
    ]
    sensitivity_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5)
    sensitivity_cmap = mpl.colormaps["RdBu_r"].copy()
    sensitivity_cmap.set_bad(COLORS["missing"])
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
                axis.set_title(rf"$S_{{close}}={sc_value:g}$", fontweight="bold", pad=3)
            f_axes.append(axis)

    g_grid = gs[2, :].subgridspec(1, 2, width_ratios=[3.15, 1.0], wspace=0.19)
    ax_g_pattern = fig.add_subplot(g_grid[0, 0])
    ax_g_similarity = fig.add_subplot(g_grid[0, 1])
    pxrd_panel(ax_g_pattern, ax_g_similarity, response)

    for axis, label, x in [
        (ax_a, "A", -0.03),
        (ax_b, "B", -0.10),
        (ax_c, "C", -0.10),
        (ax_d, "D", -0.10),
        (ax_e, "E", -0.055),
        (f_axes[0], "F", -0.16),
        (ax_g_pattern, "G", -0.035),
    ]:
        panel_label(axis, label, x=x)

    fig.canvas.draw()
    f_left = min(axis.get_position().x0 for axis in f_axes)
    f_right = max(axis.get_position().x1 for axis in f_axes)
    f_bottom = min(axis.get_position().y0 for axis in f_axes)
    bar_width = 0.43 * (f_right - f_left)
    bar_left = (f_left + f_right - bar_width) / 2
    bar_bottom = f_bottom - 0.058
    colorbar_axis = fig.add_axes([bar_left, bar_bottom, bar_width, 0.011])
    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=sensitivity_cmap,
        norm=sensitivity_norm,
        orientation="horizontal",
        extend="min",
    )
    colorbar.set_label(r"$\beta=\Delta\ln J/\Delta\ln\rho_{off}$", labelpad=1)
    colorbar.ax.tick_params(labelsize=6.6, pad=1)
    fig.text(
        (f_left + f_right) / 2,
        bar_bottom - 0.034,
        r"gray: unresolved zero-event interval; values below $-3$ are clipped only for display",
        fontsize=6.5,
        color="#69737B",
        ha="center",
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            HERE / f"figure3_prototype_v2.{extension}",
            dpi=360 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    export_panel_bundle(response, sensitivity, rate_max)
    print(HERE / "figure3_prototype_v2.png")


if __name__ == "__main__":
    main()
