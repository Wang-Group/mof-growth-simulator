"""Build Figure 5: surface-domain formation versus established UiO-66 growth."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "figure_rebuild_20260901"
FILES = {
    "mismatch": SOURCE / "figure5_surface_vs_seeded_growth_sc1_v1_mismatched_surface.csv",
    "matched": SOURCE / "figure5_surface_vs_seeded_growth_sc1_v1_matched_surface_control.csv",
    "seed": SOURCE / "figure5_surface_vs_seeded_growth_sc1_v1_seeded.csv",
}
COLORS = {"mismatch": "#C65A3A", "matched": "#7656A3", "seed": "#2F78A8"}
LABELS = {
    "mismatch": "Separate domain on\nmismatched surface",
    "matched": "Growth on matched\nUiO-66 surface",
    "seed": "Growth of established\nUiO-66 seed",
}
MARKERS = {"mismatch": "o", "matched": "^", "seed": "s"}


def load_style() -> None:
    kernel = Path.home() / ".codex" / "skills" / "figure-style" / "kernel.py"
    spec = importlib.util.spec_from_file_location("figure_style_kernel", kernel)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.apply_figure_style(frame="open", font="Arial", sizes=(8.5, 7.2, 6.7))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def load_data() -> pd.DataFrame:
    frames = []
    for process, path in FILES.items():
        data = pd.read_csv(path)
        data["process"] = process
        data["n"] = data.get("replicates", 32).astype(int) if "replicates" in data else 32
        data["successes"] = data["events"].astype(int)
        if process == "seed":
            data["outcome_fraction"] = data["probability"]
        else:
            data["outcome_fraction"] = data["trace_success_fraction"]
        ci = [wilson(int(k), int(n)) for k, n in zip(data.successes, data.n)]
        data["outcome_low"] = [x[0] for x in ci]
        data["outcome_high"] = [x[1] for x in ci]
        baselines = data[np.isclose(data.rho, 0.1)].set_index("q")["rate"]
        data["baseline_rate"] = data.q.map(baselines)
        data["rate_retained"] = data.rate / data.baseline_rate
        data["rate_retained_low"] = data.rate_low / data.baseline_rate
        data["rate_retained_high"] = data.rate_high / data.baseline_rate
        frames.append(data)
    combined = pd.concat(frames, ignore_index=True)
    keep = [
        "process", "q", "rho", "n", "successes", "exposure", "rate", "rate_low",
        "rate_high", "baseline_rate", "rate_retained", "rate_retained_low",
        "rate_retained_high", "outcome_fraction", "outcome_low", "outcome_high",
    ]
    combined[keep].to_csv(HERE / "figure5_source_data_v2.csv", index=False)
    return combined


def node(ax, xy, color="#477B9E", size=0.040, z=3):
    ax.add_patch(patches.Circle(xy, size, facecolor=color, edgecolor="white", lw=0.9, zorder=z))


def linker(ax, p1, p2, color="#7895A8", lw=2.2, z=2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=z)


def square_lattice(ax, x0, y0, dx=0.115, dy=0.115, rows=2, cols=4, alpha=1.0):
    pts = {}
    for r in range(rows):
        for c in range(cols):
            pts[(r, c)] = (x0 + c * dx, y0 + r * dy)
    for r in range(rows):
        for c in range(cols - 1):
            linker(ax, pts[(r, c)], pts[(r, c + 1)], color="#82A5BA", lw=2.0, z=1)
    for r in range(rows - 1):
        for c in range(cols):
            linker(ax, pts[(r, c)], pts[(r + 1, c)], color="#82A5BA", lw=2.0, z=1)
    for p in pts.values():
        node(ax, p, color="#39779D")


def triangular_lattice(ax, x0, y0, scale=0.125):
    pts = [(x0, y0), (x0 + scale, y0), (x0 + 2 * scale, y0),
           (x0 + 0.5 * scale, y0 + 0.105), (x0 + 1.5 * scale, y0 + 0.105)]
    edges = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]
    for i, j in edges:
        linker(ax, pts[i], pts[j], color="#C3A16F", lw=2.0, z=1)
    for p in pts:
        node(ax, p, color="#A77C3D")


def daughter_cluster(ax, centre, scale=1.0):
    cx, cy = centre
    pts = [(cx - 0.075 * scale, cy), (cx + 0.075 * scale, cy),
           (cx, cy + 0.095 * scale), (cx, cy - 0.095 * scale)]
    for i, j in [(0, 2), (2, 1), (1, 3), (3, 0)]:
        linker(ax, pts[i], pts[j], color="#82A5BA", lw=2.0, z=4)
    for p in pts:
        node(ax, p, color="#39779D", size=0.035 * scale, z=5)
    ax.add_patch(patches.Ellipse((cx, cy), 0.26 * scale, 0.29 * scale,
                                fill=False, edgecolor=COLORS["mismatch"], lw=1.2,
                                ls=(0, (3, 2)), zorder=3))


def panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Matched growth versus mismatched surface nucleation", pad=5)
    ax.text(0.5, 0.93, "Same incoming UiO-66 building units", ha="center", va="center",
            fontsize=8.2, color="#263746")
    node(ax, (0.45, 0.84), color="#39779D", size=0.025)
    ax.plot([0.50, 0.56], [0.84, 0.84], color="#7C929F", lw=2.4, solid_capstyle="round")
    ax.scatter([0.50, 0.56], [0.84, 0.84], s=18, color="#D18A67", zorder=4, edgecolors="white", linewidths=0.5)
    ax.text(0.45, 0.79, r"Zr$_6$", ha="center", va="top", fontsize=6.3, color="#4E626F")
    ax.text(0.53, 0.79, "BDC", ha="center", va="top", fontsize=6.3, color="#4E626F")
    ax.annotate("", xy=(0.28, 0.70), xytext=(0.46, 0.81),
                arrowprops=dict(arrowstyle="-|>", color="#6C7882", lw=1.1))
    ax.annotate("", xy=(0.72, 0.70), xytext=(0.54, 0.81),
                arrowprops=dict(arrowstyle="-|>", color="#6C7882", lw=1.1))
    left = patches.FancyBboxPatch((0.03, 0.08), 0.44, 0.62,
                                  boxstyle="round,pad=0.015,rounding_size=0.025",
                                  facecolor="#EEF4F8", edgecolor="#B8CBD7", lw=0.9)
    right = patches.FancyBboxPatch((0.53, 0.08), 0.44, 0.62,
                                   boxstyle="round,pad=0.015,rounding_size=0.025",
                                   facecolor="#FBF2EB", edgecolor="#DDC4AE", lw=0.9)
    ax.add_patch(left); ax.add_patch(right)
    ax.text(0.25, 0.65, "Matched surface", ha="center", weight="bold", color="#334956")
    ax.text(0.25, 0.605, "UiO-66", ha="center", fontsize=7.2, color="#334956")
    square_lattice(ax, 0.08, 0.18, dx=0.105, dy=0.11, rows=3, cols=4)
    # added units share the lattice direction
    square_lattice(ax, 0.185, 0.40, dx=0.105, dy=0.11, rows=2, cols=3)
    ax.text(0.25, 0.115, "Continued UiO-66 lattice", ha="center", color=COLORS["matched"], weight="bold", fontsize=7.2)
    ax.text(0.75, 0.65, "Mismatched surface", ha="center", weight="bold", color="#5C4935")
    ax.text(0.75, 0.605, "MOF-808", ha="center", fontsize=7.2, color="#5C4935")
    triangular_lattice(ax, 0.58, 0.18, scale=0.14)
    daughter_cluster(ax, (0.77, 0.45), scale=0.9)
    ax.plot([0.71, 0.69], [0.37, 0.29], color="#A9A9A9", lw=1.2, ls=(0, (2, 2)))
    ax.text(0.75, 0.115, "Separate UiO-66 nucleus", ha="center", color=COLORS["mismatch"], weight="bold", fontsize=7.2)
    ax.text(0.99, 0.015, "conceptual schematic", ha="right", va="bottom", fontsize=6.3, color="#777777")


def process_rows(data, q=1.0):
    return {p: data[(data.process == p) & np.isclose(data.q, q)].sort_values("rho")
            for p in ("mismatch", "matched", "seed")}


def panel_b(ax, data):
    for process, sub in process_rows(data).items():
        positive = sub.rate_retained > 0
        ax.errorbar(sub.loc[positive, "rho"], sub.loc[positive, "rate_retained"],
                    yerr=np.vstack([sub.loc[positive, "rate_retained"] - sub.loc[positive, "rate_retained_low"],
                                    sub.loc[positive, "rate_retained_high"] - sub.loc[positive, "rate_retained"]]),
                    color=COLORS[process], marker=MARKERS[process], ms=4.7, lw=1.5,
                    capsize=2.2, label=LABELS[process].replace("\n", " "))
        zero = ~positive
        if zero.any():
            ax.scatter(sub.loc[zero, "rho"], sub.loc[zero, "rate_retained_high"],
                       facecolors="white", edgecolors=COLORS[process], marker="v", s=38, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xlabel(r"Link-removal/addition ratio, $\rho_{\mathrm{off}}$")
    ax.set_ylabel(r"Rate retained relative to $\rho_{\mathrm{off}}=0.1$")
    ax.set_title("Rate suppression at Zr/BDC = 1")
    ax.axhline(1, color="#B8B8B8", lw=0.7, ls="--", zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="v", markerfacecolor="white", markeredgecolor="#65737E",
                          linestyle="none", markersize=5))
    labels.append("Zero events: 95% upper limit")
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.02), handlelength=1.6)
    ax.axvline(10, color="#C7CDD2", lw=0.7, ls=(0, (2, 2)), zorder=0)
    at_ten = {p: float(s[np.isclose(s.rho, 10)].rate_retained.iloc[0])
              for p, s in process_rows(data).items()}
    ax.text(10.8, at_ten["matched"] * 1.18, f"{100*at_ten['matched']:.1f}%",
            color=COLORS["matched"], fontsize=6.4, va="bottom")
    ax.text(10.8, at_ten["seed"] * 0.88, f"{100*at_ten['seed']:.1f}%",
            color=COLORS["seed"], fontsize=6.4, va="top")
    ax.text(8.9, at_ten["mismatch"] * 0.72, f"{100*at_ten['mismatch']:.3f}%",
            color=COLORS["mismatch"], fontsize=6.4, va="top", ha="right")


def panel_c(ax, data):
    offsets = {"mismatch": -0.06, "matched": 0.0, "seed": 0.06}
    for process, sub in process_rows(data).items():
        x = np.log10(sub.rho) + offsets[process]
        y = 100 * sub.outcome_fraction
        low = 100 * sub.outcome_low
        high = 100 * sub.outcome_high
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), color=COLORS[process],
                    marker=MARKERS[process], ms=4.7, lw=1.4, capsize=2.2)
    ticks = np.array([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticks(np.log10(ticks), ["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xlim(np.log10(0.075), np.log10(41)); ax.set_ylim(-5, 108)
    ax.set_xlabel(r"Link-removal/addition ratio, $\rho_{\mathrm{off}}$")
    ax.set_ylabel("Trajectories reaching endpoint (%)")
    ax.set_title("Fraction of trajectories reaching each endpoint")
    ax.axvline(np.log10(10), color="#C7CDD2", lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.text(np.log10(11), 12, "mismatched: 1/32", ha="left", color=COLORS["mismatch"], weight="bold")
    ax.text(np.log10(11), 77, "matched: 32/32\nseed: 32/32", ha="left", va="center",
            color="#4F5961", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88))
    ax.text(0.02, 0.05, "surface endpoint: attached 30-entity cyclic domain\nseed endpoint: growth from 76 to 126 entities",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color="#5E666C")


def panel_d(ax, data):
    shared_q = [0.5, 1.0, 1.5, 3.0]
    for process in ("mismatch", "matched", "seed"):
        sub = data[(data.process == process) & np.isclose(data.rho, 10) & data.q.isin(shared_q)].sort_values("q")
        positive = sub.rate_retained > 0
        ax.errorbar(sub.loc[positive, "q"], sub.loc[positive, "rate_retained"],
                    yerr=np.vstack([sub.loc[positive, "rate_retained"] - sub.loc[positive, "rate_retained_low"],
                                    sub.loc[positive, "rate_retained_high"] - sub.loc[positive, "rate_retained"]]),
                    color=COLORS[process], marker=MARKERS[process], ms=4.7, lw=1.5, capsize=2.0)
        zero = ~positive
        if zero.any():
            ax.scatter(sub.loc[zero, "q"], sub.loc[zero, "rate_retained_high"],
                       facecolors="white", edgecolors=COLORS[process], marker="v", s=38, zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(shared_q, ["0.5", "1", "1.5", "3"])
    ax.set_xlabel("Zr/BDC ratio")
    ax.set_ylabel(r"Rate at $\rho_{\mathrm{off}}=10$ / rate at 0.1")
    ax.set_title("Suppression at high removal/addition ratio")
    ax.set_xlim(0.38, 4.15)
    for process, dy in [("matched", 1.06), ("seed", 0.93), ("mismatch", 1.08)]:
        sub = data[(data.process == process) & np.isclose(data.rho, 10) & np.isclose(data.q, 3)]
        if not sub.empty and float(sub.rate_retained.iloc[0]) > 0:
            label = {"matched": "matched-surface growth", "seed": "established-seed growth",
                     "mismatch": "mismatched-surface nucleus"}[process]
            ax.text(3.08, float(sub.rate_retained.iloc[0]) * dy, label,
                    color=COLORS[process], fontsize=6.2, va="center")


def label_panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold", clip_on=False)


def save_panel(draw, letter, data):
    fig, ax = plt.subplots(figsize=(4.6, 3.55))
    draw(ax, data) if data is not None else draw(ax)
    label_panel(ax, letter)
    fig.savefig(HERE / f"figure5_panel_{letter}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / f"figure5_panel_{letter}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    load_style()
    data = load_data()
    fig = plt.figure(figsize=(7.48, 7.65))  # 190 mm wide
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.39, wspace=0.34,
                          left=0.085, right=0.98, top=0.965, bottom=0.085)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panel_a(axes[0]); panel_b(axes[1], data); panel_c(axes[2], data); panel_d(axes[3], data)
    for letter, ax in zip("ABCD", axes):
        label_panel(ax, letter)
    fig.savefig(HERE / "figure5_surface_nucleation_vs_growth_v2.png", dpi=450, facecolor="white")
    fig.savefig(HERE / "figure5_surface_nucleation_vs_growth_v2.svg", facecolor="white")
    plt.close(fig)
    for draw, letter, datum in [(panel_a, "A", None), (panel_b, "B", data),
                                (panel_c, "C", data), (panel_d, "D", data)]:
        save_panel(draw, letter, datum)


if __name__ == "__main__":
    main()
