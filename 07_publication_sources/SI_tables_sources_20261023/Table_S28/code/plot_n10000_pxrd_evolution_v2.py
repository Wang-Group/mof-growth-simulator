#!/usr/bin/env python3
"""Plot the completed UiO-66 N=10000 PXRD campaign, including censored rho=30."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import peak_widths


FIGURE_STYLE_KERNEL = Path(r"C:\Users\Cheng Wang\.codex\skills\figure-style\kernel.py")
NEUTRAL = "#69777F"


def load_figure_style():
    spec = importlib.util.spec_from_file_location("codex_figure_style", FIGURE_STYLE_KERNEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure-style kernel: {FIGURE_STYLE_KERNEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pxrd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def first_peak_metrics(pattern: pd.DataFrame) -> tuple[float, float]:
    theta = pattern.two_theta_deg.to_numpy(float)
    intensity = pattern.intensity_smoothed_normalized.to_numpy(float)
    indices = np.flatnonzero((theta >= 5.8) & (theta <= 8.05))
    peak_index = int(indices[np.argmax(intensity[indices])])
    fwhm = float(
        peak_widths(intensity, [peak_index], rel_height=0.5)[0][0]
        * np.median(np.diff(theta))
    )
    return float(theta[peak_index]), fwhm


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=360 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.05,
        )


def size_label(size: int) -> str:
    if size < 1000:
        return str(size)
    value = size / 1000.0
    return f"{value:g}k"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style = load_figure_style()
    style.apply_figure_style(frame="open", font="Arial", sizes=(9, 8, 7), grid=False)

    label_pattern = re.compile(r"rho(?P<rho>.+)_N(?P<size>\d+)_finite_debye_pxrd\.csv$")
    patterns: dict[tuple[float, int], pd.DataFrame] = {}
    metrics = []
    for path in sorted(args.pxrd_dir.glob("*_finite_debye_pxrd.csv")):
        match = label_pattern.match(path.name)
        if match is None:
            continue
        rho = float(match.group("rho").replace("p", "."))
        size = int(match.group("size"))
        pattern = pd.read_csv(path)
        patterns[(rho, size)] = pattern
        position, fwhm = first_peak_metrics(pattern)
        metrics.append(
            {
                "fixed_per_bond_removal_coefficient": rho,
                "entities": size,
                "first_peak_position_deg": position,
                "first_peak_fwhm_deg": fwhm,
            }
        )
    if not patterns:
        raise RuntimeError(f"no size-resolved patterns found in {args.pxrd_dir}")
    metric_frame = pd.DataFrame(metrics).sort_values(
        ["fixed_per_bond_removal_coefficient", "entities"]
    )

    slope_rows = []
    for rho, group in metric_frame.groupby("fixed_per_bond_removal_coefficient"):
        if len(group) < 2:
            continue
        slope, intercept = np.polyfit(
            np.log(group.entities.to_numpy(float)),
            np.log(group.first_peak_fwhm_deg.to_numpy(float)),
            1,
        )
        slope_rows.append(
            {
                "fixed_per_bond_removal_coefficient": rho,
                "log_fwhm_vs_log_N_slope": slope,
                "log_intercept": intercept,
                "n_sizes": len(group),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    metric_frame.to_csv(args.output_dir / "N10000_pxrd_size_metrics.csv", index=False)
    slopes.to_csv(args.output_dir / "N10000_pxrd_size_scaling_fits.csv", index=False)

    periodic = pd.read_csv(args.pxrd_dir / "ideal_uio66_periodic_peak_positions.csv")
    ideal_111 = float(periodic.sort_values("two_theta_deg").iloc[0].two_theta_deg)
    rhos = sorted(metric_frame.fixed_per_bond_removal_coefficient.unique())
    sizes = sorted(metric_frame.entities.unique())
    cmap = mpl.colormaps["viridis"]
    colors = {rho: cmap(value) for rho, value in zip(rhos, np.linspace(0.08, 0.9, len(rhos)))}

    fig = plt.figure(figsize=(12.8, 9.25), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.35, 1.0),
        left=0.065,
        right=0.965,
        bottom=0.075,
        top=0.955,
        hspace=0.33,
    )

    heat_grid = outer[0].subgridspec(2, 3, wspace=0.13, hspace=0.27)
    heat_axes = []
    image = None
    for index, rho in enumerate(rhos):
        ax = fig.add_subplot(heat_grid[index // 3, index % 3])
        heat_axes.append(ax)
        available = sorted(size for candidate_rho, size in patterns if candidate_rho == rho)
        matrix = np.stack(
            [patterns[(rho, size)].intensity_smoothed_normalized.to_numpy(float) for size in available]
        )
        theta = patterns[(rho, available[0])].two_theta_deg.to_numpy(float)
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=(theta.min(), theta.max(), -0.5, len(available) - 0.5),
        )
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
        style.set_frame(ax, "boxed")
    if image is None:
        raise RuntimeError("no heatmap was created")
    heat_top = max(ax.get_position().y1 for ax in heat_axes)
    heat_bottom = min(ax.get_position().y0 for ax in heat_axes)
    heat_right = max(ax.get_position().x1 for ax in heat_axes)
    cax = fig.add_axes([heat_right + 0.008, heat_bottom, 0.010, heat_top - heat_bottom])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("normalized Debye intensity")

    bottom = outer[1].subgridspec(1, 3, width_ratios=(2.25, 1.0, 1.0), wspace=0.30)
    ax_stack = fig.add_subplot(bottom[0, 0])
    ax_width = fig.add_subplot(bottom[0, 1])
    ax_position = fig.add_subplot(bottom[0, 2])

    display_rows: list[tuple[float, int, bool]] = []
    for rho in rhos:
        available = sorted(size for candidate_rho, size in patterns if np.isclose(candidate_rho, rho))
        if not available:
            continue
        size = 10000 if 10000 in available else max(available)
        display_rows.append((rho, size, size < 10000))
    offsets = np.arange(len(display_rows), 0, -1, dtype=float)
    for offset, (rho, size, censored) in zip(offsets, display_rows):
        pattern = patterns[(rho, size)]
        ax_stack.plot(
            pattern.two_theta_deg,
            pattern.intensity_smoothed_normalized + offset,
            color=NEUTRAL if censored else colors[rho],
            lw=1.05 if not censored else 0.95,
            ls="--" if censored else "-",
        )
        label = rf"$\rho_{{\mathrm{{off}}}}={rho:g}$, N={size_label(size)}"
        if censored:
            label += " (censored)"
        ax_stack.text(
            29.7,
            offset + 0.18,
            label,
            ha="right",
            color=NEUTRAL if censored else colors[rho],
        )
    baseline = 0.12
    for peak in periodic.itertuples(index=False):
        ax_stack.vlines(
            float(peak.two_theta_deg),
            baseline,
            baseline + 0.65 * float(peak.intensity_normalized),
            color="#41494F",
            lw=0.8,
        )
    ax_stack.text(29.7, baseline + 0.07, "periodic UiO-66", ha="right", color="#41494F")
    ax_stack.set_xlim(3, 30)
    ax_stack.set_ylim(0, len(display_rows) + 1.05)
    ax_stack.set_yticks([])
    ax_stack.set_xlabel(r"$2\theta$ (degrees, Cu K$\alpha$)")
    ax_stack.set_ylabel("normalized intensity (vertically offset)")
    ax_stack.set_title("Completed N=10k products retain the UiO-66 peak sequence", loc="left")
    style.set_frame(ax_stack, "open")

    for rho, group in metric_frame.groupby("fixed_per_bond_removal_coefficient"):
        group = group.sort_values("entities")
        is_censored = np.isclose(float(rho), 30.0)
        ax_width.plot(
            group.entities,
            group.first_peak_fwhm_deg,
            marker="o",
            ms=4,
            lw=1.1,
            color=NEUTRAL if is_censored else colors[float(rho)],
            ls="--" if is_censored else "-",
            markerfacecolor="white" if is_censored else colors[float(rho)],
            label=f"{rho:g}" + (" censored" if is_censored else ""),
        )
        ax_position.plot(
            group.entities,
            group.first_peak_position_deg,
            marker="o",
            ms=4,
            lw=1.1,
            color=NEUTRAL if is_censored else colors[float(rho)],
            ls="--" if is_censored else "-",
            markerfacecolor="white" if is_censored else colors[float(rho)],
        )
    if 1000 in sizes:
        anchor = float(metric_frame[metric_frame.entities == 1000].first_peak_fwhm_deg.median())
        guide_sizes = np.asarray([1000, 10000], dtype=float)
        ax_width.plot(
            guide_sizes,
            anchor * (guide_sizes / 1000.0) ** (-1.0 / 3.0),
            color="#667079",
            lw=1.0,
            ls="--",
            label=r"$N^{-1/3}$ guide",
        )
    ax_width.set_xscale("log")
    ax_width.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    ax_width.set_xlabel("aggregate size, N")
    ax_width.set_ylabel(r"first-peak FWHM ($^\circ$)")
    ax_width.set_title("The low-angle peak narrows with size", loc="left")
    ax_width.legend(title=r"fixed $\rho_{\mathrm{off}}$", ncol=2, loc="upper right", columnspacing=0.7, handlelength=1.4)
    ax_width.grid(True, axis="y", color="#DDE1E4", lw=0.5)
    style.set_frame(ax_width, "open")

    ax_position.axhline(ideal_111, color="#41494F", lw=0.9, ls=":")
    ax_position.text(9300, ideal_111 - 0.003, "periodic (111)", ha="right", va="top")
    ax_position.set_xscale("log")
    ax_position.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    ax_position.set_xlabel("aggregate size, N")
    ax_position.set_ylabel(r"first-peak position ($^\circ$)")
    ax_position.set_title("A systematic peak offset remains", loc="left")
    ax_position.grid(True, axis="y", color="#DDE1E4", lw=0.5)
    style.set_frame(ax_position, "open")

    style.panel_letter(heat_axes[0], "A", dx=-0.20, dy=1.04, case="upper", fontsize=12)
    style.panel_letter(ax_stack, "B", dx=-0.08, dy=1.03, case="upper", fontsize=12)
    style.panel_letter(ax_width, "C", dx=-0.20, dy=1.03, case="upper", fontsize=12)
    style.panel_letter(ax_position, "D", dx=-0.20, dy=1.03, case="upper", fontsize=12)

    fig.text(
        0.065,
        0.018,
        r"$\rho_{\mathrm{off}}$ is the fixed per-bond removal coefficient; the instantaneous total removal/addition propensity ratio changes as bond and free-site counts evolve.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#515C63",
    )

    save_figure(fig, args.output_dir, "n10000_pxrd_size_evolution_v2")
    plt.close(fig)
    print(metric_frame.to_string(index=False))
    if not slopes.empty:
        print("\nScaling fits:\n" + slopes.to_string(index=False))


if __name__ == "__main__":
    main()
