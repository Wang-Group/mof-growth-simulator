"""Rebuild Figure 6b-c from the current BTB molecular KMC grid.

Panel b uses direct variance-based sensitivity on the balanced full-factorial
grid.  Panel c shows the calculated stochastic formation volume and an
interpolated P(persistent assembly)=0.5 boundary.  The surface is a kinetic
formation boundary, not a thermodynamic phase boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / "common_response_surface_v2_final_condition_summary.csv"
TRACE_SOURCE = CAMPAIGN / "common_response_surface_v2_final_trace_metrics.csv"
OUTPUT_PREFIX = HERE / "figure6bc_current_parameter_phase_v1"
BOOTSTRAP_DRAWS = 2000
RNG_SEED = 20260901


def build_probability_cube(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sc = np.sort(frame.SC.unique().astype(float))
    q = np.sort(frame.q.unique().astype(float))
    rho = np.sort(frame.off_weight.unique().astype(float))
    indexed = frame.set_index(["SC", "q", "off_weight"])
    cube = np.empty((len(sc), len(q), len(rho)), dtype=float)
    n = np.empty_like(cube)
    successes = np.empty_like(cube)
    for iz, z in enumerate(sc):
        for iy, y in enumerate(q):
            for ix, x in enumerate(rho):
                row = indexed.loc[(z, y, x)]
                cube[iz, iy, ix] = float(row.persistent_assembly_probability)
                n[iz, iy, ix] = float(row.trace_count)
                successes[iz, iy, ix] = float(row.target_count)
    return sc, q, rho, cube, n, successes


def variance_indices(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-order and total-effect indices for axes SC, q, rho."""
    total_variance = float(np.var(cube, ddof=0))
    if total_variance <= 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    first = np.empty(3)
    total = np.empty(3)
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        conditional_on_axis = np.mean(cube, axis=other)
        first[axis] = np.var(conditional_on_axis, ddof=0) / total_variance
        conditional_on_others = np.mean(cube, axis=axis)
        total[axis] = 1.0 - np.var(conditional_on_others, ddof=0) / total_variance
    return first, total


def bootstrap_indices(
    probability: np.ndarray,
    n: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    first_draws = np.empty((draws, 3))
    total_draws = np.empty((draws, 3))
    integer_n = n.astype(int)
    for i in range(draws):
        sampled = rng.binomial(integer_n, probability) / integer_n
        first_draws[i], total_draws[i] = variance_indices(sampled)
    return (
        np.quantile(first_draws, 0.025, axis=0),
        np.quantile(first_draws, 0.975, axis=0),
        np.quantile(total_draws, 0.025, axis=0),
        np.quantile(total_draws, 0.975, axis=0),
    )


def sensitivity_panel(
    ax: plt.Axes,
    first: np.ndarray,
    total: np.ndarray,
    first_low: np.ndarray,
    first_high: np.ndarray,
    total_low: np.ndarray,
    total_high: np.ndarray,
) -> None:
    # Reorder from cube axes [SC, q, rho] to the chemical reading order.
    order = np.array([2, 1, 0])
    labels = [
        r"Link removal / addition, $\rho_{\mathrm{off}}$",
        "Atomic M / BTB ratio",
        r"Ring-closure coefficient, $s_{\mathrm{close}}$",
    ]
    y = np.arange(3)
    width = 0.34
    first_o = first[order]
    total_o = total[order]
    ax.barh(
        y + width / 2,
        first_o,
        height=width,
        color="#5185C0",
        label="First-order effect",
        xerr=np.vstack([first_o - first_low[order], first_high[order] - first_o]),
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
    )
    ax.barh(
        y - width / 2,
        total_o,
        height=width,
        color="#E99D4E",
        label="Total effect including interactions",
        xerr=np.vstack([total_o - total_low[order], total_high[order] - total_o]),
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, float(np.nanmax(total_o) * 1.18)))
    ax.set_xlabel("Fraction of variance in persistent-assembly probability")
    ax.set_title("b  Importance of the three current kinetic coordinates")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, lw=0.7)


def phase_panel(
    ax,
    sc: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    cube: np.ndarray,
) -> dict[str, int]:
    z_dense = np.linspace(sc.min(), sc.max(), 70)
    y_dense = np.linspace(np.log10(q.min()), np.log10(q.max()), 70)
    x_dense = np.linspace(np.log10(rho.min()), np.log10(rho.max()), 70)
    interpolator = RegularGridInterpolator(
        (sc, np.log10(q), np.log10(rho)),
        cube,
        bounds_error=False,
        fill_value=None,
    )
    zz, yy, xx = np.meshgrid(z_dense, y_dense, x_dense, indexing="ij")
    dense = interpolator(np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])).reshape(zz.shape)
    vertices, faces, _, _ = marching_cubes(
        dense,
        level=0.5,
        spacing=(
            z_dense[1] - z_dense[0],
            y_dense[1] - y_dense[0],
            x_dense[1] - x_dense[0],
        ),
    )
    vertices[:, 0] += z_dense.min()
    vertices[:, 1] += y_dense.min()
    vertices[:, 2] += x_dense.min()
    # Matplotlib axes are x=log rho, y=log q, z=SC.
    xyz = np.column_stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]])
    surface = Poly3DCollection(
        xyz[faces],
        facecolor="#5185C0",
        edgecolor="none",
        alpha=0.34,
    )
    ax.add_collection3d(surface)

    sc_grid, q_grid, rho_grid = np.meshgrid(sc, q, rho, indexing="ij")
    intermediate = (cube >= 0.1) & (cube <= 0.9)
    reliable = cube >= 0.9
    suppressed = cube <= 0.1
    ax.scatter(
        np.log10(rho_grid[intermediate]),
        np.log10(q_grid[intermediate]),
        sc_grid[intermediate],
        c=cube[intermediate],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=12,
        alpha=0.8,
        edgecolors="none",
        label="Calculated transition-region conditions",
    )
    # Sparse endpoint guides: plot every fourth point so the volume remains legible.
    for mask, color, label in [
        (suppressed, "#5185C0", "Calculated: P <= 0.1"),
        (reliable, "#C96144", "Calculated: P >= 0.9"),
    ]:
        indices = np.flatnonzero(mask.ravel())[::4]
        flat_x = np.log10(rho_grid.ravel()[indices])
        flat_y = np.log10(q_grid.ravel()[indices])
        flat_z = sc_grid.ravel()[indices]
        ax.scatter(flat_x, flat_y, flat_z, color=color, s=7, alpha=0.30, label=label)

    ax.set_xlim(np.log10(rho.min()), np.log10(rho.max()))
    ax.set_ylim(np.log10(q.min()), np.log10(q.max()))
    ax.set_zlim(sc.min(), sc.max())
    rho_ticks = np.array([0.01, 0.1, 1, 10, 100, 1500.0])
    q_ticks = np.array([0.01, 0.1, 1, 10, 100.0])
    ax.set_xticks(np.log10(rho_ticks), ["0.01", "0.1", "1", "10", "100", "1500"])
    ax.set_yticks(np.log10(q_ticks), ["0.01", "0.1", "1", "10", "100"])
    ax.set_zticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel(r"Link removal / addition, $\rho_{\mathrm{off}}$", labelpad=8)
    ax.set_ylabel("Atomic M / BTB ratio", labelpad=8)
    ax.set_zlabel(r"Ring closure, $s_{\mathrm{close}}$", labelpad=7)
    ax.set_title("c  Current-parameter kinetic formation boundary", pad=14)
    ax.view_init(elev=25, azim=-56)
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(
        0,
        Patch(
            facecolor="#5185C0",
            edgecolor="none",
            alpha=0.34,
            label="Interpolated P = 0.5 surface",
        ),
    )
    labels.insert(0, "Interpolated P = 0.5 surface")
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.97),
    )
    return {
        "suppressed_points": int(suppressed.sum()),
        "transition_points": int(intermediate.sum()),
        "reliable_points": int(reliable.sum()),
        "surface_vertices": int(len(vertices)),
        "surface_faces": int(len(faces)),
    }


def main() -> None:
    source = pd.read_csv(SOURCE)
    btb = source[source.system == "btb_mol"].copy()
    btb[["SC", "q", "off_weight"]] = btb[["SC", "q", "off_weight"]].astype(float)
    sc, q, rho, cube, n, successes = build_probability_cube(btb)
    first, total = variance_indices(cube)
    first_low, first_high, total_low, total_high = bootstrap_indices(cube, n)

    sensitivity = pd.DataFrame(
        {
            "parameter": ["s_close", "atomic_metal_linker_ratio", "rho_off"],
            "first_order": first,
            "first_order_ci_low": first_low,
            "first_order_ci_high": first_high,
            "total_effect": total,
            "total_effect_ci_low": total_low,
            "total_effect_ci_high": total_high,
            "scope": "uniform weighting over the sampled 9 x 14 x 10 BTB grid",
        }
    )
    sensitivity.to_csv(OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_sensitivity.csv"), index=False)

    fig = plt.figure(figsize=(13.2, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.25])
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1], projection="3d")
    sensitivity_panel(ax_b, first, total, first_low, first_high, total_low, total_high)
    counts = phase_panel(ax_c, sc, q, rho, cube)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.92, wspace=0.18)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")

    metadata = {
        "system": "btb_mol",
        "outcome": "persistent_assembly_probability",
        "event_definition": str(btb.nucleation_event_definition.iloc[0]),
        "grid_shape": [int(len(sc)), int(len(q)), int(len(rho))],
        "s_close_values": sc.tolist(),
        "atomic_metal_linker_ratios": q.tolist(),
        "rho_off_values": rho.tolist(),
        "replicates_per_condition": sorted(btb.trace_count.astype(int).unique().tolist()),
        "sensitivity_method": (
            "direct first-order and total-effect variance indices on the uniformly "
            "weighted discrete full-factorial grid; binomial parametric bootstrap"
        ),
        "phase_boundary": (
            "trilinear interpolation in s_close, log10(q), and log10(rho_off); "
            "isosurface at persistent-assembly probability 0.5"
        ),
        **counts,
        "source": SOURCE.as_posix(),
        "trace_source": TRACE_SOURCE.as_posix(),
    }
    OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2), encoding="utf-8"
    )
    print(OUTPUT_PREFIX.with_suffix(".png"))


if __name__ == "__main__":
    main()
