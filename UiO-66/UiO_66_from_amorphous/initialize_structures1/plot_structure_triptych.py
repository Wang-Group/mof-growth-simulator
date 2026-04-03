import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = SCRIPT_DIR / "output" / "mixeddef800_zx2_seed800_continuous_from2124"
DEFAULT_SEED_MOL2 = DEFAULT_CASE_DIR / "references" / "UiO-66_R3_Mixeddef_0.40_seed800.mol2"
DEFAULT_INITIAL_MOL2 = DEFAULT_CASE_DIR / "references" / "assembly_2026-04-03_10-39-56_entity_number_2960.mol2"
DEFAULT_FINAL_MOL2 = DEFAULT_CASE_DIR / "assembly_final_entity_number_3651.mol2"
DEFAULT_PNG = DEFAULT_CASE_DIR / "seed_initial_final_structure_triptych.png"
DEFAULT_SVG = DEFAULT_CASE_DIR / "seed_initial_final_structure_triptych.svg"


def parse_mol2(path: Path, keep_elements=("O", "Zr")):
    coords = []
    elems = []
    in_atoms = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if in_atoms and line.startswith("@<TRIPOS>BOND"):
                break
            if not in_atoms:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            elem = parts[5].split(".")[0]
            if elem not in keep_elements:
                continue
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elems.append(elem)
    return np.asarray(coords, dtype=float), np.asarray(elems, dtype=object)


def rotation_matrix(ax_deg=24.0, ay_deg=-38.0, az_deg=0.0):
    ax = np.deg2rad(ax_deg)
    ay = np.deg2rad(ay_deg)
    az = np.deg2rad(az_deg)
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)],
        ]
    )
    ry = np.array(
        [
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)],
        ]
    )
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ]
    )
    return rz @ ry @ rx


def entity_count_from_name(path: Path):
    stem = path.stem
    marker = "entity_number_"
    if marker not in stem:
        return None
    return int(stem.split(marker, 1)[1].split(".")[0])


def load_series(label, path):
    coords, elems = parse_mol2(path)
    centered = coords - coords.mean(axis=0, keepdims=True)
    rotated = centered @ rotation_matrix().T
    return {
        "label": label,
        "path": path,
        "coords": rotated,
        "elems": elems,
        "entity_count": entity_count_from_name(path),
    }


def make_plot(series, png_out: Path, svg_out: Path):
    all_xy = np.concatenate([item["coords"][:, :2] for item in series], axis=0)
    lim = np.max(np.abs(all_xy)) * 1.08

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f7f4ee")

    colors = {"O": "#c7b08a", "Zr": "#2563eb"}
    sizes = {"O": 1.2, "Zr": 9.5}
    alphas = {"O": 0.24, "Zr": 0.88}

    for ax, item in zip(axes, series):
        ax.set_facecolor("#fffdf9")
        order = np.argsort(item["coords"][:, 2])
        xy = item["coords"][order, :2]
        elems = item["elems"][order]
        for elem in ("O", "Zr"):
            mask = elems == elem
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=sizes[elem],
                c=colors[elem],
                alpha=alphas[elem],
                linewidths=0,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d6d3d1")
            spine.set_linewidth(1.0)
        subtitle = item["label"]
        if item["entity_count"] is not None:
            subtitle += f"\n{item['entity_count']} entities"
        ax.set_title(subtitle, fontsize=12.5, color="#1f2937")

    fig.suptitle("UiO-66 same-condition continuous trajectory", fontsize=18, color="#1f2937")
    fig.text(
        0.5,
        0.94,
        "Left to right: artificial mixed-defect seed, defect-containing nucleus formed under the fixed condition, latest endpoint under the same condition.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.text(0.15, 0.05, "Zr atoms", color="#2563eb", fontsize=10)
    fig.text(0.22, 0.05, "O atoms", color="#8f7b58", fontsize=10)
    fig.text(0.83, 0.05, "Same scale, same view angle", ha="right", color="#64748b", fontsize=9)

    fig.savefig(png_out, dpi=220, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a three-panel structure comparison for the same-condition UiO-66 trajectory.")
    parser.add_argument("--seed-mol2", type=Path, default=DEFAULT_SEED_MOL2)
    parser.add_argument("--initial-mol2", type=Path, default=DEFAULT_INITIAL_MOL2)
    parser.add_argument("--final-mol2", type=Path, default=DEFAULT_FINAL_MOL2)
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def main():
    args = parse_args()
    series = [
        load_series("Mixed-defect seed", args.seed_mol2),
        load_series("Initial nucleus", args.initial_mol2),
        load_series("Final endpoint", args.final_mol2),
    ]
    make_plot(series, args.png_out, args.svg_out)
    print(args.png_out)
    print(args.svg_out)


if __name__ == "__main__":
    main()
