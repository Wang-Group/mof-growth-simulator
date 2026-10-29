"""Standalone Figure 3D, adapted from render_uio66_entity13_dls_linkage.py.

Retain the original nine thresholds, 24 runs, jitter, median, interquartile
band, palette and line settings. Undo the exporter's historical multiplier:
u = source CSV time_s / 0.175. No NMR-to-physical-time calibration is applied.
Only the simulation panel is drawn. No existing artwork is read or cropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP = Path(__file__).resolve().parent.parent
HISTORICAL_EXPORT_FACTOR = 0.175
THRESHOLDS = (5, 9, 13, 20, 30, 100, 300, 500, 1000)


def render(data_file: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(data_file)
    assert set(curves.case_index) == set(range(24))
    rows = []
    for case_index, frame in curves.groupby("case_index"):
        assert frame.time_s.is_monotonic_increasing
        row = {"case_index": int(case_index)}
        for threshold in THRESHOLDS:
            reached = frame.loc[frame.entity >= threshold, "time_s"]
            assert not reached.empty, (case_index, threshold)
            row[f"t_entity_{threshold}_u"] = float(reached.iloc[0]) / HISTORICAL_EXPORT_FACTOR
        rows.append(row)
    first = pd.DataFrame(rows)
    cols = [f"t_entity_{n}_u" for n in THRESHOLDS]
    values = first[cols]
    assert np.isfinite(values.to_numpy()).all() and (values.to_numpy() > 0).all()
    medians = values.median()
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    # Original renderer style and standalone-panel canvas from panel_runtime.py.
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
        "svg.hashsalt": "Figure3D_raw_u_20261027",
    })
    fig, ax = plt.subplots(figsize=(3.65, 3.65))
    rng = np.random.default_rng(7)
    y = np.asarray(THRESHOLDS, dtype=float)
    for i, col in enumerate(cols):
        jitter = rng.uniform(-0.11, 0.11, len(first))
        ax.scatter(first[col], np.full(len(first), y[i]) * (1.0 + 0.030 * jitter),
                   s=13, color="#6E6E6E", alpha=0.45, edgecolor="none", zorder=2)
    ax.plot(medians.to_numpy(), y, color="#D95F02", lw=1.8,
            marker="o", ms=4.2, zorder=4)
    ax.fill_betweenx(y, q25.to_numpy(), q75.to_numpy(), color="#D95F02",
                     alpha=0.18, linewidth=0, zorder=1)
    ax.axhline(13, color="#B22222", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.array([0.01, 5.0e4]) / HISTORICAL_EXPORT_FACTOR)
    ax.set_ylim(4, 1300)
    ax.set_yticks(y)
    ax.set_yticklabels([str(n) for n in THRESHOLDS])
    ax.set_xlabel(r"First-passage time, $u$")
    ax.set_ylabel("Entity Number")
    ax.set_title("Simulation: Latent Growth Threshold")
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    svg = output_dir / "Figure3D_raw_algorithm_time_u.svg"
    png = svg.with_suffix(".png")
    fig.savefig(svg, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.035,
                metadata={"Title": "Figure 3D: raw algorithmic first-passage time",
                          "Description": "24 trajectories, 9 thresholds; u = source time_s / 0.175. Original scatter, median and interquartile band; no physical-time calibration.",
                          "Creator": Path(__file__).name, "Date": None})
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    xml = ET.parse(svg).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert not xml.findall(f".//{ns}image"), "Raster embedded in SVG"
    assert xml.findall(f".//{ns}text") and xml.findall(f".//{ns}path")

    first.to_csv(output_dir / "Figure3D_first_passage_24_raw_u.csv", index=False)
    summary = pd.DataFrame({"entity": THRESHOLDS, "median_u": medians.to_numpy(),
                            "q25_u": q25.to_numpy(), "q75_u": q75.to_numpy()})
    summary.to_csv(output_dir / "Figure3D_first_passage_summary_raw_u.csv", index=False)
    report = {"figure": "3D", "coordinate": "raw algorithm time u",
              "conversion": "u = historical time_s / 0.175",
              "trajectories": len(first), "thresholds": list(THRESHOLDS),
              "plotted_time_values": int(values.size),
              "source_data": str(data_file),
              "source_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
              "original_renderer": str(GROUP / "original_code/render_uio66_entity13_dls_linkage.py"),
              "svg": str(svg), "png": str(png), "vector_only": True}
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path,
                        default=GROUP / "source_data/linkage/uio66_induction_vs_dls_first_passage_curves.csv")
    parser.add_argument("--output-dir", type=Path, default=GROUP / "Fig3D_raw_u_20261027")
    args = parser.parse_args()
    print(json.dumps(render(args.data_file, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
