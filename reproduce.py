"""Offline plotting adapters. Never rerun KMC or mutate archived sources."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCALE = 0.175


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 11,
                         "legend.fontsize": 8, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "svg.fonttype": "none"})


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def long_growth(out):
    root = ROOT / "01_uio66_long_growth"
    first = pd.read_csv(root / "data/first_passage_24.csv")
    assert len(first) == 24 and (first.max_entity >= 1000).all()
    raw = first.copy()
    for col in list(raw.columns):
        if col.endswith("_s"):
            raw[col[:-2] + "_u"] = raw.pop(col) / SCALE
    raw.to_csv(out / "first_passage_24_raw_u.csv", index=False)
    time_cols = [c for c in raw if c.endswith("_u")]
    raw[time_cols].describe(percentiles=[.25, .5, .75]).T.to_csv(out / "first_passage_summary_raw_u.csv")
    traces = pd.read_csv(root / "data/representative_pre30_trajectories.csv")
    residence = pd.read_csv(root / "data/residence_pre30_by_case_entity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4), layout="constrained")
    a, b, c, d = axes.ravel()
    for case, trace in traces.groupby("case_index", sort=True):
        t30 = raw.loc[raw.case_index == case, "t_entity_30_u"].iloc[0]
        a.plot(trace.time_s / SCALE, trace.entity, lw=.55, label=f"I{int(case):02d}, u30={t30:.0f}")
    a.set(xlabel="Raw Algorithm Time u", ylabel="Entity Number", ylim=(0, 35),
          title="(a) Repeated Waiting Before Growth")
    a.set_xticks(np.arange(0, 160001, 40000))
    a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    a.legend(loc="upper left", framealpha=.95)
    for n in (5, 6, 9, 13, 20, 30):
        x = np.sort(raw[f"t_entity_{n}_u"].to_numpy())
        b.step(np.r_[x[0] * .9, x], np.r_[1., 1 - np.arange(1, len(x) + 1) / len(x)],
               where="post", lw=1.2, label=f"N = {n}")
    b.set(xscale="log", xlabel="First-Passage Time u", ylabel="Fraction Not Yet Reached",
          ylim=(-.03, 1.03), title="(b) First Passage to Each Size")
    b.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)
    intervals = [(3, 5), (5, 6), (6, 9), (9, 13), (13, 20), (20, 30)]
    vals = [raw[f"t_{lo}_to_{hi}_u"].to_numpy() for lo, hi in intervals]
    assert all(np.all(v > 0) for v in vals)
    # Match the historical linear-space KDE, then display it on a log axis.
    c.violinplot(vals, showextrema=False, showmedians=True)
    c.set_xticks(range(1, 7), [f"{lo}-{hi}" for lo, hi in intervals])
    c.set(yscale="log", xlabel="Entity-Number Interval", ylabel="Interval in Raw Algorithm Time u",
          title="(c) Paired Transition Intervals")
    heat = residence.pivot(index="case_index", columns="entity", values="residence_time_s") / SCALE
    order = raw.sort_values("t_entity_30_u").case_index
    heat = heat.reindex(index=order, columns=range(1, 30)).fillna(0)
    im = d.imshow(np.log10(heat.to_numpy() + 1e-3), origin="upper", aspect="auto",
                  interpolation="nearest", cmap="YlOrRd")
    d.set_xticks(np.arange(0, 29, 4), np.arange(1, 30, 4))
    d.set(xlabel="Entity Number", ylabel="Trajectory Rank by u30",
          title="(d) Residence Below 30 Entities")
    fig.colorbar(im, ax=d, fraction=.046, pad=.02).set_label("log10 (Residence u + 0.001)")
    save(fig, out, "FigS7_raw_u")

    levels = [5, 9, 13, 20, 30, 100, 300, 500, 1000]
    values = raw[[f"t_entity_{n}_u" for n in levels]]
    fig, ax = plt.subplots(figsize=(4.5, 4.2), layout="constrained")
    for n in levels:
        ax.scatter(raw[f"t_entity_{n}_u"], np.full(len(raw), n), s=12, color=".4", alpha=.4)
    ax.plot(values.median().to_numpy(), levels, "o-", color="#D95F02", lw=1.6, ms=4)
    ax.fill_betweenx(levels, values.quantile(.25).to_numpy(), values.quantile(.75).to_numpy(), color="#D95F02", alpha=.18)
    ax.axhline(13, color="#B22222", ls="--", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel="First-Passage Time u", ylabel="Entity Number", ylim=(4, 1300))
    ax.set_yticks(levels, [str(n) for n in levels])
    ax.set_box_aspect(1)
    save(fig, out, "Fig3D_raw_u")

    archived = module(root / "linkage_original/render_uio66_entity13_dls_linkage.py", "original_linkage")
    # Only change output destinations, never the historical data or rendering logic.
    target = out / "Fig3DE_legacy_scaled"
    target.mkdir(exist_ok=True)
    archived.OUTPUT_DIR = target
    archived.OUT_PNG = target / "uio66_entity13_dls_linkage.png"
    archived.OUT_SVG = target / "uio66_entity13_dls_linkage.svg"
    archived.render()


def dls(out):
    fit = module(ROOT / "02_dls/fit_uio66_dls_gualtieri.py", "original_dls")
    fit.OUTPUT_DIR = out / "DLS"
    fit.main()
    style()
    trace = pd.read_csv(fit.OUTPUT_DIR / "uio66_dls_gualtieri_processed_trace.csv")
    curve = pd.read_csv(fit.OUTPUT_DIR / "uio66_dls_gualtieri_fit_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
    axes[0].scatter(trace.time_s_with_preparation, trace.raw_size_nm, s=14, label="Raw DLS")
    axes[0].plot(curve.time_s, curve.fit_size_nm, color="#F28E2B", label="Mapped Fit")
    axes[0].set(xlabel="Time (s)", ylabel="DLS Size (nm)")
    axes[1].scatter(trace.time_s_with_preparation, trace.alpha_monotonic, s=14, label="Processed Progress")
    axes[1].plot(curve.time_s, curve.fit_alpha, color="#F28E2B", label="Gualtieri Fit")
    axes[1].set(xlabel="Time (s)", ylabel="Apparent Progress")
    for ax in axes:
        ax.legend(loc="upper left")
    save(fig, fit.OUTPUT_DIR, "full_range_diagnostic")


def btb_trajectory(out):
    table = pd.read_csv(ROOT / "03_btb_trajectory/data/trajectory_constant_segments.csv.gz", float_precision="round_trip")
    x = np.r_[table.time_start_stored, table.time_end_stored.iloc[-1]]
    y = np.r_[table.entity, table.entity.iloc[-1]]
    first20 = table.loc[table.entity >= 20, "time_start_stored"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 3.7), layout="constrained")
    ax.step(x, y, where="post", lw=.65, color="#1f77b4")
    ax.set(xlim=(first20 * .9, first20), ylim=(0, 20), xlabel="Stored Source Time (Calibration Pending)", ylabel="Entity Number")
    save(fig, out, "Fig5A_source_clock")


def btb_motifs(out):
    data = pd.read_csv(ROOT / "04_btb_motifs/data/unique_structures_result.csv")
    rows = []
    for n in range(1, 14):
        ids = data[f"entity_{n}"].dropna()
        for index, count in ids.value_counts().items():
            rows.append(dict(entity=n, structure_index=int(index), count=int(count),
                             denominator=len(ids), percent=100 * count / len(ids)))
    frequencies = pd.DataFrame(rows).sort_values(["entity", "structure_index"])
    frequencies.to_csv(out / "btb_motif_frequencies.csv", index=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 8), layout="constrained")
    for n, ax in enumerate(axes.ravel(), start=1):
        if n > 13:
            ax.set_visible(False)
            continue
        f = frequencies[frequencies.entity == n]
        ax.bar(f.structure_index, f.percent, color="#4C78A8", width=.75)
        ax.set(title=f"N = {n}; Count = {f.denominator.iloc[0]}", ylim=(0, 105))
        ax.set_xticks(f.structure_index)
        ax.tick_params(axis="x", labelsize=7)
        if n >= 10:
            ax.set_xlabel("Structure Index")
        if n % 4 == 1:
            ax.set_ylabel("Frequency (%)")
    save(fig, out, "Fig5D_frequency_check")


def mixed(out):
    path = next((ROOT / "05_mixed_zr6_zr12/data").glob("*.summary_trace.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = pd.DataFrame(data["mean_trace"])
    mean.to_csv(out / "mixed_seed_archived_mean_trace.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for col, label, color in (("Zr6_AA", "Zr6", "#008B8B"), ("Zr12_AA", "Zr12", "#C44E52"), ("BDC", "BDC", "#4C78A8")):
        for run in data["replay_runs"]:
            trace = pd.DataFrame(run["traces"])
            ax.step(trace.sim_time_seconds, trace[col], where="post", color=color, lw=.5, alpha=.18)
        ax.plot(mean.sim_time_seconds, mean[col + "_mean"], color=color, lw=1.8, label=label)
    ax.set(xlabel="Stored Simulation Time (s; Original x0.1 Convention)", ylabel="Entity Count")
    ax.legend(loc="upper left")
    save(fig, out, "FigS21_mixed_counts_check")


def repair(out):
    data = pd.read_csv(ROOT / "06_vacancy_repair/data/repair_progress_vs_growth.csv")
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(data.total_entities, data.bdc_fill, "o-", ms=2.3, color="#B7791F", label="Original BDC Vacancies")
    ax.plot(data.total_entities, data.zr_fill, "o-", ms=2.3, color="#2563EB", label="Original Zr Vacancies")
    ax.set(xlabel="Entity Number", ylabel="Fraction of Original Vacancies Filled", ylim=(-.03, 1.04))
    ax.legend(loc="lower right")
    save(fig, out, "FigS23_vacancy_fill_check")


GROUPS = dict(long_growth=long_growth, dls=dls, btb_trajectory=btb_trajectory,
              btb_motifs=btb_motifs, mixed=mixed, repair=repair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", *GROUPS], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Refuse writes into immutable evidence directories.
    for folder in ROOT.iterdir():
        if folder.is_dir() and (folder.name[:2].isdigit() or folder.name == "metadata"):
            if out == folder.resolve() or folder.resolve() in out.parents:
                raise ValueError("Choose a separate output directory, not an archived source folder")
    out.mkdir(parents=True, exist_ok=True)
    for name, action in GROUPS.items():
        if args.group in ("all", name):
            style()
            action(out)
            print(f"Reproduced {name}: {out}", flush=True)


if __name__ == "__main__":
    main()
