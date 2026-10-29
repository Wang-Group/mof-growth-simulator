"""Validate and analyze the paired exact-fragment molecular response surface.

The script keeps three de novo observables distinct:

1. probability that one initialized candidate produces a 30-entity persistent assembly;
2. those successful nucleation events divided by candidate model-time exposure;
3. pooled committors for excursions of each intermediate size.

Propagation is measured in the independent frozen-seed calculation. A span
between the first-ever visits to 20 and 30 in a de novo trace is not used as a
growth rate because failed excursions may occur between those visits. None of
these quantities is labelled as a volumetric nucleation rate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MIN_RESOLVED_EXCURSIONS = 20


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def poisson_rate_interval(events: int, exposure: float) -> tuple[float, float]:
    """Garwood 95% interval; zero events returns a finite upper detection limit."""

    if exposure <= 0.0:
        return math.nan, math.nan
    try:
        from scipy.stats import chi2

        lower = 0.0 if events == 0 else 0.5 * chi2.ppf(0.025, 2 * events) / exposure
        upper = 0.5 * chi2.ppf(0.975, 2 * (events + 1)) / exposure
        return float(lower), float(upper)
    except Exception:
        if events == 0:
            return 0.0, -math.log(0.05) / exposure
        estimate = events / exposure
        half = 1.96 * math.sqrt(events) / exposure
        return max(0.0, estimate - half), estimate + half


def _iter_trace_json(results: Path):
    """Yield ``(name, trace)`` from a directory or a compressed tar archive."""

    if results.is_file():
        with tarfile.open(results, "r:*") as archive:
            members = [
                member
                for member in archive
                if member.isfile() and member.name.endswith(".json")
            ]
            if any(member.name.endswith(".error.json") for member in members):
                raise ValueError(f"Archive contains error records: {results}")
            if not members:
                raise FileNotFoundError(f"No result JSON files in {results}")
            for member in members:
                handle = archive.extractfile(member)
                if handle is None:
                    raise OSError(f"Could not read {member.name} from {results}")
                yield Path(member.name).name, json.loads(handle.read())
        return
    files = sorted(results.glob("*.json"))
    error_files = sorted(results.glob("*.error.json"))
    if error_files:
        raise ValueError(f"Found {len(error_files)} error files in {results}")
    if not files:
        raise FileNotFoundError(f"No result JSON files in {results}")
    for path in files:
        yield path.name, json.loads(path.read_text(encoding="utf-8"))


def read_results(results: Path) -> tuple[pd.DataFrame, dict[tuple[Any, ...], dict[int, dict[str, int]]], dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    pooled: dict[tuple[Any, ...], dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"target": 0, "collapse": 0, "censored": 0})
    )
    failures: list[str] = []
    runner_hashes: set[str] = set()
    selector_hashes: set[str] = set()
    engine_hashes: dict[str, set[str]] = defaultdict(set)

    for name, trace in _iter_trace_json(results):
        if trace.get("schema_version") != "common-molecular-kmc-v1":
            failures.append(f"{name}: schema")
            continue
        system = trace["system"]
        q = float(trace["composition_mapping"]["atomic_metal_to_linker_ratio"])
        off_weight = float(trace["coordinates"]["off_weight"])
        sc = float(trace["coordinates"]["closure_entropy_coefficient"])
        key = (system, q, off_weight, sc)
        termination = trace["termination"]
        final_state = trace["final_state"]
        target_entities = int(termination["target_entities"])
        effective_depth = int(termination["effective_fragment_search_depth"])
        if effective_depth < target_entities + 1:
            failures.append(f"{name}: depth={effective_depth}")
        if not bool(final_state.get("is_single_connected_component")):
            failures.append(f"{name}: disconnected")
        if list(final_state.get("connected_component_sizes", [])) != [
            int(final_state["entities"])
        ]:
            failures.append(f"{name}: component sizes")
        provenance = trace["provenance"]
        runner_hashes.add(str(provenance.get("runner_sha256", "")))
        selector_hashes.add(str(provenance.get("selector_sha256", "")))
        engine_hashes[system].add(str(provenance.get("engine_sha256", "")))

        for size_text, counts in trace["excursions"]["outcome_by_size"].items():
            size = int(size_text)
            for outcome in ("target", "collapse", "censored"):
                pooled[key][size][outcome] += int(counts.get(outcome, 0))

        fp = trace["first_passage_model_time"]
        target_reached = termination["status"] == "target_reached"
        first_arrival_time_20 = float(fp["20"]) if fp.get("20") is not None else math.nan
        first_arrival_20 = math.isfinite(first_arrival_time_20)
        first_arrival_exposure_20 = (
            first_arrival_time_20
            if first_arrival_20
            else float(trace["clock"]["model_time"])
        )
        events = trace["event_counts"]
        external_attempts = sum(
            int(events.get(name, 0))
            for name in ("add_node_attempt", "add_linker_attempt")
        )
        external_accepts = sum(
            int(events.get(name, 0))
            for name in ("add_node_accepted", "add_linker_accepted")
        )
        rows.append(
            {
                "run_id": trace["run_id"],
                "condition_id": trace["condition_id"],
                "system": system,
                "q": q,
                "node_activity_share": float(trace["coordinates"]["node_fraction"]),
                "off_weight": off_weight,
                "SC": sc,
                "closure_weight": float(trace["coordinates"]["closure_weight"]),
                "replicate": int(trace["replicate"]),
                "target_reached": target_reached,
                "first_arrival_20": first_arrival_20,
                "first_arrival_time_20": first_arrival_time_20,
                "first_arrival_exposure_time_20": first_arrival_exposure_20,
                "termination_status": termination["status"],
                "model_time": float(trace["clock"]["model_time"]),
                "physical_seconds": trace["clock"].get("physical_seconds"),
                "executed_events": int(termination["executed_events"]),
                "maximum_entities": int(termination["maximum_entities"]),
                "external_attempts": external_attempts,
                "external_accepts": external_accepts,
                "close_events": int(events.get("close_internal", 0)),
                "remove_events": int(events.get("remove_bond", 0)),
                "final_cycle_rank": int(final_state["cycle_rank"]),
                "effective_fragment_search_depth": effective_depth,
            }
        )

    if failures:
        raise ValueError(f"Validation failed ({len(failures)}):\n" + "\n".join(failures[:30]))
    if len(runner_hashes) != 1 or "" in runner_hashes:
        raise ValueError(f"Runner hashes: {runner_hashes}")
    if len(selector_hashes) != 1 or "" in selector_hashes:
        raise ValueError(f"Selector hashes: {selector_hashes}")
    if any(len(values) != 1 or "" in values for values in engine_hashes.values()):
        raise ValueError(f"Engine hashes: {dict(engine_hashes)}")
    provenance_summary = {
        "runner_sha256": sorted(runner_hashes),
        "selector_sha256": sorted(selector_hashes),
        "engine_sha256_by_system": {
            key: sorted(values) for key, values in engine_hashes.items()
        },
    }
    return pd.DataFrame(rows), pooled, provenance_summary


def committor_summary(pooled: dict[tuple[Any, ...], dict[int, dict[str, int]]]) -> tuple[pd.DataFrame, dict[tuple[Any, ...], float]]:
    records = []
    n50: dict[tuple[Any, ...], float] = {}
    for condition, sizes in pooled.items():
        crossing = math.nan
        for size in sorted(sizes):
            counts = sizes[size]
            resolved = counts["target"] + counts["collapse"]
            probability = counts["target"] / resolved if resolved else math.nan
            records.append(
                {
                    "system": condition[0],
                    "q": condition[1],
                    "off_weight": condition[2],
                    "SC": condition[3],
                    "entity_count": size,
                    "target_excursions": counts["target"],
                    "collapse_excursions": counts["collapse"],
                    "censored_excursions": counts["censored"],
                    "resolved_excursions": resolved,
                    "persistent_assembly_committor": probability,
                }
            )
            if math.isnan(crossing) and resolved >= MIN_RESOLVED_EXCURSIONS and probability >= 0.5:
                crossing = float(size)
        n50[condition] = crossing
    return pd.DataFrame(records), n50


def condition_summary(
    traces: pd.DataFrame,
    committor: pd.DataFrame,
    n50: dict[tuple[Any, ...], float],
) -> pd.DataFrame:
    q20_lookup = {
        (row.system, row.q, row.off_weight, row.SC): row.persistent_assembly_committor
        for row in committor.itertuples()
        if int(row.entity_count) == 20
    }
    records = []
    keys = ["system", "q", "off_weight", "SC"]
    for condition, group in traces.groupby(keys, sort=True):
        target_count = int(group["target_reached"].sum())
        first_arrival_count_20 = int(group["first_arrival_20"].sum())
        count = int(len(group))
        exposure = float(group["model_time"].sum())
        first_arrival_exposure_20 = float(group["first_arrival_exposure_time_20"].sum())
        probability = target_count / count
        probability_low, probability_high = wilson_interval(target_count, count)
        rate = target_count / exposure if exposure > 0.0 else math.nan
        rate_low, rate_high = poisson_rate_interval(target_count, exposure)
        first_arrival_probability_20 = first_arrival_count_20 / count
        first_arrival_probability_20_low, first_arrival_probability_20_high = wilson_interval(
            first_arrival_count_20, count
        )
        first_arrival_rate_20 = (
            first_arrival_count_20 / first_arrival_exposure_20
            if first_arrival_exposure_20 > 0.0
            else math.nan
        )
        first_arrival_rate_20_low, first_arrival_rate_20_high = poisson_rate_interval(
            first_arrival_count_20, first_arrival_exposure_20
        )
        acceptance = group["external_accepts"] / group["external_attempts"].replace(0, np.nan)
        records.append(
            {
                "system": condition[0],
                "q": condition[1],
                "off_weight": condition[2],
                "SC": condition[3],
                "node_activity_share": float(group["node_activity_share"].iloc[0]),
                "closure_weight": float(group["closure_weight"].iloc[0]),
                "trace_count": count,
                "target_count": target_count,
                "first_arrival_count_20": first_arrival_count_20,
                "event_cap_count": int((group["termination_status"] == "event_cap").sum()),
                "wall_time_cap_count": int((group["termination_status"] == "wall_time_cap").sum()),
                "no_enabled_event_count": int((group["termination_status"] == "no_enabled_event").sum()),
                "persistent_assembly_probability": probability,
                "persistent_assembly_probability_ci_low": probability_low,
                "persistent_assembly_probability_ci_high": probability_high,
                "candidate_model_time_exposure": exposure,
                "persistent_assembly_event_rate_per_candidate_model_time": rate,
                "persistent_assembly_event_rate_ci_low": rate_low,
                "persistent_assembly_event_rate_ci_high": rate_high,
                "zero_event_upper_detection_limit": rate_high if target_count == 0 else math.nan,
                "nucleation_event_definition": "first production of a 30-entity persistent assembly",
                "nucleation_event_count": target_count,
                "nucleation_event_probability": probability,
                "nucleation_event_probability_ci_low": probability_low,
                "nucleation_event_probability_ci_high": probability_high,
                "nucleation_event_rate_per_candidate_model_time": rate,
                "nucleation_event_rate_ci_low": rate_low,
                "nucleation_event_rate_ci_high": rate_high,
                "first_arrival_probability_20": first_arrival_probability_20,
                "first_arrival_probability_20_ci_low": first_arrival_probability_20_low,
                "first_arrival_probability_20_ci_high": first_arrival_probability_20_high,
                "first_arrival_model_time_exposure_20": first_arrival_exposure_20,
                "first_arrival_rate_20_per_candidate_model_time": first_arrival_rate_20,
                "first_arrival_rate_20_ci_low": first_arrival_rate_20_low,
                "first_arrival_rate_20_ci_high": first_arrival_rate_20_high,
                "first_arrival_zero_event_upper_detection_limit_20": (
                    first_arrival_rate_20_high if first_arrival_count_20 == 0 else math.nan
                ),
                "pooled_committor_at_20": q20_lookup.get(condition, math.nan),
                "pooled_committor_n50_entities": n50.get(condition, math.nan),
                "median_external_acceptance": float(acceptance.median()) if acceptance.notna().any() else math.nan,
                "median_maximum_entities": float(group["maximum_entities"].median()),
                "median_final_cycle_rank": float(group["final_cycle_rank"].median()),
            }
        )
    return pd.DataFrame(records)


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _linear_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(np.unique(values)), dtype=float)
    if len(values) == 1:
        return np.array([max(0.0, values[0] - 0.5), values[0] + 0.5])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(([values[0]], midpoints, [values[-1]]))


def _log_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(np.unique(values)), dtype=float)
    logs = np.log10(values)
    if len(values) == 1:
        return 10.0 ** np.array([logs[0] - 0.5, logs[0] + 0.5])
    midpoints = (logs[:-1] + logs[1:]) / 2.0
    edges = np.concatenate(([logs[0]], midpoints, [logs[-1]]))
    return 10.0**edges


def plot_sc_page(summary: pd.DataFrame, sc: float, destination: Path) -> None:
    _configure_style()
    core = summary[(summary["SC"] == sc) & summary["q"].between(0.1, 10.0)].copy()
    systems = (("uio66", "UiO-66"), ("btb_mol", "BTB-MOL"))
    metrics = (
        ("nucleation_event_probability", "Nucleation-event probability", "viridis", (0, 1), False),
        ("nucleation_event_rate_per_candidate_model_time", "Nucleation events / candidate model time", "magma", None, True),
        ("pooled_committor_at_20", "Pooled committor at 20 entities", "cividis", (0, 1), False),
    )
    display_limits: dict[str, tuple[float, float]] = {}
    for metric, _label, _cmap, fixed_limits, log_transform in metrics:
        if fixed_limits is not None:
            display_limits[metric] = fixed_limits
            continue
        raw = core[metric].to_numpy(float)
        transformed = (
            np.log10(np.where(raw > 0, raw, np.nan)) if log_transform else raw
        )
        finite = transformed[np.isfinite(transformed)]
        display_limits[metric] = (
            (float(np.min(finite)), float(np.max(finite)))
            if len(finite)
            else (0.0, 1.0)
        )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True, sharey=True)
    for row_index, (system, title) in enumerate(systems):
        selected = core[core["system"] == system]
        for column_index, (metric, label, cmap, limits, log_transform) in enumerate(metrics):
            ax = axes[row_index, column_index]
            q_values = np.array(sorted(selected["q"].unique()), dtype=float)
            off_values = np.array(sorted(selected["off_weight"].unique()), dtype=float)
            grid = (
                selected.pivot(index="off_weight", columns="q", values=metric)
                .reindex(index=off_values, columns=q_values)
                .to_numpy(float)
            )
            plotted = grid.copy()
            if log_transform:
                plotted = np.log10(np.where(plotted > 0, plotted, np.nan))
            metric_limits = display_limits[metric]
            colormap = mpl.colormaps[cmap].copy()
            colormap.set_bad("#E5E5E5")
            artist = ax.pcolormesh(
                _linear_edges(q_values),
                _log_edges(off_values),
                np.ma.masked_invalid(plotted),
                cmap=colormap,
                vmin=metric_limits[0],
                vmax=metric_limits[1],
                shading="flat",
                rasterized=True,
            )
            zero_events = selected["nucleation_event_count"].eq(0)
            if metric == "nucleation_event_rate_per_candidate_model_time" and zero_events.any():
                ax.scatter(
                    selected.loc[zero_events, "q"],
                    selected.loc[zero_events, "off_weight"],
                    marker="x",
                    c="#222222",
                    s=18,
                    linewidths=0.65,
                    label="zero events; upper limit tabulated",
                )
            ax.set_yscale("log")
            ax.set_xlim(q_values[0], q_values[-1])
            ax.set_ylim(off_values[0], off_values[-1])
            ax.grid(False)
            if row_index == 0:
                ax.set_title(label)
            if column_index == 0:
                ax.set_ylabel(f"{title}\nBond-removal bias")
            if row_index == 1:
                ax.set_xlabel("Atomic metal/linker ratio, q")
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.025)
            if log_transform:
                colorbar.set_label("log$_{10}$(metric)")
    fig.suptitle(f"Matched molecular response surfaces; closure coefficient = {sc:g}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(Path(str(destination) + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(str(destination) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sc_sensitivity(summary: pd.DataFrame, destination: Path) -> None:
    _configure_style()
    selections = (("uio66", 1.0, "UiO-66, q=1"), ("btb_mol", 3.0, "BTB-MOL, q=3"))
    off_values = (0.01, 0.63, 10.0, 160.0)
    colors = dict(zip(off_values, ("#2878B5", "#56B4A9", "#F39B1E", "#C93C3C")))
    metrics = (
        ("nucleation_event_probability", "Nucleation-event probability", False),
        ("nucleation_event_rate_per_candidate_model_time", "Nucleation-event rate / candidate model time", True),
        ("pooled_committor_at_20", "Pooled committor at 20 entities", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1), sharex=True)
    for row, (system, q, title) in enumerate(selections):
        selected = summary[(summary["system"] == system) & np.isclose(summary["q"], q)]
        for column, (metric, label, log_transform) in enumerate(metrics):
            ax = axes[row, column]
            for off in off_values:
                line = selected[np.isclose(selected["off_weight"], off)].sort_values("SC")
                ax.plot(line["SC"], line[metric], marker="o", ms=3.5, lw=1.2, color=colors[off], label=f"rho={off:g}")
            if log_transform:
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(label)
            if column == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel("Closure coefficient")
            ax.grid(alpha=0.18, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="outside lower center")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(Path(str(destination) + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(str(destination) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-traces", type=int, default=80640)
    parser.add_argument("--output-prefix", default="common_response_surface_v2")
    args = parser.parse_args()
    results = args.results.resolve()
    traces, pooled, provenance = read_results(results)
    if len(traces) != args.expected_traces:
        raise ValueError(f"Expected {args.expected_traces} traces, found {len(traces)}")
    committor, n50 = committor_summary(pooled)
    summary = condition_summary(traces, committor, n50)
    prefix = HERE / args.output_prefix
    traces.to_csv(prefix.with_name(prefix.name + "_trace_metrics.csv"), index=False)
    committor.to_csv(prefix.with_name(prefix.name + "_pooled_committor.csv"), index=False)
    summary.to_csv(prefix.with_name(prefix.name + "_condition_summary.csv"), index=False)
    for sc in sorted(summary["SC"].unique()):
        plot_sc_page(summary, float(sc), prefix.with_name(prefix.name + f"_SC_{sc:g}"))
    plot_sc_sensitivity(summary, prefix.with_name(prefix.name + "_closure_sensitivity"))
    report = {
        "trace_count": int(len(traces)),
        "condition_count": int(len(summary)),
        "systems": sorted(traces["system"].unique()),
        "target_count": int(traces["target_reached"].sum()),
        "first_arrival_count_20": int(traces["first_arrival_20"].sum()),
        "termination_counts": {
            str(key): int(value)
            for key, value in traces["termination_status"].value_counts().items()
        },
        "connected_graph_validation": "passed",
        "minimum_fragment_search_depth": int(traces["effective_fragment_search_depth"].min()),
        "provenance": provenance,
        "claim_boundary": (
            "A successful de novo nucleation event is operationally defined as the first "
            "production of a 30-entity persistent assembly from one initialized candidate. "
            "Its event rate is per candidate model-time exposure. Intermediate-size pooled "
            "committors are reported separately. Propagation is measured only in the "
            "independent frozen-seed calculation. No reported quantity is a volumetric "
            "nucleation rate."
        ),
    }
    prefix.with_name(prefix.name + "_analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
