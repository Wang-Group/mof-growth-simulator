from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_FILE = PACKAGE_DIR / "data" / "260316.csv"
OUTPUT_DIR = PACKAGE_DIR / "output"
PREFIX = "uio66_dls_gualtieri"
FIT_SEED = 3001
INDUCTION_END_S = 225.0


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
    }
)


def logistic(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gualtieri_alpha(
    time_s: np.ndarray,
    a_n_s: float,
    b_n_s: float,
    k_g_per_s: float,
    n_g: int,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    growth = 1.0 - np.exp(
        -np.power(np.clip(k_g_per_s * np.maximum(time_s, 0.0), 0.0, None), n_g)
    )
    nucleation = logistic((time_s - a_n_s) / b_n_s)
    return np.clip(nucleation * growth, 0.0, 1.0)


def score_model(
    observed_time_s: np.ndarray,
    observed_alpha: np.ndarray,
    params: tuple[float, float, float],
    n_g: int,
) -> float:
    a_n_s, b_n_s, k_g_per_s = params
    predicted = gualtieri_alpha(observed_time_s, a_n_s, b_n_s, k_g_per_s, n_g)
    return float(np.mean(np.square(predicted - observed_alpha)))


def fit_gualtieri_curve(
    observed_time_s: np.ndarray,
    observed_alpha: np.ndarray,
    seed: int = FIT_SEED,
) -> tuple[int, float, float, float]:
    positive_times = observed_time_s[observed_time_s > 0.0]
    min_positive = float(positive_times.min()) if positive_times.size else 1.0
    max_time = max(float(observed_time_s.max()), min_positive * 10.0)
    rng = np.random.default_rng(seed)

    peak = float(observed_alpha.max()) if observed_alpha.size else 0.0
    if peak > 0.0:
        t_peak = float(observed_time_s[np.argmin(np.abs(observed_alpha - peak / 2.0))])
    else:
        t_peak = max_time / 2.0
    t_peak = max(t_peak, min_positive)

    a_lo = math.log10(min_positive / 20.0)
    a_hi = math.log10(max_time * 20.0)
    b_lo = math.log10(min_positive / 50.0)
    b_hi = math.log10(max_time * 20.0)
    k_lo = math.log10(1.0 / (max_time * 20.0))
    k_hi = math.log10(10.0 / min_positive)

    best_score = float("inf")
    best_fit = (1, t_peak, max(t_peak / 4.0, min_positive / 10.0), 1.0 / t_peak)

    heuristic_a = np.array([t_peak / 2.0, t_peak, t_peak * 2.0, min_positive, max_time / 2.0])
    heuristic_b = np.array([t_peak / 10.0, t_peak / 4.0, t_peak / 2.0, t_peak, max_time / 3.0])
    heuristic_k = np.array([1.0 / t_peak, 0.5 / t_peak, 2.0 / t_peak, 1.0 / max_time, 5.0 / max_time])

    for n_g in (1, 2, 3):
        for a_n_s in heuristic_a:
            for b_n_s in heuristic_b:
                for k_g_per_s in heuristic_k:
                    candidate = (
                        max(float(a_n_s), 1e-12),
                        max(float(b_n_s), 1e-12),
                        max(float(k_g_per_s), 1e-12),
                    )
                    candidate_score = score_model(
                        observed_time_s, observed_alpha, candidate, n_g
                    )
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_fit = (n_g, *candidate)

        sample_count = 4000
        a_samples = 10 ** rng.uniform(a_lo, a_hi, sample_count)
        b_samples = 10 ** rng.uniform(b_lo, b_hi, sample_count)
        k_samples = 10 ** rng.uniform(k_lo, k_hi, sample_count)
        for a_n_s, b_n_s, k_g_per_s in zip(a_samples, b_samples, k_samples):
            candidate = (float(a_n_s), float(b_n_s), float(k_g_per_s))
            candidate_score = score_model(
                observed_time_s, observed_alpha, candidate, n_g
            )
            if candidate_score < best_score:
                best_score = candidate_score
                best_fit = (n_g, *candidate)

    best_n_g, best_a, best_b, best_k = best_fit
    best_logs = np.log10(np.array([best_a, best_b, best_k], dtype=float))

    for step in (0.7, 0.35, 0.18, 0.08, 0.04):
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for delta in (-step, step):
                    trial_logs = best_logs.copy()
                    trial_logs[axis] += delta
                    trial_logs[0] = np.clip(trial_logs[0], a_lo, a_hi)
                    trial_logs[1] = np.clip(trial_logs[1], b_lo, b_hi)
                    trial_logs[2] = np.clip(trial_logs[2], k_lo, k_hi)
                    trial = tuple(10 ** trial_logs)
                    trial_score = score_model(
                        observed_time_s, observed_alpha, trial, best_n_g
                    )
                    if trial_score < best_score:
                        best_score = trial_score
                        best_logs = trial_logs
                        improved = True

    best_a, best_b, best_k = (10 ** best_logs).tolist()
    return best_n_g, float(best_a), float(best_b), float(best_k)


def load_and_process_trace() -> tuple[pd.DataFrame, dict[str, float]]:
    source = pd.read_csv(DATA_FILE)
    if source.shape[1] < 4:
        raise ValueError("Expected four columns in data/260316.csv")

    # The last two columns include the 90 s sample-preparation period.
    time_s = source.iloc[:, 2].to_numpy(dtype=float)
    raw_size_nm = source.iloc[:, 3].to_numpy(dtype=float)
    smoothed_size_nm = (
        pd.Series(raw_size_nm)
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )

    baseline_nm = float(np.median(smoothed_size_nm[:5]))
    plateau_nm = float(np.mean(smoothed_size_nm[-5:]))
    size_range_nm = max(plateau_nm - baseline_nm, 1e-12)
    alpha_raw = np.clip((smoothed_size_nm - baseline_nm) / size_range_nm, 0.0, 1.0)
    alpha_monotonic = np.maximum.accumulate(alpha_raw)

    trace = pd.DataFrame(
        {
            "time_s_with_preparation": time_s,
            "raw_size_nm": raw_size_nm,
            "smoothed_size_nm": smoothed_size_nm,
            "alpha_raw": alpha_raw,
            "alpha_monotonic": alpha_monotonic,
        }
    )
    metadata = {
        "baseline_nm": baseline_nm,
        "plateau_nm": plateau_nm,
        "size_range_nm": size_range_nm,
    }
    return trace, metadata


def render_figure(
    trace: pd.DataFrame,
    fit_curve: pd.DataFrame,
    parameters: dict[str, float | int | str],
) -> None:
    fig, ax = plt.subplots(figsize=(3.15, 2.75))

    ax.axvspan(0.0, INDUCTION_END_S, color="#F4C7A1", alpha=0.45, linewidth=0)
    ax.axvline(float(parameters["aN_s"]), color="#BCA08C", linestyle=":", linewidth=0.8)
    ax.axhline(float(parameters["baseline_nm"]), color="#BDBDBD", linestyle=":", linewidth=0.8)
    ax.scatter(
        trace["time_s_with_preparation"],
        trace["raw_size_nm"],
        s=11,
        marker="s",
        color="#1F1A1A",
        edgecolor="none",
        zorder=3,
    )
    ax.plot(
        fit_curve["time_s"],
        fit_curve["fit_size_nm"],
        color="#F28E2B",
        linewidth=1.8,
        zorder=2,
    )

    ax.text(8, 38.0, "Induction Period", ha="left", va="top", fontsize=8)
    ax.text(
        0.97,
        0.08,
        "Gualtieri fit\n" + rf"$R^2$={float(parameters['r2_alpha']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Size (nm)", fontweight="bold")
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(direction="out", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT_DIR / f"{PREFIX}_fit.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{PREFIX}_fit.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace, metadata = load_and_process_trace()
    time_s = trace["time_s_with_preparation"].to_numpy(dtype=float)
    observed_alpha = trace["alpha_monotonic"].to_numpy(dtype=float)

    n_g, a_n_s, b_n_s, k_g_per_s = fit_gualtieri_curve(time_s, observed_alpha)
    fit_at_points = gualtieri_alpha(time_s, a_n_s, b_n_s, k_g_per_s, n_g)
    trace["gualtieri_fit_alpha"] = fit_at_points
    trace["gualtieri_fit_size_nm"] = (
        metadata["baseline_nm"] + fit_at_points * metadata["size_range_nm"]
    )

    residual = fit_at_points - observed_alpha
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    r2_denom = float(np.sum(np.square(observed_alpha - observed_alpha.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / r2_denom

    dense_time_s = np.linspace(0.0, max(600.0, float(time_s.max())), 1000)
    dense_alpha = gualtieri_alpha(dense_time_s, a_n_s, b_n_s, k_g_per_s, n_g)
    fit_curve = pd.DataFrame(
        {
            "time_s": dense_time_s,
            "fit_alpha": dense_alpha,
            "fit_size_nm": metadata["baseline_nm"]
            + dense_alpha * metadata["size_range_nm"],
        }
    )

    source_sha256 = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    parameters: dict[str, float | int | str] = {
        "source_csv": "data/260316.csv",
        "source_sha256": source_sha256,
        "time_column_zero_based_index": 2,
        "size_column_zero_based_index": 3,
        "n_points": len(trace),
        "time_start_s": float(time_s.min()),
        "time_end_s": float(time_s.max()),
        "baseline_nm": metadata["baseline_nm"],
        "plateau_nm": metadata["plateau_nm"],
        "nG": int(n_g),
        "aN_s": a_n_s,
        "bN_s": b_n_s,
        "kG_per_s": k_g_per_s,
        "kN_per_s": 1.0 / a_n_s,
        "rmse_alpha": rmse,
        "r2_alpha": r2,
    }

    trace.to_csv(OUTPUT_DIR / f"{PREFIX}_processed_trace.csv", index=False)
    fit_curve.to_csv(OUTPUT_DIR / f"{PREFIX}_fit_curve.csv", index=False)
    pd.DataFrame([parameters]).to_csv(
        OUTPUT_DIR / f"{PREFIX}_parameters.csv", index=False
    )
    render_figure(trace, fit_curve, parameters)

    summary = [
        "# UiO-66 DLS Gualtieri Fit Result",
        "",
        "- Source: `data/260316.csv` (last two columns include preparation time)",
        f"- Source SHA-256: `{source_sha256}`",
        f"- Points: `{len(trace)}`",
        f"- Time range: `{time_s.min():.1f}-{time_s.max():.1f} s`",
        f"- Baseline size: `{metadata['baseline_nm']:.3f} nm`",
        f"- Plateau size: `{metadata['plateau_nm']:.3f} nm`",
        f"- nG: `{n_g}`",
        f"- aN: `{a_n_s:.6f} s`",
        f"- bN: `{b_n_s:.6f} s`",
        f"- kG: `{k_g_per_s:.9g} s^-1`",
        f"- R^2: `{r2:.6f}`",
        f"- RMSE: `{rmse:.6f}`",
        "",
        "The fit is performed against the monotonic apparent-progress proxy, not directly against crystallinity.",
    ]
    (OUTPUT_DIR / f"{PREFIX}_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print(f"R^2 = {r2:.6f}")
    print(f"aN = {a_n_s:.6f} s")
    print(f"bN = {b_n_s:.6f} s")
    print(f"kG = {k_g_per_s:.9g} s^-1")


if __name__ == "__main__":
    main()
