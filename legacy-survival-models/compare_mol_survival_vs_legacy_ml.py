import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def effective_equilibrium_constant(kc, fa_conc_mM, h2o_dmf_ratio=0.6):
    h2o_pure = 55500.0
    dmf_pure = 12900.0
    equilibrium_constant = 1.64

    dmf_conc = dmf_pure / (1.0 + h2o_dmf_ratio)
    h2o_conc = h2o_pure * h2o_dmf_ratio / (1.0 + h2o_dmf_ratio)

    h2o_formate_coord_power = h2o_conc / fa_conc_mM * 0.01
    dmf_formate_coord_power = dmf_conc / fa_conc_mM * 0.01

    return kc * equilibrium_constant / (1.0 + h2o_formate_coord_power + dmf_formate_coord_power)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a survival model for the legacy BTB-MOL nucleation dataset "
            "against the older notebook-style RF classification model."
        )
    )
    parser.add_argument(
        "--flat-csv",
        required=True,
        help="Flat survival-ready MOL CSV from extract_mol_survival_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for metrics, predictions, and plots.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified condition-level CV folds. Default: 5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=45,
        help="Random state for CV. Default: 45.",
    )
    return parser.parse_args()


def load_data(flat_csv_path):
    rep_df = pd.read_csv(flat_csv_path)
    rep_df["condition_id"] = rep_df["condition_id"].astype(str)
    rep_df["event"] = rep_df["event"].astype(int)
    rep_df["time_seconds"] = rep_df["time_seconds"].astype(float)
    rep_df["replicate_index"] = rep_df["replicate_index"].astype(int)
    rep_df["K_eff"] = effective_equilibrium_constant(rep_df["kc"].to_numpy(), rep_df["fa_mM"].to_numpy())

    grouped = rep_df.groupby("condition_id", sort=True)
    cond_df = grouped.agg(
        zr_mM=("zr_mM", "first"),
        fa_mM=("fa_mM", "first"),
        linker_mM=("linker_mM", "first"),
        sc=("sc", "first"),
        kc=("kc", "first"),
        K_eff=("K_eff", "first"),
        n_rep=("event", "size"),
        success_fraction=("event", "mean"),
    ).reset_index()
    cond_df["majority_label"] = (cond_df["success_fraction"] >= 0.5).astype(int)
    return rep_df, cond_df


def majority_probability(single_event_prob, n_rep):
    threshold = math.ceil(0.5 * n_rep)
    total = 0.0
    for k in range(threshold, n_rep + 1):
        total += math.comb(n_rep, k) * (single_event_prob ** k) * ((1.0 - single_event_prob) ** (n_rep - k))
    return total


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_corr(x, y, fn):
    if len(set(np.asarray(x))) < 2 or len(set(np.asarray(y))) < 2:
        return float("nan")
    return float(fn(x, y)[0])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_df, cond_df = load_data(args.flat_csv)
    feature_cols = ["zr_mM", "fa_mM", "linker_mM", "sc", "kc", "K_eff"]
    horizon_seconds = float(rep_df["time_seconds"].max())

    X_cond = cond_df[feature_cols].to_numpy(dtype=float)
    y_majority = cond_df["majority_label"].to_numpy(dtype=int)
    condition_ids = cond_df["condition_id"].to_numpy()

    class_counts = np.bincount(y_majority)
    max_splits = int(class_counts.min()) if len(class_counts) >= 2 else 2
    n_splits = max(2, min(args.n_splits, max_splits))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    oof_rows = []
    fold_metric_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_cond, y_majority), start=1):
        train_conditions = set(condition_ids[train_idx])
        test_conditions = set(condition_ids[test_idx])

        train_cond = cond_df.iloc[train_idx].copy()
        test_cond = cond_df.iloc[test_idx].copy()

        # Legacy notebook-style RF classifier at condition level.
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42 + fold_idx,
            n_jobs=-1,
        )
        rf.fit(train_cond[feature_cols], train_cond["majority_label"])
        rf_prob = rf.predict_proba(test_cond[feature_cols])[:, 1]
        rf_pred = (rf_prob >= 0.5).astype(int)

        # Survival model at replicate level, split by condition to avoid leakage.
        rep_train = rep_df[rep_df["condition_id"].isin(train_conditions)].copy()
        rep_test = rep_df[rep_df["condition_id"].isin(test_conditions)].copy()

        scaler = StandardScaler()
        rep_train_scaled = rep_train.copy()
        rep_test_scaled = rep_test.copy()
        rep_train_scaled[feature_cols] = scaler.fit_transform(rep_train[feature_cols])
        rep_test_scaled[feature_cols] = scaler.transform(rep_test[feature_cols])

        cph_train = rep_train_scaled[feature_cols + ["time_seconds", "event"]].copy()
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(cph_train, duration_col="time_seconds", event_col="event")

        rep_test_risk = cph.predict_partial_hazard(rep_test_scaled[feature_cols]).to_numpy().reshape(-1)
        rep_test_cindex = concordance_index(
            rep_test_scaled["time_seconds"].to_numpy(),
            -rep_test_risk,
            rep_test_scaled["event"].to_numpy(),
        )

        test_cond_scaled = test_cond.copy()
        test_cond_scaled[feature_cols] = scaler.transform(test_cond[feature_cols])
        surv_at_horizon = cph.predict_survival_function(test_cond_scaled[feature_cols], times=[horizon_seconds])
        surv_prob = surv_at_horizon.loc[horizon_seconds].to_numpy()
        event_prob = 1.0 - surv_prob
        surv_majority_prob = np.array(
            [majority_probability(float(p), int(n_rep)) for p, n_rep in zip(event_prob, test_cond["n_rep"])]
        )
        surv_majority_pred = (surv_majority_prob >= 0.5).astype(int)

        fold_metric_rows.append(
            {
                "fold": fold_idx,
                "rf_majority_roc_auc": safe_auc(test_cond["majority_label"], rf_prob),
                "rf_majority_brier": brier_score_loss(test_cond["majority_label"], rf_prob),
                "rf_majority_accuracy": accuracy_score(test_cond["majority_label"], rf_pred),
                "survival_majority_roc_auc": safe_auc(test_cond["majority_label"], surv_majority_prob),
                "survival_majority_brier": brier_score_loss(test_cond["majority_label"], surv_majority_prob),
                "survival_majority_accuracy": accuracy_score(test_cond["majority_label"], surv_majority_pred),
                "replicate_c_index": rep_test_cindex,
            }
        )

        for row_idx, (_, row) in enumerate(test_cond.iterrows()):
            oof_rows.append(
                {
                    "fold": fold_idx,
                    "condition_id": row["condition_id"],
                    "zr_mM": row["zr_mM"],
                    "fa_mM": row["fa_mM"],
                    "linker_mM": row["linker_mM"],
                    "sc": row["sc"],
                    "kc": row["kc"],
                    "K_eff": row["K_eff"],
                    "n_rep": int(row["n_rep"]),
                    "empirical_success_fraction": row["success_fraction"],
                    "majority_label": int(row["majority_label"]),
                    "rf_majority_prob": float(rf_prob[row_idx]),
                    "rf_majority_pred": int(rf_pred[row_idx]),
                    "survival_event_prob_by_horizon": float(event_prob[row_idx]),
                    "survival_majority_prob": float(surv_majority_prob[row_idx]),
                    "survival_majority_pred": int(surv_majority_pred[row_idx]),
                }
            )

    oof_df = pd.DataFrame(oof_rows).sort_values("condition_id").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_metric_rows)

    # Overall metrics from OOF predictions.
    overall_metrics = [
        {
            "model": "legacy_rf_majority",
            "metric": "roc_auc_majority",
            "value": safe_auc(oof_df["majority_label"], oof_df["rf_majority_prob"]),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "brier_majority",
            "value": brier_score_loss(oof_df["majority_label"], oof_df["rf_majority_prob"]),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "accuracy_majority",
            "value": accuracy_score(oof_df["majority_label"], oof_df["rf_majority_pred"]),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "f1_majority",
            "value": f1_score(oof_df["majority_label"], oof_df["rf_majority_pred"]),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "rmse_vs_empirical_success_fraction",
            "value": float(np.sqrt(np.mean((oof_df["rf_majority_prob"] - oof_df["empirical_success_fraction"]) ** 2))),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "spearman_vs_empirical_success_fraction",
            "value": safe_corr(oof_df["rf_majority_prob"], oof_df["empirical_success_fraction"], spearmanr),
        },
        {
            "model": "legacy_rf_majority",
            "metric": "pearson_vs_empirical_success_fraction",
            "value": safe_corr(oof_df["rf_majority_prob"], oof_df["empirical_success_fraction"], pearsonr),
        },
        {
            "model": "survival_cox_majority",
            "metric": "roc_auc_majority",
            "value": safe_auc(oof_df["majority_label"], oof_df["survival_majority_prob"]),
        },
        {
            "model": "survival_cox_majority",
            "metric": "brier_majority",
            "value": brier_score_loss(oof_df["majority_label"], oof_df["survival_majority_prob"]),
        },
        {
            "model": "survival_cox_majority",
            "metric": "accuracy_majority",
            "value": accuracy_score(oof_df["majority_label"], oof_df["survival_majority_pred"]),
        },
        {
            "model": "survival_cox_majority",
            "metric": "f1_majority",
            "value": f1_score(oof_df["majority_label"], oof_df["survival_majority_pred"]),
        },
        {
            "model": "survival_cox_eventprob",
            "metric": "rmse_vs_empirical_success_fraction",
            "value": float(np.sqrt(np.mean((oof_df["survival_event_prob_by_horizon"] - oof_df["empirical_success_fraction"]) ** 2))),
        },
        {
            "model": "survival_cox_eventprob",
            "metric": "spearman_vs_empirical_success_fraction",
            "value": safe_corr(oof_df["survival_event_prob_by_horizon"], oof_df["empirical_success_fraction"], spearmanr),
        },
        {
            "model": "survival_cox_eventprob",
            "metric": "pearson_vs_empirical_success_fraction",
            "value": safe_corr(oof_df["survival_event_prob_by_horizon"], oof_df["empirical_success_fraction"], pearsonr),
        },
        {
            "model": "survival_cox_replicate",
            "metric": "mean_fold_c_index",
            "value": float(fold_df["replicate_c_index"].mean()),
        },
        {
            "model": "survival_cox_replicate",
            "metric": "std_fold_c_index",
            "value": float(fold_df["replicate_c_index"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        },
    ]
    metrics_df = pd.DataFrame(overall_metrics)

    # Write outputs.
    oof_df.to_csv(output_dir / "mol_survival_vs_legacy_ml_condition_predictions.csv", index=False)
    fold_df.to_csv(output_dir / "mol_survival_vs_legacy_ml_fold_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "mol_survival_vs_legacy_ml_metrics.csv", index=False)

    # Markdown report.
    def metric_value(model, metric):
        match = metrics_df[(metrics_df["model"] == model) & (metrics_df["metric"] == metric)]
        return float(match["value"].iloc[0])

    report_lines = [
        "# MOL Survival vs Legacy ML",
        "",
        f"- Source flat dataset: `{args.flat_csv}`",
        f"- Conditions: `{len(cond_df)}`",
        f"- Replicates: `{len(rep_df)}`",
        f"- Event threshold: `entity = 20`",
        f"- Time horizon for event-probability comparison: `{horizon_seconds:.3f} s`",
        f"- Cross-validation: `{n_splits}` stratified condition-level folds",
        "",
        "## Key Results",
        "",
        f"- Legacy RF majority ROC-AUC: `{metric_value('legacy_rf_majority', 'roc_auc_majority'):.3f}`",
        f"- Survival-derived majority ROC-AUC: `{metric_value('survival_cox_majority', 'roc_auc_majority'):.3f}`",
        f"- Legacy RF Brier: `{metric_value('legacy_rf_majority', 'brier_majority'):.3f}`",
        f"- Survival-derived majority Brier: `{metric_value('survival_cox_majority', 'brier_majority'):.3f}`",
        f"- Legacy RF RMSE vs empirical success fraction: `{metric_value('legacy_rf_majority', 'rmse_vs_empirical_success_fraction'):.3f}`",
        f"- Survival event-probability RMSE vs empirical success fraction: `{metric_value('survival_cox_eventprob', 'rmse_vs_empirical_success_fraction'):.3f}`",
        f"- Survival replicate-level mean C-index: `{metric_value('survival_cox_replicate', 'mean_fold_c_index'):.3f}`",
        "",
        "## Interpretation",
        "",
        "- The legacy RF is trained on a condition-level binary label: whether the empirical crystallization fraction is at least 0.5.",
        "- The survival model is trained on replicate-level time/event data and preserves censoring.",
        "- The survival model can be converted back to a condition-level probability of reaching the target by the observation horizon, and then compared directly against the older ML framing.",
    ]
    (output_dir / "mol_survival_vs_legacy_ml_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Plot.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(
        oof_df["empirical_success_fraction"],
        oof_df["rf_majority_prob"],
        s=36,
        alpha=0.8,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("Empirical success fraction")
    ax.set_ylabel("Legacy RF predicted P(majority crystal)")
    ax.set_title("Legacy ML vs empirical fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[0, 1]
    ax.scatter(
        oof_df["empirical_success_fraction"],
        oof_df["survival_event_prob_by_horizon"],
        s=36,
        alpha=0.8,
        color="#d62728",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("Empirical success fraction")
    ax.set_ylabel(f"Survival predicted P(event by {horizon_seconds:.1f} s)")
    ax.set_title("Survival model vs empirical fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 0]
    labels = ["Legacy RF", "Survival -> majority"]
    roc_vals = [
        metric_value("legacy_rf_majority", "roc_auc_majority"),
        metric_value("survival_cox_majority", "roc_auc_majority"),
    ]
    brier_vals = [
        metric_value("legacy_rf_majority", "brier_majority"),
        metric_value("survival_cox_majority", "brier_majority"),
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, roc_vals, width=width, color="#4c78a8", label="ROC-AUC")
    ax.bar(x + width / 2, brier_vals, width=width, color="#f58518", label="Brier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Condition-level majority-label metrics")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(
        oof_df["rf_majority_prob"],
        oof_df["survival_majority_prob"],
        c=oof_df["empirical_success_fraction"],
        cmap="viridis",
        s=42,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("Legacy RF predicted P(majority)")
    ax.set_ylabel("Survival-derived P(majority)")
    ax.set_title("Model-to-model comparison")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Legacy BTB-MOL nucleation: survival model vs older ML classifier", fontsize=15)
    fig.savefig(output_dir / "mol_survival_vs_legacy_ml_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
