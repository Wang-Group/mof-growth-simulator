# Legacy Survival Models

This folder re-frames the older nucleation datasets as survival / time-to-event
problems instead of only condition-level classification problems.

Current active focus: `BTB-MOL` nucleation from
[MOL_nucleation](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/MOL_nucleation).
The `UiO-66 from amorphous` helper is retained only as an optional utility and is
not the main analysis path.

## Why Survival Analysis Fits the Old MOL Dataset

For the older `BTB-MOL` notebook in
[MOL_nucleation/ml_v1.ipynb](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/MOL_nucleation/ml_v1.ipynb),
the underlying `MOL_batch0.pkl` data already contains:

- a time-like trajectory outcome
- whether the trajectory reached `entity = 20`
- multiple replicates per condition

The older ML workflow compresses this into `P(crystal)` or a majority binary label.
That is useful, but it throws away censoring information. A survival model keeps:

- `event = reached target`
- `time = time to event`
- `censored = simulation ended before target was reached`

This is a better language for nucleation kinetics, especially when many trajectories
never reach the target within the observation window.

## Current Status

The old `BTB-MOL` dataset is now available as a survival-ready flat table:

- [mol_target20_survival.csv](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_target20_survival.csv)

Key dataset facts:

- `100` conditions
- `800` replicate trajectories
- `570` events
- event threshold: `entity = 20`

The current baseline comparison is:

- old notebook-style `RandomForest` classifier on the legacy majority-label task
- replicate-level `Cox` survival model, converted back to horizon event probability
  and majority probability for apples-to-apples comparison

Results from
[mol_survival_vs_legacy_ml_metrics.csv](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_survival_vs_legacy_ml_metrics.csv):

- legacy RF majority ROC-AUC: `0.910`
- survival-derived majority ROC-AUC: `0.850`
- legacy RF majority Brier: `0.109`
- survival-derived majority Brier: `0.158`
- survival replicate-level mean C-index: `0.760`

Interpretation:

- If the task is exactly the old notebook target, "does this condition crystallize in
  the majority of replicates?", the legacy RF is still stronger.
- If the goal is to discuss nucleation as a physical time-to-event process, the
  survival framing is better because it preserves censoring and replicate-level timing.
- So the survival model is not replacing the old RF on its own optimized binary task;
  it is adding a more mechanistic view of the same dataset.

## Important Chemistry Caveat

In the old `MOL_nucleation` / `KMC_example/MOL_KMC` line, `Zr` does **not** mean the
same thing as in the newer explicit `Zr-BTB` model.

In the older model, `Zr` mainly biases whether an external addition event is metal-like
or ligand-like. It does **not** include:

- an explicit `Zr-BTB` prebound species
- ligand sequestration into `Zr-BTB`
- a blocked prebound insertion channel
- poisoning through geometrically hindered prebound growth

So in the legacy survival analysis, `Zr` should be interpreted as part of the old
metal-vs-ligand addition-bias model, not as the newer explicit `Zr-BTB` chemistry.

## What Seems To Matter Most In The Old MOL Survival Fit

From the current Cox baseline, the strongest old-model survival drivers are:

- `SC`: strong positive effect on nucleation hazard
- `FA`: strong negative effect
- `KC`: clear positive effect
- `L`: moderate positive effect
- `Zr`: present but secondary in the old model

This is consistent with the old RF feature ranking as well: `SC`, `KC`, and `FA` are
more influential than `Zr` in the legacy dataset.

## Main Outputs

Comparison outputs:

- [mol_survival_vs_legacy_ml_summary.md](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_survival_vs_legacy_ml_summary.md)
- [mol_survival_vs_legacy_ml_metrics.csv](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_survival_vs_legacy_ml_metrics.csv)
- [mol_survival_vs_legacy_ml_condition_predictions.csv](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_survival_vs_legacy_ml_condition_predictions.csv)
- [mol_survival_vs_legacy_ml_comparison.png](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_survival_vs_legacy_ml_comparison.png)

Example survival visualization:

- [mol_target20_survival_by_zr_quartile.svg](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/output/mol_survival_vs_legacy_ml/mol_target20_survival_by_zr_quartile.svg)

## What Is Here

- [extract_mol_survival_dataset.py](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/extract_mol_survival_dataset.py)
  converts `MOL_batch0.pkl` into a flat survival-ready CSV.
- [compare_mol_survival_vs_legacy_ml.py](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/compare_mol_survival_vs_legacy_ml.py)
  reproduces the older notebook-style RF baseline and compares it against a Cox survival model.
- [render_survival_svg.py](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/render_survival_svg.py)
  renders Kaplan-Meier-style survival curves from a flat CSV.
- [extract_uio66_from_amorphous_survival_dataset.py](C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator/legacy-survival-models/extract_uio66_from_amorphous_survival_dataset.py)
  is kept only as an optional legacy helper.

## Suggested Usage

### 1. Extract the old BTB-MOL dataset into survival format

```powershell
C:\Users\yibinjiang\anaconda3\envs\chemistry_tutor\python.exe `
  C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\extract_mol_survival_dataset.py `
  --batch-pkl C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\MOL_nucleation\MOL_batch0.pkl `
  --target-entities 20 `
  --output-csv C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\output\mol_target20_survival.csv
```

### 2. Compare survival modeling against the old ML baseline

```powershell
C:\Users\yibinjiang\anaconda3\envs\chemistry_tutor\python.exe `
  C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\compare_mol_survival_vs_legacy_ml.py `
  --flat-csv C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\output\mol_target20_survival.csv `
  --output-dir C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\output\mol_survival_vs_legacy_ml
```

### 3. Render a Kaplan-Meier-style plot from any flat survival CSV

```powershell
C:\Users\yibinjiang\anaconda3\envs\chemistry_tutor\python.exe `
  C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\render_survival_svg.py `
  --input-csv <flat survival csv> `
  --group-column zr_mM `
  --time-column time_seconds `
  --event-column event `
  --output-svg <output svg>
```

## Notes

- The main purpose of this folder is now the old `BTB-MOL` nucleation line, not the
  old `UiO-66 from amorphous` line.
- The `BTB-MOL` extractor preserves replicate-level rows because `MOL_batch0.pkl`
  already stores multiple replicate outcomes per condition.
- The current survival baseline is intentionally simple. A future nonlinear survival
  model such as random survival forest or XGBoost-AFT may close the gap to the old RF
  while retaining time-to-event information.
