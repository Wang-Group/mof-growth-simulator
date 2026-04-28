# MOL Survival vs Legacy ML

- Source flat dataset: `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\legacy-survival-models\output\mol_target20_survival.csv`
- Conditions: `100`
- Replicates: `800`
- Event threshold: `entity = 20`
- Time horizon for event-probability comparison: `249.796 s`
- Cross-validation: `5` stratified condition-level folds

## Key Results

- Legacy RF majority ROC-AUC: `0.910`
- Survival-derived majority ROC-AUC: `0.850`
- Legacy RF Brier: `0.109`
- Survival-derived majority Brier: `0.158`
- Legacy RF RMSE vs empirical success fraction: `0.296`
- Survival event-probability RMSE vs empirical success fraction: `0.345`
- Survival replicate-level mean C-index: `0.760`

## Interpretation

- The legacy RF is trained on a condition-level binary label: whether the empirical crystallization fraction is at least 0.5.
- The survival model is trained on replicate-level time/event data and preserves censoring.
- The survival model can be converted back to a condition-level probability of reaching the target by the observation horizon, and then compared directly against the older ML framing.