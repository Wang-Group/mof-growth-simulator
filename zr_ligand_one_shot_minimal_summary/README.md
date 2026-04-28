# Zr-Ligand One-Shot Minimal Summary

This folder is the compact data package for the `Zr-link` study.

It keeps only the datasets consistent with the simplified one-shot rule:

`one KMC event -> sample one configuration only -> if collision occurs, reject immediately`

No extra arm/site/bend retries are allowed within the same event.

## What to Keep

### 1. `Hf-BTB-MOL` one-shot target = 20

- [01_mol_one_shot_target20_core.csv](01_mol_one_shot_target20_core.csv)
- [11_mol_final_experiment_vs_kmc_logy.png](11_mol_final_experiment_vs_kmc_logy.png)
- [11_mol_final_experiment_vs_kmc_logy.svg](11_mol_final_experiment_vs_kmc_logy.svg)
- [11_mol_final_experiment_vs_kmc_logy_plot_data.csv](11_mol_final_experiment_vs_kmc_logy_plot_data.csv)

This is the final retained `BTB-MOL` dataset and plot for the manuscript-style
comparison. It uses the selected nonmonotonic one-shot condition and is the
source for `Figure 7D`.

### 2. `UiO-66` experiment-consistent one-shot target = 8

- [03_uio66_one_shot_target8_experiment_consistent.csv](03_uio66_one_shot_target8_experiment_consistent.csv)
- [04_uio66_one_shot_target8_experiment_survival.csv](04_uio66_one_shot_target8_experiment_survival.csv)

This is the primary retained `UiO-66` dataset because:

- it matches the one-shot rule
- it is under experiment-consistent conditions (`BDC = 4 mM`, `AA = 2400 mM`)
- many trajectories are right-censored, so survival analysis is more meaningful
  than a simple time average

Current success counts:

- `2 mM`: `5/24`
- `4 mM`: `4/24`
- `8 mM`: `5/24`
- `16 mM`: `2/24`
- `32 mM`: `0/24`

### 3. `UiO-66` one-shot target = 20 collapse

- [05_uio66_one_shot_target20_collapse.csv](05_uio66_one_shot_target20_collapse.csv)

This is retained as supporting mechanistic evidence. It shows directly that
productive growth collapses at high `Zr`, but it is not the main quantitative
dataset because many trajectories never reach `entity = 20`.

### 4. Cross-system helper tables and shared plot components

- [02_uio66_one_shot_target8_mechanism_scan.csv](02_uio66_one_shot_target8_mechanism_scan.csv)
- [06_cross_system_overview.csv](06_cross_system_overview.csv)
- [09_experimental_trend_overlay_data.csv](09_experimental_trend_overlay_data.csv)
- [10_prebound_site_success_comparison.csv](10_prebound_site_success_comparison.csv)
- [09_experimental_trend_overlay.png](09_experimental_trend_overlay.png)
- [10_prebound_site_success_comparison.png](10_prebound_site_success_comparison.png)

These files support the cross-system interpretation used in `Figure 7B` and
`Figure 7F`.

## Archived Items

Earlier exploratory `BTB-MOL` dual-axis plots have been moved to:

- [archive/obsolete_mol_plots](archive/obsolete_mol_plots)

Those files were useful during tuning, but they were superseded by
`11_mol_final_experiment_vs_kmc_logy.*`.

## Minimal Story Preserved Here

- `Hf-BTB-MOL`: one-shot target=20 is sufficient and remains fully productive
  across the retained scan
- `UiO-66`: one-shot target=8 should be interpreted with a survival model
- `UiO-66 target=20`: keep only as supporting evidence for high-`Zr` growth
  collapse
