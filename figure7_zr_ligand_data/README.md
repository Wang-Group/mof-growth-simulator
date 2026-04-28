# Figure 7 Data Package

This folder is the curated manuscript package for the `Zr-link` comparison between
`Hf-BTB-MOL` and `UiO-66`.

It is intended to be the lightweight, push-ready version of the study. The raw
pilot scans and large intermediate output folders remain in their original local
workspaces under `MOL-Zr-Ligand/output/` and
`UiO-66/Zr-Ligand-ExplicitChannel/output/`, but the figure assembly in this
folder depends only on the curated data stored here and in
[zr_ligand_one_shot_minimal_summary](../zr_ligand_one_shot_minimal_summary/README.md).

## Main Deliverables

- [Figure7_draft.png](Figure7_draft.png)
- [Figure7_draft.svg](Figure7_draft.svg)
- [make_figure7_draft.py](make_figure7_draft.py)
- [SOURCE_CODE.md](SOURCE_CODE.md)

The current draft keeps `Panel A` as a placeholder for the preassociated
structure images that will be inserted manually.

## Figure Panels

- `A`: placeholder boxes for the preassociated `Hf-BTB` and `Zr-BDC` motifs
- `B`: relative experimental trend vs `Metal Concentration`
- `C`: seeded overgrowth linker-loading trends for `Hf-BTB-MOL` and `UiO-66`
- `D`: `Hf-BTB-MOL` experiment vs one-shot KMC time to `entity = 20`
- `E`: `UiO-66` survival to `entity = 8`
- `F`: productive insertion success for metal-side and linker-side attachment

## Curated CSV Files

### Figure-level tables

- [01_mol_target20_experiment_vs_one_shot.csv](01_mol_target20_experiment_vs_one_shot.csv)
- [02_mol_target8_20_100_summary.csv](02_mol_target8_20_100_summary.csv)
- [03_mol_mechanistic_summary.csv](03_mol_mechanistic_summary.csv)
- [04_uio66_experiment_consistent_target8_summary.csv](04_uio66_experiment_consistent_target8_summary.csv)
- [05_uio66_target8_survival_curves.csv](05_uio66_target8_survival_curves.csv)
- [06_uio66_target20_success_terminal.csv](06_uio66_target20_success_terminal.csv)
- [07_uio66_survival_best_fit_points.csv](07_uio66_survival_best_fit_points.csv)
- [08_uio66_mechanism_scan_diagnostics.csv](08_uio66_mechanism_scan_diagnostics.csv)
- [09_uio66_mechanism_vs_target20.csv](09_uio66_mechanism_vs_target20.csv)
- [10_source_manifest.csv](10_source_manifest.csv)

### Seeded overgrowth tables used in `Panel C`

- [BTB-MOL.csv](BTB-MOL.csv)
- [UiO-66.csv](UiO-66.csv)

### Workbook

- [figure7_zr_ligand_data.xlsx](figure7_zr_ligand_data.xlsx)

## Design Notes

- All figure text is formatted in `Arial`.
- `Panel D` uses log-scaled `y` axes with explicit major ticks for readability in
  Word and SVG editing.
- `Panel E` uses the local curated survival table rather than reading directly
  from the large `UiO-66` raw output folder.
- `Panel F` uses red shades for `MOL` and blue shades for `UiO-66` to separate
  the two systems visually.

## Provenance

The curated CSV files in this folder were consolidated from larger exploratory
output directories during the `Zr-link` study. Those raw scan folders are not
required to regenerate `Figure7_draft.*` once the curated data package is
present.

## Archive

- [archive/README.md](archive/README.md)

The archive keeps legacy helper scripts that are no longer required for the
final lightweight package.
