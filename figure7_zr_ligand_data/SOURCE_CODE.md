# Source Code for the Zr-Link Study

This note lists the simulation and plotting code that is directly relevant to
the `Zr-link` comparison between `Hf-BTB-MOL` and `UiO-66`.

The goal is to keep the final manuscript-facing data package lightweight without
duplicating the source code itself. The curated data are stored in
[figure7_zr_ligand_data](README.md) and
[../zr_ligand_one_shot_minimal_summary](../zr_ligand_one_shot_minimal_summary/README.md),
while the underlying simulation code remains in the system-specific folders
below.

## Figure assembly

- [make_figure7_draft.py](make_figure7_draft.py)
  Builds the current multi-panel `Figure 7` draft from curated CSV tables only.

## Hf-BTB-MOL simulation code

### Core one-shot Zr-link model

- [../MOL-Zr-Ligand/MOL_Assembly_Large_Correction_20250811.py](../MOL-Zr-Ligand/MOL_Assembly_Large_Correction_20250811.py)
  Main assembly and geometry engine for the `Hf-BTB-MOL` system. This file
  contains the coordination logic and the one-shot preassociated insertion
  behavior used in the retained study.

- [../MOL-Zr-Ligand/MOL_growth_main_20250811.py](../MOL-Zr-Ligand/MOL_growth_main_20250811.py)
  Main KMC growth driver for `Hf-BTB-MOL`.

- [../MOL-Zr-Ligand/scan_distorted_time_to_target.py](../MOL-Zr-Ligand/scan_distorted_time_to_target.py)
  Batch scan driver used to generate the time-to-target datasets summarized in
  the curated one-shot tables.

### Supporting local utilities

- [../MOL-Zr-Ligand/distorted_ligand_model.py](../MOL-Zr-Ligand/distorted_ligand_model.py)
  Shared helper for the distorted / preassociated ligand treatment.

- [../MOL-Zr-Ligand/multisite_linker_exchange_model.py](../MOL-Zr-Ligand/multisite_linker_exchange_model.py)
  Shared helper for multisite exchange behavior in the `MOL` line.

- [../MOL-Zr-Ligand/run_mol_zr_ligand_case.py](../MOL-Zr-Ligand/run_mol_zr_ligand_case.py)
  Single-case launcher for local `MOL` runs.

## UiO-66 simulation code

### Core one-shot Zr-link model

- [../UiO-66/Zr-Ligand-ExplicitChannel/UiO66_Assembly_Large_Correction_20250811.py](../UiO-66/Zr-Ligand-ExplicitChannel/UiO66_Assembly_Large_Correction_20250811.py)
  Main assembly and geometry engine for the `UiO-66` explicit-channel model.
  This file contains the one-shot preassociated insertion logic used for the
  retained `target = 8` survival analysis and `target = 20` collapse summary.

- [../UiO-66/Zr-Ligand-ExplicitChannel/UiO66_growth_main_20250811.py](../UiO-66/Zr-Ligand-ExplicitChannel/UiO66_growth_main_20250811.py)
  Main KMC growth driver for `UiO-66`.

- [../UiO-66/Zr-Ligand-ExplicitChannel/scan_distorted_time_to_target.py](../UiO-66/Zr-Ligand-ExplicitChannel/scan_distorted_time_to_target.py)
  Batch scan driver used to generate the time-to-target and survival datasets
  retained in the curated package.

### Supporting local utilities

- [../UiO-66/Zr-Ligand-ExplicitChannel/distorted_ligand_model.py](../UiO-66/Zr-Ligand-ExplicitChannel/distorted_ligand_model.py)
  Shared helper for the distorted / preassociated ligand treatment in `UiO-66`.

- [../UiO-66/Zr-Ligand-ExplicitChannel/multisite_linker_exchange_model.py](../UiO-66/Zr-Ligand-ExplicitChannel/multisite_linker_exchange_model.py)
  Shared helper for multisite exchange behavior in the `UiO-66` line.

- [../UiO-66/Zr-Ligand-ExplicitChannel/run_zr_ligand_case.py](../UiO-66/Zr-Ligand-ExplicitChannel/run_zr_ligand_case.py)
  Single-case launcher for local `UiO-66` runs.

## Local input structures

- [../MOL-Zr-Ligand/input_monomers](../MOL-Zr-Ligand/input_monomers)
- [../UiO-66/Zr-Ligand-ExplicitChannel/input_monomers](../UiO-66/Zr-Ligand-ExplicitChannel/input_monomers)

These folders contain the local molecular building blocks used by the retained
simulation code.

## Scope

This source list is intentionally limited to the code needed to understand and
reproduce the final curated `Zr-link` study package. Large exploratory output
directories and superseded helper scripts are documented separately in the
archive notes and are not required to interpret the final figures.
