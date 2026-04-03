# MOF Growth Simulator

This repository archives the code, input tables, and processed outputs used for event-based kinetic Monte Carlo (KMC) simulations of nucleation and growth in metal-organic layers (MOLs) and metal-organic frameworks (MOFs).

The simulation model treats framework assembly as a sequence of reversible coordination-exchange events:

- external addition of a new SBU or linker
- internal linking / ring closure between existing coordination sites
- dissociation of an existing coordination bond

Event probabilities are controlled by reagent concentrations, an effective ligand-exchange equilibrium term, and an entropy correction that favors internal ring-forming events. Across the systems studied here, the main qualitative conclusion is that ring-based connected motifs act as key nuclei for sustained growth.

This README is intended to be the primary top-level documentation for the repository.

## Systems covered

- BTB-based MOL growth (the Hf/Zr-BTB layered system used as the main 2D case study)
- UiO-66
- HKUST-1
- PCN-240

## Repository layout

```text
.
|-- KMC_example/
|   |-- input_monomers/                # Local .mol building blocks used by the simulators
|   `-- MOL_KMC/                       # Generic MOL KMC engine + 100-condition input table
|-- MOL_geometry/                      # Post-processing for MOL morphology / plane analysis
|-- MOL_nucleation/                    # Post-processing for MOL nucleation probability analysis
|-- UiO-66/
|   |-- code/                          # UiO-66 KMC engine + 100-condition input table
|   |-- UiO_66_batch0/                 # Packaged UiO-66 simulation outputs and analysis notebook
|   `-- UiO_66_from_amorphous/         # Active defect-seeded / amorphous-derived UiO-66 workspace
|-- HKUST-1/                           # HKUST-1 KMC engine snapshot
`-- PCN-240/                           # PCN-240 KMC engine snapshot
```

## What each folder contains

### `KMC_example/`

`KMC_example/MOL_KMC/` contains the generic MOL simulation code:

- `mol_growth_main_20250811.py`: the main KMC loop
- `MOL_Assembly_Large_Correction_20250811.py`: geometry handling, entity classes, and assembly logic
- `MOL_batch0.xlsx`: a 100-point Latin-hypercube parameter table for the MOL screen

`KMC_example/input_monomers/` stores the local `.mol` files for the SBUs and linkers used by the assembly code, including:

- `Zr6.mol`, `Zr12.mol`
- `Zr6_AA.mol`, `Zr12_AA.mol`
- `BDC.mol`, `BDC_OH.mol`
- `BTC.mol`
- `BTBa.mol`, `BTBb.mol`
- `Cu2.mol`
- `Fe_Co3.mol`
- `Hf6.mol`, `Hf12.mol`

These files are the local replacements for the absolute Linux paths still hard-coded inside several assembly scripts.

### `MOL_nucleation/`

This folder contains post-processing for the MOL nucleation screen:

- `MOL_batch0.pkl`: pickled results for 100 parameter sets
- `ml_v1.ipynb`: machine-learning / SHAP analysis of crystallization outcomes

`MOL_batch0.pkl` is a Python `dict` keyed by encoded condition names such as:

`Zr_<...>_FA_<...>_L_<...>_Ratio_0.6_Step_1e15.0_SC_<...>_KC_<...>`

Each key maps to 8 replicate outcome records (stored as 2-value arrays used by the analysis notebook), matching the `repeat_number = 8` setting in `KMC_example/MOL_KMC/MOL_batch0.xlsx`.

### `MOL_geometry/`

This folder contains post-processing for final MOL morphology:

- `_result.csv`: 284 analyzed structures with geometric descriptors
- `corr_one_plane_standardized.csv`: correlations for single-plane probability
- `corr_growth_rate_one_plane_standardized.csv`: correlations for single-plane growth rate
- `v1.ipynb`: the notebook that generates the correlation tables

`_result.csv` includes columns such as:

- `SBU_count`, `BTB_count`, `plane_count`
- `ellipse_a`, `ellipse_b`
- encoded synthesis parameters (`Zr`, `FA`, `L`, `Ratio`, `SC`, `KC`)
- per-plane areas (`plane_area_0`, `plane_area_1`, ...)

The `dir_path` column preserves original Linux workspace paths for provenance only.

### `UiO-66/`

`UiO-66/code/` contains the UiO-66 simulation engine:

- `UiO66_growth_main_20250811.py`: main KMC loop
- `UiO66_Assembly_Large_Correction_20250811.py`: assembly / geometry logic
- `UiO66_batch0.xlsx`: 100-condition parameter table for the UiO-66 scan

`UiO-66/UiO_66_batch0/` contains packaged outputs from the UiO-66 parameter scan:

- 400 result folders
- one `level_idx.json` per folder
- one `summary_plot.png` per folder
- `analyze_growth_rate_4_20.ipynb` for post-processing nucleation and growth statistics

The 400 folders correspond to `100 conditions x 4 repeats`, matching the `repeat_number = 4` field in `UiO66_batch0.xlsx`.

`UiO-66/UiO_66_from_amorphous/initialize_structures1/` is the active research workspace for defect-seeded and amorphous-derived UiO-66 continuation studies.

The current retained manuscript-oriented case there is:

- `UiO-66/UiO_66_from_amorphous/initialize_structures1/output/mixeddef800_zx2_seed800_continuous_from2124/`

That case follows an artificial mixed-defect UiO-66 seed under one fixed Zr-rich condition and currently documents:

- `Zr fill = 1.000`
- `BDC fill = 0.870` at the current latest endpoint
- continued growth from `2124` to `3651` entities

The workspace README for that line is:

- `UiO-66/UiO_66_from_amorphous/initialize_structures1/README.md`

### `HKUST-1/`

This folder contains a system-specific HKUST-1 simulation snapshot:

- `HKUST_1_growth_main_conc.py`
- `HKUST_1_Assembly_Large_Correction_conc.py`

Compared with the generic example, this version adds more robust pickling helpers (`safe_pickle_save`, `safe_pickle_load`) and saves "last structure by entity number" snapshots.

### `PCN-240/`

This folder contains the analogous PCN-240 simulation snapshot:

- `PCN_240_growth_main_conc.py`
- `PCN_240_Assembly_Large_Correction_conc.py`

Like the HKUST-1 version, it includes safer pickling utilities for large assembly objects.

## Scientific context

This repository implements a nucleus-centered KMC framework for identifying structural motifs that commit MOF/MOL growth toward persistent crystals.

The main scientific story preserved here is:

- nuclei are treated as recurrent structural motifs rather than just a critical size
- internal ring-closing events receive an entropy advantage in the model
- ring-based motifs dominate the enriched nucleus ensembles across the studied systems
- for the BTB-MOL system, high capping-agent concentration suppresses de novo out-of-plane nucleation and favors lateral 2D growth

Key parameter-scan context captured in the repository:

- MOL screen: 100 Latin-hypercube conditions, 8 repeats per condition, `H2O_DMF_RATIO = 0.6`
- MOL scan ranges: entropy coefficient `0.1-2`, equilibrium coefficient `0.1-2`, `Zr = 1-45 mM`, capping agent `0-22000 mM`, linker `1-35 mM`
- UiO-66 screen: 100 conditions, 4 repeats per condition, `H2O_DMF_RATIO = 0`
- UiO-66 scan ranges: entropy coefficient `0.1-2`, equilibrium coefficient `0.1-2`, `Zr = 1-45 mM`, capping agent `100-8000 mM`, linker `1-150 mM`
- The MOL geometry table contains 284 successful crystallized structures selected for downstream morphology analysis

## How the code is organized

Each simulation system follows the same basic pattern:

1. An `Assembly_*.py` file defines:
   - entity classes for SBUs and linkers
   - parsing of `.mol` input files
   - the `Assembly` object
   - event selection and structural update methods
2. A `*_growth_main*.py` file sets top-level parameters directly in Python variables.
3. The main loop repeatedly chooses among:
   - internal linking
   - external growth
   - dissolution
4. The script writes trajectory and structure snapshots to the output folder.

Typical outputs include:

- `entities_number.pkl`: trajectory as `[time_like_value, entity_count]`
- intermittent `entities_number_<...>.pkl` snapshots
- `assembly.mol2` or `assembly_<...>.mol2`
- for HKUST-1 / PCN-240: `last_by_n/` snapshots and pickled `Assembly` objects

The raw trajectory timing is stored in simulation units and converted during downstream analysis notebooks.

## Running the simulations

This repository is an archival research code snapshot, not a packaged Python library. There is no CLI wrapper and no environment file yet. To rerun simulations, you edit parameters directly in the scripts and then execute the script from its own directory.

### Recommended Python environment

Core simulation dependencies:

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `ipython`

Useful / required for packaged analyses:

- `pandas`
- `openpyxl`
- `jupyter`
- `scikit-learn`
- `shap`
- `xgboost`
- `plotly`
- `dill` (recommended for HKUST-1 and PCN-240 snapshot pickling)

Example install:

```bash
pip install numpy scipy matplotlib tqdm ipython pandas openpyxl jupyter scikit-learn shap xgboost plotly dill
```

### Important path caveat

Several assembly scripts still hard-code absolute Linux paths inside their `PRELOADED_DATA` blocks. Before rerunning, replace those paths with local paths under `KMC_example/input_monomers/`.

Examples:

- `UiO-66/code/UiO66_Assembly_Large_Correction_20250811.py` expects `Zr6_AA.mol`, `Zr12_AA.mol`, `BDC.mol`
- `HKUST-1/HKUST_1_Assembly_Large_Correction_conc.py` expects `Cu2.mol`, `Zr12_AA.mol`, `BTC.mol`
- `PCN-240/PCN_240_Assembly_Large_Correction_conc.py` expects `Fe_Co3.mol`, `Zr12_AA.mol`, `BDC_OH.mol`
- `KMC_example/MOL_KMC/MOL_Assembly_Large_Correction_20250811.py` expects `Zr6.mol`, `Zr12.mol`, `BTBa.mol`, `BTBb.mol`

### Important output-folder caveat

The main scripts are not parameterized by command-line arguments. Output location is controlled by the `current_folder` variable defined at the top of each growth script.

- In `KMC_example/MOL_KMC/mol_growth_main_20250811.py` and `UiO-66/code/UiO66_growth_main_20250811.py`, `current_folder` is currently `None` and must be set manually before running.
- In `HKUST-1/HKUST_1_growth_main_conc.py` and `PCN-240/PCN_240_growth_main_conc.py`, `current_folder` is still set to an original Linux output path and should be changed to a local writable directory.

### Minimal rerun workflow

1. Open the assembly file for the target system and replace the absolute `.mol` input paths with local repo-relative paths.
2. Open the corresponding growth script and set:
   - concentrations and coefficients at the top of the file
   - `current_folder`
   - any system-specific limits such as `max_entities`
3. Run the script from the folder containing that script.

Examples:

```bash
cd UiO-66/code
python UiO66_growth_main_20250811.py
```

```bash
cd KMC_example/MOL_KMC
python mol_growth_main_20250811.py
```

```bash
cd HKUST-1
python HKUST_1_growth_main_conc.py
```

```bash
cd PCN-240
python PCN_240_growth_main_conc.py
```

## Notes on the packaged analysis files

- `MOL_nucleation/ml_v1.ipynb` analyzes crystallization probability across the MOL parameter screen and uses SHAP-based feature attribution.
- `MOL_geometry/v1.ipynb` analyzes morphology metrics such as single-plane probability and growth rate.
- `UiO-66/UiO_66_batch0/analyze_growth_rate_4_20.ipynb` parses `level_idx.json` files to analyze nucleation and post-nucleation growth behavior in UiO-66.

## Current limitations

- No `requirements.txt`, `environment.yml`, or automated workflow launcher is included.
- The simulation scripts are designed as editable research snapshots, not reusable modules.
- Several file paths still point to the original Linux workspace and need manual replacement before rerunning.
- The repository mainly contains simulation code and processed data products; raw experimental data are not packaged here.
- There are no automated tests in the current snapshot.

## Suggested citation

If you reuse this code or data, cite the associated publication if available, or contact the repository authors for preferred citation information.
