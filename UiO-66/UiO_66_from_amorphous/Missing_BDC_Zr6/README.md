# Missing_BDC_Zr6

This folder is a self-contained recovery workspace for the UiO-66 amorphous study where a mixed-defect seed is repaired and further grown under one fixed Zr6-only condition.

## What is recovered here

- The retained Zr6-only continuous case:
  `output/mixeddef800_seed800_zr6only_candidate01_continuous`
- The reference mixed-defect seed:
  `output/mixeddef800_seed800_zr6only_candidate01_continuous/references/UiO-66_R3_Mixeddef_0.40_seed800.pkl`
- One tracked repaired nucleus used in the figures:
  `output/mixeddef800_seed800_zr6only_candidate01_continuous/references/assembly_2026-04-03_15-42-13_entity_number_2800.pkl`
- The final recovered endpoint:
  `output/mixeddef800_seed800_zr6only_candidate01_continuous/assembly_final_entity_number_100001.pkl`
- Regenerated repair plots and a structure triptych:
  `repair_progress_vs_growth.{csv,png,svg}`
  `seed_initial_final_structure_triptych.{png,svg}`

This workspace intentionally keeps a curated subset of the original run. The original source case in `initialize_structures1` contains about 3.9 GB of checkpoint files, so only the key structures, logs, traces, and regenerated figures are copied here.

## Fixed growth condition

The recovered trajectory was run under one fixed chemistry:

- `ZR6_PERCENTAGE = 1.0`
- `Zr_conc = 5000.0`
- `Linker_conc = 100.0`
- `Capping_agent_conc = 200.0`
- `equilibrium_constant_coefficient = 10.0`
- `entropy_correction_coefficient = 0.789387907185137`
- `H2O_DMF_RATIO = 3e-10`
- `BUMPING_THRESHOLD = 2.0`
- `EXCHANGE_RXN_TIME_SECONDS = 0.1`

The seed used for the first segment is the mixed-defect nucleus:

- `output/mixeddef800_seed800_zr6only_candidate01_continuous/references/UiO-66_R3_Mixeddef_0.40_seed800.pkl`

The continuation to larger sizes used the same chemistry and resumed from later checkpoints of the same case.

## Main scripts

- `recover_missing_bdc_zr6.py`
  Rebuilds this curated workspace from the surviving `initialize_structures1` case, copies the retained files, regenerates the CSV, and redraws the figures.
- `analysis_utils.py`
  Local helper functions for loading assemblies, rebuilding the pristine reference, and calculating missing-site repair fractions.
- `plot_missing_bdc_repair_progress.py`
  Redraws the BDC/Zr fill curve from the local CSV.
- `plot_structure_triptych.py`
  Redraws the seed/milestone/final structure comparison.
- `run_seeded_growth.py`
  Wrapper to continue a seed or checkpoint using the local `UiO66_growth_main_conc.py` template.
- `UiO66_growth_main_conc.py`
  Local growth template. In this recovered copy the entropy lookup table scales with `max_entities`, so long continuations above 25,000 entities do not hit the old hard-coded limit.

## Recovery commands

Run from this directory:

```powershell
python .\recover_missing_bdc_zr6.py
python .\plot_missing_bdc_repair_progress.py
python .\plot_structure_triptych.py
```

If `python` resolves to a 32-bit interpreter without plotting packages, use the 64-bit interpreter directly:

```powershell
C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe .\recover_missing_bdc_zr6.py
```

## Continue growth from this workspace

Example: restart from the recovered mixed-defect seed under the same fixed chemistry into a new output folder.

```powershell
python .\run_seeded_growth.py `
  --seed-pkl .\output\mixeddef800_seed800_zr6only_candidate01_continuous\references\UiO-66_R3_Mixeddef_0.40_seed800.pkl `
  --output-dir .\output\rerun_same_condition `
  --rng-seed 930 `
  --zr6-percentage 1.0 `
  --zr-conc 5000.0 `
  --linker-conc 100.0 `
  --capping-agent-conc 200.0 `
  --equilibrium-constant-coefficient 10.0 `
  --entropy-correction-coefficient 0.789387907185137 `
  --h2o-dmf-ratio 3e-10 `
  --exchange-rxn-time-seconds 0.1 `
  --bumping-threshold 2 `
  --total-steps 500000 `
  --max-entities 6500 `
  --output-inter 100
```

For larger continuations, keep the chemistry fixed and only change `--seed-pkl`, `--rng-seed`, `--total-steps`, `--max-entities`, and `--output-inter`.
