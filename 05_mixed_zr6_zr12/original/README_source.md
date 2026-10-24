# Zr6_Zr12_mix

This folder is a clean recovery workspace for the mixed `Zr6/Zr12` UiO-66 growth discussion.

Everything needed for the recovered reproducible workflow is kept here:

- seed builder
- growth runner
- trace plotter
- copied core assembly code
- regenerated outputs
- historical trace images copied as reference only

## Working Directory

Use this folder itself as the working directory:

- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\Zr6_Zr12_mix`

## Main Entry Point

One command rebuilds the current reproducible result set:

```powershell
python recover_mix_results.py
```

That command will:

1. regenerate the mixed seed `ratio_controlled_mixed_seed_160_seed612_frac055_fill6_regenerated`
2. rerun the fixed-chemistry `Zr6-only` growth case
3. regenerate the trajectory trace with time-aligned averaging
4. copy the remaining historical trace images from `initialize_structures1` as reference

## Core Scripts

- `UiO66_Assembly_Large_Correction_conc.py`
- `build_internal_zr12_seed.py`
- `build_ratio_controlled_mixed_seed.py`
- `fragment_cleanup.py`
- `probe_zr6_only_growth.py`
- `run_zr6_only_growth_case.py`
- `plot_zr6_only_growth_trace.py`
- `recover_mix_results.py`

## Output Layout

Recovered outputs are written under:

- `output/mixed_nuclei/`

Historical reference-only images are copied under:

- `output/historical_reference/`

The recovery manifest is written to:

- `output/recovery_manifest.json`

## Current Recovered Result

The workspace has already been regenerated once and currently contains:

- regenerated mixed seed counts: `Zr6_AA = 37`, `Zr12_AA = 37`, `BDC = 98`, `total_entities = 172`
- growth rerun replicates: `4`
- all recovered reruns reached `final Zr12_AA = 0`
- mean `delta_zr6 = +253.5`
- mean `delta_zr12 = -37.0`
- mean `delta_bdc = +384.5`
- mean `final_total_entities = 773.0`

Useful output files:

- `output/mixed_nuclei/ratio_controlled_mixed_seed_160_seed612_frac055_fill6_regenerated.mol2`
- `output/mixed_nuclei/zr6_only_growth_runs/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned.summary.json`
- `output/mixed_nuclei/zr6_only_growth_runs/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned.summary_trace.svg`
- `output/mixed_nuclei/zr6_only_growth_runs/oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned/ratio_controlled_mixed_seed_160_seed612_frac055_fill6_regenerated__zr6only_opt__rep03__seed2026040313.mol2`
- `output/historical_reference/oneshot_seed160_eq2_cap180_zr5000_pruned.summary_trace.svg`
- `output/historical_reference/oneshot_seed160_eq2_cap180_zr5000_pruned.summary_trace.png`

## Notes

- The exact original older seed `ratio_controlled_mixed_seed_160_seed612_frac055_fill6` is not recoverable from the current repo state.
- The reproducible path in this workspace therefore uses the regenerated seed with suffix `_regenerated`.
- The trace plotter in this workspace uses time-aligned averaging by default, not step-aligned averaging.
