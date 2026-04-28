# mixeddef800_seed800_zr6only_candidate01_continuous

Recovered same-condition Zr6-only growth case for the Missing_BDC study.

## Key points

- Reference mixed-defect seed:
  `references/UiO-66_R3_Mixeddef_0.40_seed800.pkl`
  2124 total entities, 1857 BDC, 267 Zr
- Tracked repaired nucleus:
  `references/assembly_2026-04-03_15-42-13_entity_number_2800.pkl`
  2800 total entities, 2153 BDC, 647 Zr
  `BDC fill = 0.2233502538071066`
  `Zr fill = 1.0`
- Final recovered endpoint:
  `assembly_final_entity_number_100001.pkl`
  100001 total entities, 77614 BDC, 22387 Zr
  `BDC fill = 0.988155668358714`
  `Zr fill = 1.0`

## Files in this curated recovery

- `repair_progress_vs_growth.csv`
  Recovered repair fractions for the first saved checkpoint at each entity count in the original continuous run.
- `repair_progress_vs_growth.png`
- `repair_progress_vs_growth.svg`
- `seed_initial_final_structure_triptych.png`
- `seed_initial_final_structure_triptych.svg`
- `recovery_summary.json`
  Provenance, fixed chemistry, copied files, and regenerated outputs.
- `segment01_seed_to6500.log` to `segment07_resume_to100000.log`
  Saved execution logs for the retained run segments.

## Fixed chemistry

- `ZR6_PERCENTAGE = 1.0`
- `Zr_conc = 5000.0`
- `Linker_conc = 100.0`
- `Capping_agent_conc = 200.0`
- `equilibrium_constant_coefficient = 10.0`
- `entropy_correction_coefficient = 0.789387907185137`
- `H2O_DMF_RATIO = 3e-10`
- `BUMPING_THRESHOLD = 2.0`
- `EXCHANGE_RXN_TIME_SECONDS = 0.1`

## Note

This folder is a curated recovery, not a full raw-checkpoint mirror. The original source trajectory remains under:

`..\..\initialize_structures1\output\mixeddef800_seed800_zr6only_candidate01_continuous`
