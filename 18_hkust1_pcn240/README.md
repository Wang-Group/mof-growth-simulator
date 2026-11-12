# HKUST-1 and PCN-240 Growth

Framework-specific kinetic Monte Carlo engines, growth runners and molecular
models for HKUST-1 and PCN-240. The four Python sources and five MOL files are
preserved unchanged from repository commit
`946020282ce93132029f455d158785838cd82780`.

## Files

| System | Runner | Assembly engine | Node and linker models |
|---|---|---|---|
| HKUST-1 | `HKUST-1/HKUST_1_growth_main_conc.py` | `HKUST-1/HKUST_1_Assembly_Large_Correction_conc.py` | `Cu2.mol`, `BTC.mol` |
| PCN-240 | `PCN-240/PCN_240_growth_main_conc.py` | `PCN-240/PCN_240_Assembly_Large_Correction_conc.py` | `Fe_Co3.mol`, `BDC_OH.mol` |

Models are in `KMC_example/input_monomers/`. Both engines also load
`Zr12_AA.mol` during initialization. The shared class names `Zr6_AA` and `BDC`
are retained from the original implementation; their actual geometry is given
by the system-specific models above.

## Running a Calculation

From the repository root, install the simulation dependencies:

```bash
python -m pip install -r 18_hkust1_pcn240/requirements.txt
```

For a new calculation, work on a copy of the selected runner and engine in an
output directory, keeping both Python files together. Set these local paths:

1. In the engine's `PRELOADED_DATA` dictionary, replace the three absolute model
   paths with paths to the included MOL files.
2. In the runner, set `current_folder` to a new output directory. Keep
   `pkl_path = None` to start a new assembly.
3. Set the desired parameters and stopping settings at the top of the runner,
   then execute that runner with Python.

The archived settings are:

| Parameter | HKUST-1 | PCN-240 |
|---|---:|---:|
| `Zr_conc` (metal concentration, mM) | 22.9154705859894 | 22.9154705859894 |
| `Linker_conc` (mM) | 34.57984361265395 | 34.57984361265395 |
| `Capping_agent_conc` (mM) | 1473.06341756944 | 200 |
| `entropy_correction_coefficient` | 0.789387907185137 | 0.789387907185137 |
| `equilibrium_constant_coefficient` | 1.32975172557788 | 1.32975172557788 |
| `H2O_DMF_RATIO` | 0 | 0 |
| `max_entities` | 200 | 500 |

`Total_steps` is set to 10^15 in the archived runners; choose a suitable finite
cycle budget for a new run. The original stopping test exits when entity count
exceeds `max_entities`. The runners do not fix a random seed; set both Python's
`random` seed and NumPy's random seed in a working copy when needed.

`entities_number.pkl` records the algorithm time and entity count. The saved
time is not multiplied by `exchange_rxn_time`; physical-time scaling is a
separate plotting operation. The `last_by_n/` directory stores the last saved
MOL2 structure at each visited entity count.

Source commit, file sizes and SHA-256 hashes are in `source_manifest.json`.
Code and model licensing follow the repository's [licensing guide](../LICENSING.md).
