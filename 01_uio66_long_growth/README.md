# UiO-66 Long-Growth Curated Package

This lightweight package archives the manuscript-facing analysis of 24 repeated
UiO-66 KMC trajectories that reached `entity = 1000`. It replaces a 26.4 GB
local run directory with the exact conditions, source snapshot, compact derived
tables, entity-13 structures, and reproducible plotting code needed to audit the
scientific claims.

## Contents

- `data/first_passage_24.csv`: exact first-passage times for each repeat and
  thresholds from entity 3 through entity 1000.
- `data/residence_pre30_by_case_entity.csv`: pre-30 residence time for every
  repeat and entity number 1-29.
- `data/representative_pre30_trajectories.csv`: plotting-only downsampled traces
  for the fastest, median-rank, and slowest trajectories by time to entity 30.
- `data/entity13_first_last_decomposition_24.csv`: first/last entity-13 timing
  decomposition retained from the trajectory-level analysis.
- `data/uio66_entity13_structure_*.csv`: geometry-reconstructed structural
  classification of the 24 entity-13 snapshots.
- `structures/entity13_groups/`: all 24 entity-13 mol2 snapshots, grouped by
  reconstructed SBU-linker topology, plus one representative copy per group.
- `source/`: the exact assembly implementation used by all repeats, the
  parameterized case-I00 main-script snapshot, the common main-script template,
  batch generator/launcher, and one-row Excel condition table.
- `data/simulation_case_manifest.csv`: local raw-file provenance and source
  hashes confirming the common assembly implementation and main-script body
  across all 24 repeats.
- `scripts/export_long_growth_data.py`: regenerates compact trajectory tables
  from the local full-run directory.
- `scripts/classify_entity13_structures.py` and
  `scripts/export_entity13_structure_groups.py`: regenerate entity-13 topology
  tables and grouped structures from the local snapshots.
- `scripts/analyze_curated_long_growth.py`: reads only committed curated data
  and regenerates the SI figure, statistics table, and summary.
- `output/uio66_long_growth_figure_s19.png` and `.svg`: four-panel repeated-
  trajectory analysis for direct SI use.

## Run Conditions

The 24 repeats use one common condition:

| Parameter | Value |
|---|---:|
| Zr concentration | 22.9154706 mM |
| Capping-agent concentration | 1473.063418 mM |
| Linker concentration | 34.5798436 mM |
| Entropy correction coefficient | 0.789387907 |
| Equilibrium-constant coefficient | 1.329751726 |
| Zr6 fraction | 1 |
| H2O/DMF ratio | 0 |
| Bumping threshold | 2 |
| Maximum entity number | 1000 |
| Repeats | 24 |

The original values are preserved in
`source/UiO66_260206_long1000_stage_1.xlsx` and exported to
`data/run_conditions.csv`.

## Time Conversion

`entities_number.pkl` stores the KMC variable `timing` before multiplication by
an exchange timescale. For this manuscript-facing data set, physical time is
defined as

```text
Simulation Time (s) = stored timing x 0.175 s
```

The `0.175 s` factor is the 25 degC ligand-exchange estimate adopted for the
long-growth analysis. Changing this factor rescales all time-like parameters
linearly but does not change trajectory order, structural thresholds, survival
ordering, or dimensionless curve shape.

## Reproduce

The committed lightweight package is sufficient to regenerate the analysis:

```powershell
conda run -n base python .\UiO-66\Long_Growth_Curated\scripts\analyze_curated_long_growth.py
```

If the full local run directory is available, all compact trajectory tables and
entity-13 classifications can be rebuilt first:

```powershell
conda run -n base python .\UiO-66\Long_Growth_Curated\scripts\export_long_growth_data.py
conda run -n base python .\UiO-66\Long_Growth_Curated\scripts\classify_entity13_structures.py
conda run -n base python .\UiO-66\Long_Growth_Curated\scripts\export_entity13_structure_groups.py
conda run -n base python .\UiO-66\Long_Growth_Curated\scripts\analyze_curated_long_growth.py
```

## Interpretation and Limits

The median first-passage time rises from 0.3 s at entity 5 to 74.6 s at entity
9 and 4688.1 s at entity 13. The median intervals from entity 13 to 30 and from
entity 30 to 1000 are only about 9.0 and 21.9 s, respectively. This identifies
the pre-13 regime as the principal stochastic kinetic bottleneck.

Entity 13 is not one unique structure. The 24 last-saved entity-13 snapshots
contain nine reconstructed SBU-linker topologies; the largest class contains
11 trajectories. The mol2 snapshots contain zero bond records, so BDC-Zr6
connectivity is reconstructed from geometry with a 0.1 A matching cutoff.
Accordingly, entity 13 should be described as a recurrent motif family or an
operational growth threshold, not as a deterministic critical nucleus.

## Intentionally Omitted

The local full-run directory contains raw pickles, checkpoints, complete final
structures, and worker logs totaling approximately 26.4 GB. Those files are
excluded by `.gitignore`; no file in this package is required to reproduce the
underlying KMC simulation itself without the standard monomer inputs, but the
committed source and condition snapshots identify the exact implementation and
settings used for this batch.
