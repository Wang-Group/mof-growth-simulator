# Mixed-Node Growth: April 3 Dataset

`source_archive/` preserves the 24-file April 3 source bundle, including
saved trajectories, plotting code, reference artwork and source hashes.

## Dataset Versions

| Variant | Location | SHA-256 prefix | Stored mean Zr12 first zero (s) |
|---|---|---|---:|
| April 3, Zr6_Zr12_mix | `source_archive/source_data/*.summary_trace.json` | `d7be8f68b070` | 0.701326598313 |
| April 3, initialize_structures1 | `source_archive/other_versions/initialize_structures1/mixed_old_trace_schema.json` | `f08ad2680b27` | 0.306265453269 |
| April 10 rerun | `../05_mixed_zr6_zr12/data/*rerun_20260410.summary_trace.json` | `827fa5f219fa` | 0.270600817133 |

Each contains four runs and a 400-point mean grid. Means use previous-value
sampling; completed runs retain their last value on later grid points.
The three datasets are separate histories and are not pooled.
Stored time coordinates are preserved.

## Plotting

From the repository root:

```bash
python 11_s21_april03/plot_saved_trace.py --output-dir reproduced_local/S21_april03
```

The wrapper uses the archived plotting helper and saved April 3 JSON, sets
Arial and writes only to the output folder. It does not rerun a simulation.
The root `reproduce.py --group mixed` entry instead plots the April 10 dataset.

`source_archive/plot_code/replot_saved_trace.py` is the saved-data plotting helper.
The historical `plot_zr6_only_growth_trace.py` is replay/simulation code with
external runtime dependencies; it is a different workflow.

Run `python verify_publication.py` to check source hashes and reproduce the
saved mean curves for all three datasets.
