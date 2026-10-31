# S21 April 3 Source Package

`source_archive/` is a byte-identical copy of the 24-file shared folder
`SI_Figure_S21/April03_source_copy_20261030/`, imported on 2026-10-31.
The supplied Chinese README, verification records and artwork are preserved
unchanged. The source README identifies the `Zr6_Zr12_mix` variant as matching
S21; this integration independently verifies hashes and saved mean calculations,
but does not re-establish the mapping to the current Word document.

## Do Not Mix These Three Datasets

| Variant | Location | SHA-256 prefix | Stored mean Zr12 first zero (s) |
| --- | --- | --- | ---: |
| April 3, `Zr6_Zr12_mix` | `source_archive/source_data/*.summary_trace.json` | `d7be8f68b070` | 0.701326598313 |
| April 3, `initialize_structures1` alternative | `source_archive/other_versions/initialize_structures1/mixed_old_trace_schema.json` | `f08ad2680b27` | 0.306265453269 |
| April 10 rerun, retained unchanged | `../05_mixed_zr6_zr12/data/*rerun_20260410.summary_trace.json` | `827fa5f219fa` | 0.270600817133 |

Each contains four saved runs and a 400-point mean grid. Recomputing all three
species means by previous-value sampling reproduces every saved value exactly
in all three datasets. Completed runs retain their last value on later grid
points; they are not removed from the mean. These are distinct histories, not
twelve independent repeats that should be pooled. Stored seconds are preserved,
not newly physically calibrated.

## Code and Artwork

- `source_archive/plot_code/replot_saved_trace.py` is a plotting-only helper;
  it reads the main April 3 saved JSON and verifies its mean before plotting.
- `plot_zr6_only_growth_trace.py` is historical replay/simulation code. It
  depends on external simulation modules and should NOT be used to redraw an
  archived trace when a saved-data plot is intended.
- `original_artwork/` contains supplied original exports. `redrawn_verification/`
  contains separately labeled supplied checks, not replacements for the originals.
- `verification/` contains the supplied source manifest, mean check and image
  comparison. Its Word-image reference is a PNG, not an editable manuscript.

To redraw the main April 3 source from the repository root without simulation:

```bash
python 11_s21_april03/plot_saved_trace.py --output-dir reproduced_local/S21_april03
```

The wrapper copies only the archived plotting helper and its saved JSON into
the requested output workspace before running it. It sets Arial and leaves
all archival bytes untouched. It does not alter `reproduce.py --group mixed`,
which continues to draw the historical April 10 dataset.

Run `python verify_publication.py` for full per-file hashes and independent
mean checks of all three variants. Final S21/S22 artwork and seed/checkpoint
selection still require author confirmation; no MS/SI files were modified.
