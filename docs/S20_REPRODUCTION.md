# Controlled Modulator-Geometry Comparison

Recompute assembly and seeded-growth outcomes for twelve monocarboxylate
geometries under matched chemical parameters. The script and output names retain
the historical `s20` identifier for compatibility, not as a dependency on the
table number in a later manuscript.

Run from the repository root with Python 3.11 or newer. This entry uses only the
Python standard library and packaged files, without network access or KMC runs:

```bash
python tools/reproduce_s20.py
```

The script returns a nonzero exit status if source hashes, cohort completeness,
endpoint definitions, saved reference values or the dated display snapshot disagree.
It is also included in `python verify_publication.py` and the full unit-test suite.

## Inputs and Scope

The inputs are under
`07_publication_sources/SI_tables_sources_20261023/Table_S20/matched_source_files/`:

- `source_trace_summary.csv`: 27,648 per-trajectory terminal records used for
  all calculations. These are individual simulation outcomes, not a previously
  aggregated condition table or the full per-event structural histories.
- `steric_matched_nine_conditions.csv`: saved high-precision results, used only
  as a comparison target, never as an input to the recomputed statistics.

Their SHA-256 values and the nine displayed SI rows are recorded in
[`metadata/s20_reference.json`](../metadata/s20_reference.json). The Table S20
caption and rows were read in final view on November 3, 2026, including insertions
and excluding deleted/moved-from text and deleted rows. This targeted check does
not update the whole-document baseline or edit/include MS/SI.

The source grid has 12 molecular identities, two assays, four `rho_off` levels
(0.1, 1, 3, 10), three atomic Zr/BDC ratios (0.3, 1, 3), and three `eta_S` values
(0.1, 0.5, 0.9). Each of the 864 groups must contain exactly one record for each
replicate 0-31. Run IDs must also be unique. Unexpected statuses, invalid physical
outcomes, nonfinite times and inconsistent endpoints are rejected.

The nine-condition subset selects benzoate, 4-OMe and 4-tBu at `rho_off = q = 1`
across the three `eta_S` values. The archived protocol specifies `s_close = 1`; this fixed parameter is
not independently encoded as a column in the terminal-record CSV.

## Statistics

Assembly uses the source assay label `nucleation`, starting at N=1 and stopping
at N=30 or model time 100. Seeded growth starts at N=76 and stops at N=126 or
model time 20. No time conversion is applied to the source `model_time` column.

- Target fraction is `k/n`, where `n=32` includes observation-window endpoints.
  Nonattainment is not a failed computation or evidence of zero eventual growth.
- Pointwise 95% Wilson bounds use `z=1.959963984540054`, without continuity
  correction: center `(p + z^2/(2n))/(1 + z^2/n)` and half-width
  `z*sqrt(p*(1-p)/n + z^2/(4n^2))/(1 + z^2/n)`, clipped to [0,1].
- Passage times are medians among `target_reached` trajectories only. The
  conditional growth measure is `50/median(time)`, not `median(50/time)`.
- No successful trajectory means undefined passage time and conditional growth
  measure, stored as empty numeric CSV cells, not zero. The display table uses
  `Not reached`. The target fraction and Wilson interval remain defined.

All 864 groups are recomputed. The nine matched rows are compared to the saved
reference using exact count checks and an absolute tolerance of 1e-12 for numeric
statistics. Fractions and three-decimal time strings must match the frozen
November 3 reference exactly; this does not assert the numbering of a later SI.
This is a new portable reanalysis entry, not recovery of an original aggregation
script or an independent validation of the KMC physical model.

## Outputs

Default destination: `reproduced_local/S20/` (ignored by Git).

| Output | Content |
|---|---|
| `s20_all_conditions.csv` | 864 recomputed assay-condition rows with fractions, Wilson bounds and conditional passage times |
| `s20_recalculated.csv` | Nine matched rows at full precision, with the saved-reference column convention |
| `s20_display.csv` | Nine rows formatted as in the reviewed SI table |
| `s20_verification.json` | Source hashes, comparisons, counts, definitions and verification scope |

An alternative output directory may be supplied with `--output-dir`. Inside the
repository, outputs are restricted to `reproduced_local/` to protect the sources.
No older intermediate CSV filenames are required. File-level details belong in
this repository guide; the scientific SI can cite the repository as a whole.
