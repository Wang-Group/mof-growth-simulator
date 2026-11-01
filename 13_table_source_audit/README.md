# Tables S1, S2, S6 and S8: Source Audit

Historical source audit dated 2026-10-31; status updated 2026-11-01.
Neither MS nor SI was edited. The latest author confirmations and NAS follow-up
are recorded in `../metadata/publication_decisions_20261031.json`.
`selected_si_tables_20261031.json` records the SHA-256 of both the original-named
SI and the October 31 text-synced SI. For these four tables their numerical
contents agree within that historical baseline. The newer S2-updated SI differs
as noted below. This is a narrow table check, not a full-manuscript review.

Word XML was read as the final view: inserted text and moved-to text retained;
deleted text, moved-from text and deleted rows excluded. Ordinary and Office Math
text were extracted in document order. The Word files are not redistributed here.

## S1: Author-Confirmed 69/53 Counts

The shared Figure 1 evidence folder supplied `s1_original/B_seed_studies_original.csv`,
its saved count JSON, the original counting caveat, upstream statistics code and
provenance note. They are preserved without editing or rerunning the archived code.

On 2026-11-01 the user confirmed that the literature had been checked and the
manuscript's **69 total / 53 Zr/Hf publication counts** are correct. Retain those
counts; the count-confirmation request is closed. Do not replace them using an
archival grouping field or repeatedly request the same confirmation.

For historical provenance only, the archived selection contains 69 DOI records and 68
publication groups. The Zr/Hf subset has 53 DOI records but 52 publication
groups. The source groups `10.1002/ange.201907074` and `10.1002/anie.201907074`
together. These unchanged source statistics are not alternative adopted counts
or a pending correction. Author confirmation is distinct from independently
re-reading all cited papers or reconstructing the final direction labels, which
this update does not claim to have done.

## S2: Confirmed 23 Of 30, MS/SI Aligned

The author confirmed that the checked study must be retained and the adopted
result is **23 smaller/thinner outcomes among 30 publications**. A read-only
final-view check of the subsequently updated NAS MS and SI found agreement on
23/30; the SI classes are 23 + 2 + 2 + 1 + 2. The count discrepancy is closed.
Do not remove that study or revert to 22 without a new author decision.

The packaged extraction totalling 31 papers and the earlier MS count of 22
are historical, not alternative adopted results. The decisions file records
the current MS/SI hashes and the author's retention decision. This confirms
document agreement; it does not claim an independent re-read of all 30 papers
or a reconstruction of their row-level classification.

## S6: All 18 Displayed Numbers Matched

The shared `MOF_growth_Data/Entropy calculation/entropy_results.csv` is byte-identical
to the existing `08_entropy_inputs/data/entropy_results.csv`. Its SBU, BTB, MOL_1,
MOL_1', MOL_2 and MOL_3 rows match all 18 translational, rotational and total
entropy values in S6 to displayed precision. The largest absolute difference
is about 4.55e-9 in the dimensionless S/R values; each source total is the sum of
its translational and rotational components.

`s6_value_comparison.csv` contains the row/column mapping and numerical differences.
That packaged comparison verifies a derived numerical source. In the separate
November 1 NAS follow-up, `calculate_entropy.ipynb`, the molecular-properties
notebook, six mapped geometries and the solvent/model inputs were recovered.
Recalculation with the notebook's rounded inputs matched all 18 SI values to
displayed precision (maximum difference 4.54604e-9). BTB uses `BTBa_new.xyz`, not
`BTBa.xyz`. Molecular moments of inertia match after the notebook's three-digit
rounding; the recorded temperature is 393 K. No batch-writing cells were run.

The reviewed sources are now packaged in `08_entropy_inputs/source/` with
original and stored hashes. Run `python tools/reproduce_entropy.py`; the same
six-species geometry and 18-value calculation check is included in
`verify_publication.py`. Only reviewed numerical helpers run, never batch cells
or in-place preprocessing. This S6 source/calculation request is complete;
it does not claim independent validation of the model's physical approximations.

## S8: Closed By Author Confirmation

On 2026-10-31 the author confirmed that the existing Table S8 is the original
recorded time table. The user accepted this clarification and explicitly closed
the source request on 2026-11-01. No additional observation sheet or separately
created spreadsheet is requested. The decision is recorded in
`../metadata/publication_decisions_20261031.json` under `TableS8`.
This confirmation supersedes earlier missing-source notes in archived material;
those historical source files retain their original bytes.

The 12 supplied acetic-acid/induction-time observations remain unchanged in the
selected-table extraction. Times are in minutes and refer to first visible
turbidity, not the in-situ DLS particle-size trace. This is author-confirmed
provenance, not independent experimental remeasurement. No extra repeats or
uncertainty estimates are inferred. S1/S2 counts are confirmed and the S6
source package and numerical reproduction are complete as described above.

The previously recovered `260316.csv` DLS trace supports the DLS fit in group 02;
it must not be relabelled as the original source for these 12 observations.

## Recompute

```bash
python tools/audit_runtime_and_tables.py
```

This checks original-source hashes, independently counts S1, compares both SI
table extracts and recalculates all S6 differences. `--write-derived` regenerates
the comparison CSV and combined audit JSON. Source bytes are not modified.

Search scope was the shared project source bundle and entropy directory,
relevant Song Yuhang MOF source/experiment folders, and bounded project inventories
on the 5090. "Not located" does not mean the sources do not exist on another
machine, in another account or in an unexported experiment notebook.
