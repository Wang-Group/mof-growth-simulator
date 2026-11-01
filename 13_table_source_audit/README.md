# Tables S1, S2, S6 and S8: Source Audit

Read-only source audit dated 2026-10-31. Neither MS nor SI was edited.
`selected_si_tables_20261031.json` records the SHA-256 of both the original-named
SI and the October 31 text-synced SI. For these four tables their numerical
contents agree. This is a narrow table check, not a new full-manuscript review.

Word XML was read as the final view: inserted text and moved-to text retained;
deleted text, moved-from text and deleted rows excluded. Ordinary and Office Math
text were extracted in document order. The Word files are not redistributed here.

## S1: Source Recovered, Counting Unit Needs Confirmation

The shared Figure 1 evidence folder supplied `s1_original/B_seed_studies_original.csv`,
its saved count JSON, the original counting caveat, upstream statistics code and
provenance note. They are preserved without editing or rerunning the archived code.

An independent count reproduces **69 dose/presence DOI records but 68 distinct
publication groups**. The Zr/Hf subset has **53 DOI records but 52 publication
groups**. The source groups `10.1002/ange.201907074` and `10.1002/anie.201907074`
together. The currently inspected SI labels 69 and 53 as publications.

Ask the authors whether the unit should be DOI records or deduplicated
publications. Do not silently replace the paper's values. The final Zr/Hf
40/12/1 directional classification still needs a DOI-level final decision table;
the historical raw direction strings are not an authoritative substitute.
This audit does not independently re-read all cited papers.

## S2: Final Selection Still Missing

The current table totals 31 papers (21 + 4 + 3 + 1 + 2). The broader 1031-record
literature catalogue is already in group 07. The exact 31-paper inclusion list
and final directional labels were not located in the inspected shared/5090
project sources. A catalogue and Word-table transcription do not close this gap.

## S6: All 18 Displayed Numbers Matched

The shared `MOF_growth_Data/Entropy calculation/entropy_results.csv` is byte-identical
to the existing `08_entropy_inputs/data/entropy_results.csv`. Its SBU, BTB, MOL_1,
MOL_1', MOL_2 and MOL_3 rows match all 18 translational, rotational and total
entropy values in S6 to displayed precision. The largest absolute difference
is about 4.55e-9 in the dimensionless S/R values; each source total is the sum of
its translational and rotational components.

`s6_value_comparison.csv` contains the row/column mapping and numerical differences.
This verifies a **derived numerical source**, not a new entropy calculation.
The remaining upstream gap is the actual computation script, geometry-to-species
mapping and solvent/model inputs. The inspected SI describes a solution-entropy
correction, so a missing Gaussian output is not automatically the right diagnosis.
Existing mass/inertia tables in group 08 do not by themselves establish that full chain.

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
uncertainty estimates are inferred. S1/S2 and S6 remain pending as described above.

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
