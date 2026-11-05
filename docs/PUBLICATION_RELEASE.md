# Publication Preparation

Updated 2026-11-05. The current publication branch contains the recovered
code, compact data and source mappings. MIT for original code and CC BY 4.0 for
project-produced data are confirmed; see [LICENSING.md](../LICENSING.md).
MS/SI are neither edited nor included in this branch.

This checklist is organized by scientific content. Figure/table numbers in
archived paths or audits identify a dated manuscript version, not a requirement
to keep the manuscript layout fixed. See [cross-reference policy](MANUSCRIPT_CROSSWALK.md).

## Current Source Status

| Item | Status and evidence |
|---|---|
| Literature classifications | Author-confirmed 69/53 and 23/30; older extraction counts are not substituted for the adopted counting scope |
| Ring-closure entropy | Six species, source code and geometries; 18 values independently recalculated |
| Turbidity appearance times | Author-confirmed original 12-row time record; closed, not a request for another experiment |
| Controlled modulator-geometry comparisons | [Portable reanalysis](S20_REPRODUCTION.md) from 27,648 trajectory records; nine matched results checked; all 864 groups summarized |
| BTB variance and kinetic boundary | Same 1260-condition inputs; variance reproduced; group 16 supplies the validated interpolation/isosurface implementation and grid checks |
| BTB motif lineage | All 284 rows linked to historical run folders; classification and entity 1-13 snapshots in group 15 |
| Runtime dependencies | Executed code/seed versions and checkpoint lineage in group 12; large payloads selected separately for external archiving |
| BTB finite-cluster scattering | N=2000 and four trajectories per condition; reviewed artwork snapshot recorded and reference panel in group 17 |
| Ligand-exchange NMR | Ten rows mapped to selected acquisitions; 90 acquisition/processed/export files packaged, including acquisition 2 at 268 K |
| Modulator-identity comparisons | Three original CSVs packaged with hashes; seven regression comparisons independently checked |
| Time and source definitions | Reference-scale historical plots and raw-u UiO-66 ensemble entry documented; versions are kept separate rather than overwritten |

[publication_verification.json](../publication_verification.json) records the
checks actually run. Its `publication_ready: false` concerns unfinished formal
publication actions below, not a newly identified numerical failure.

## Remaining Publication Actions

- [ ] Select and deposit necessary external raw trajectories/checkpoints, preserving
  their recorded hashes and the distinction between independent runs and branches.
  Record the returned Zenodo DOI. See [archive plan](ZENODO_DATA_PLAN.md).
- [ ] Freeze a formal GitHub release/tag and cite that exact version and dataset DOI
  in the final paper. Do not present an unassigned DOI or an uncreated tag as published.

Licensing is settled. The literature counts, entropy sources, turbidity records,
BTB motif lineage, NMR acquisition selection and matched-size scattering source
registration are not open data requests.
Final Word editing and submission remain with the manuscript editors.

## Reproduction Scope

Use the commands in the root README for offline analyses. Archived production
scripts retain their original versions and may require externally archived
checkpoints. Local verification does not rerun every KMC campaign or every
bootstrap interval. Source coverage details remain in
[table_source_limits.csv](../metadata/table_source_limits.csv), not a blanket
claim that all published analyses were independently repeated. The table labels
in that audit ledger belong to its recorded document snapshots, not the latest draft.

Historical search notes and older document baselines are provenance records, not
current missing lists; see [PROVENANCE_NOTES.md](PROVENANCE_NOTES.md).
The default branch `publication-release-20261029` retains independent history and
does not replace the original `main`.
