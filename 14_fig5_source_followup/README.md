# Fig. 5 Source Follow-up

**Historical search snapshot (2026-11-01).** The Fig. 5D gap described below was
subsequently resolved by the [Jiageng recovery in group 15](../15_fig5d_trajectory_lineage/README.md)
on November 2. Keep this earlier evidence map unchanged as history; it is not the
current missing-data list. The original panel C module is still unlocated;
[group 16](../16_fig5c_reconstruction/README.md) now provides a validated,
explicitly labelled independent reconstruction, not recovery of that module.

Read-only search completed on 2026-11-01. This is **partial provenance**, not a
completed reconstruction of panels C and D. No MS/SI or original source was edited.

## Panel C

`analyze_figure6_current_parameter_phase_v1.py` was not located in the checked
shared project or four related 5090 project roots. The shared search enumerated
28,395 files and five ZIP directories. The 5090 search enumerated 806,882 files,
read 616 bounded code/text files and inspected 251,870 members in 29 archives.
No original module or definitions of `build_probability_cube` / `phase_panel`
were found in the searched code. Dependency references are present in copies of
the already packaged plotter. Search scripts, exclusions, size limits and reports
are in `search_evidence/`; this is not a claim to have searched every computer.

The supplied plotter resolves the missing module under its campaign's
`figure_rebuild_20260901/` directory. It also references the original author's
Windows figure-style kernel. The next useful source is the original plotting
workstation's campaign/rebuild folder or its backup, not another KMC rerun.
No replacement interpolation code is presented as the original.

## Panel D

- `source/unique_structures.ipynb` retains every original source cell from the
  supplied S29/S30 notebook. Outputs and execution metadata are removed.
- `source/make_entity_structure_grid.py` is a byte-identical shared-drive plotting
  script. It is archival code, not an automatically executed entry point.
- `structures/` contains nine large MOL2 assemblies from `MOL_structrue_data.zip`,
  losslessly gzip-compressed. Eight full directory parameter keys match eight
  rows of the historical classification CSV. One shortened directory name does
  not contain the full key and is not assigned to a row.
- `source_manifest.json` records original archive members, source/stored hashes
  and three external trajectory-file hashes. These three pickles total
  352,537,047 bytes and remain on the shared drive; none was deserialized.
- `row_source_map.csv` lists all 284 rows using one-based data-row IDs (header
  excluded). Eight rows have `STRUCTURE_CONDITIONS_ONLY`, four have
  `AMBIGUOUS_TRACE_INDEX`, and 272 have `NOT_LOCATED`.

The eight structure-condition matches are rows 26, 118, 119, 134, 172, 216, 249
and 252. They are **large-assembly evidence**, not the missing entity-number
snapshots used to assign the motif IDs. All 284 rows therefore retain
`classification_snapshot_link_verified=false`.

The three trajectory filenames say `index1`, `index2` and `index3`, but their
common parent says `Index_0`. The other seven parameters match rows 141-144.
All four rows are listed as candidates; neither the filename nor the parent is
silently treated as authoritative. The shortened ninth structure is retained
without assigning a row from an incomplete name.

## Why the Original Mapping Is Still Open

The supplied classification notebook reads `entity_number*.mol2` files, extracts
conditions from the parent directory, and assigns IDs in filesystem traversal
order. Its shown method checks SBU/linker counts and element-wise radial
histogram similarity greater than 0.8; this is not an exact graph-isomorphism
test. It copies representative structures and writes condition/ID records, but
the final CSV omits the original path and enumeration order. We did not establish
that this notebook revision generated the historical 284-row table.

Closing the gap requires the original 71-condition x 4-repeat small-entity MOL2
folders (or a saved per-file assignment log), plus the executed notebook version
and ordering. Large final assemblies or matching concentrations alone cannot
restore those assignments. Existing frequencies and user-confirmed artwork are
not changed by this provenance finding.

## Verify Safely

From the release root:

```bash
python -m tools.audit_fig5_sources
```

The verifier checks source hashes, lossless decompression and all 284 row statuses.
It does not access the NAS, deserialize pickle, rerun classification, draw new
figures or execute archived notebooks. Do not Run All on the original notebook:
its filesystem writes depend on the working directory.
