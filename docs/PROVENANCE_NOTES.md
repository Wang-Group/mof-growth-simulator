# Version and Provenance Notes

The root README and publication checklist describe the current release state.
Dated search reports, original source notes and earlier audit JSONs are retained
as historical records, not automatically active requests for missing data.

## Figure 5

Group 16 is the release implementation for Fig. 5C: it uses the same 1260-condition
success fractions, interpolates in the documented coordinates and extracts the
P=0.5 isosurface. The default 67 x 75 x 43 mesh has 99.74% projected overlap with
the historical boundary and 99.88% overlap with the finer 133 x 149 x 85 mesh.
These are projected-region checks, not an assertion of identical 3D vertices.
The executed historical analysis module was not recovered. The independent
implementation is preserved as such; this version distinction is not a missing
input dataset or a missing reproducible analysis entry.

Group 15 supersedes the partial Fig. 5D search in group 14: all 284 rows are
linked to run folders, 7978 assignments were checked at recovery, and 3688
entity 1-13 snapshots can be reclassified offline. Historical selection takes
the last occurrence before the first maximum. Composition/radial-histogram
matching is not exact graph isomorphism. Four raw-stream spot checks are not
an exhaustive re-extraction of every saved event.

## Distinct Trajectories and Clocks

The 24 UiO-66 long-growth trajectories are separate from the historical motif
collection. Their entity-13 files are last-saved structures, not first-passage
snapshots; nine topology classes and the I03 delay exception are retained.
S7 uses raw algorithm time u, recovered from stored scaled time by division by
0.175. Historical seconds renderers remain archived, not the S7 release entry.

Fig. 3A/5A/7 reference scaling follows the current MS captions: raw timing times
0.2405 ms, based on the 120 C site-opening reference. This is not a universal
absolute growth calibration. Keep fixed-coordinate and finite-bath rate mappings
distinct. Continuations and paired branches are not new independent particles.

The current S21 bitmap matches the archived April 3 reference. Group 11 retains
that source and its separate initialization alternative. Group 05 retains an
April 10 rerun for comparison; do not pool these as one experiment or silently
replace one with the other. Current Fig. 6C uses N=2000 and four runs per condition;
group 17 and the document baseline register the adopted panel and composite.

## Documents and Verification Coverage

The current baseline records the 20261031 MS and 20261031_v2 SI as inspected on
November 2. Tracked-change final text and embedded figure sources were used for
the targeted closeout; older baselines remain nested for traceability. This is
not a claim to have re-edited or re-reviewed every sentence of the manuscripts.

S1/S2 counts and S8 record status are author confirmations. S6 numerical values,
listed first-passage statistics and other checks have separate computational
evidence. S4 linewidths are author/software measurements with recovered spectra
and row mappings, not independently refitted linewidths. Per-table verification
coverage is retained in `metadata/table_source_limits.csv`.

The aggregate catalogue export `all_catalogued_calculation_families.csv` named
under Table S25 was not located in the packaged sources or the November 2 search
of the shared `MOF_growth_Data` tree. This is a specific catalogue-file coverage
limit, not evidence that the separately mapped trajectories are absent. No
replacement catalogue has been invented or presented as the historical original.

The final two publication actions are an external large-data deposit and a
versioned release reference. A source path/hash identifies data but is not proof
that a dataset has already been uploaded to Zenodo.
