# Data Versions and Conventions

Use the [topic-based code index](CODE_INDEX.md) to locate calculations.
Original source versions and file hashes are recorded in the group manifests.

## BTB Boundary, Sensitivity and Structures

Group 16 contains the original combined analysis module and a separate portable
boundary renderer. Both use the same 1,260-condition success fractions.
The portable renderer interpolates in the documented coordinates and extracts
the P = 0.5 isosurface. Its 67 x 75 x 43 mesh has 99.74% projected overlap with
the historical boundary and 99.88% overlap with the finer 133 x 149 x 85 mesh.
These are projected-region comparisons, not identical 3D vertices.

`tools/reproduce_btb_sensitivity.py` uses the original bootstrap functions.
The six sensitivity indices and twelve confidence bounds reproduce the saved table.

Group 15 provides all 284 motif-table row/run links and 7,978 recorded assignments.
Four small classifier examples and the reference structures are included;
the full cohort is external. Snapshot selection takes the last occurrence
before the first maximum. Classification uses composition and radial-histogram
similarity, rather than exact graph isomorphism.

## Trajectories and Time

The 24 UiO-66 long-growth trajectories are separate from the historical motif
collection. Their entity-13 files are last-saved structures, not first-passage
snapshots; the analysis distinguishes nine topology classes and the I03 delay
exception. The ensemble plots use raw algorithm time `u`, recovered by dividing
stored scaled time by 0.175.

Historical representative UiO-66, BTB, HKUST-1 and PCN-240 plots use a
0.2405 ms factor based on the 120 C site-opening reference. This is a reference
timescale, not a universal absolute growth calibration. Fixed-coordinate and
finite-bath rate mappings are separate. Continuations and paired branches
share their recorded parent particle.

Group 11 contains the April 3 mixed-node dataset and its initialization
alternative; group 05 contains the separate April 10 dataset. The combined
renderer uses the latter. The matched-size BTB scattering comparison uses
N = 2000 and four trajectories per condition.

## Reference Data

Literature counts use the adopted 69 total / 53 Zr-Hf DOI records and 23
smaller-size outcomes among 30 modulator studies. Historical grouping fields
and table extracts retain their original definitions and dates.

The twelve turbidity observations are in minutes and are distinct from the
DLS size trace. NMR linewidths are the supplied spectrum measurements;
the code checks their spectrum mapping and exchange-rate arithmetic.

[Analysis conventions](../metadata/analysis_conventions.json) records the
time coordinate, adopted literature counts, turbidity units and licenses.
