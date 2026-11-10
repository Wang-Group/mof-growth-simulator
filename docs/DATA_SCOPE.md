# Code and Necessary Data Only

Policy updated November 10, 2026. GitHub contains code and compact inputs needed
to run the documented analyses, not a full raw simulation archive.

## Retained

- Simulation engines, runners, selectors, analysis and plotting code, including
  historical executed versions and notebook source cells.
- Molecular models, parameter manifests, and the 28 audited small seed/host
  inputs. These start calculations; they are not production checkpoints.
- Compact per-run terminal statistics, condition summaries, first-passage and
  residence tables, and plot-ready growth curves. Per-run endpoints are needed
  to recompute statistics and censoring; they are not complete event histories.
- Representative structures used for classification/interpretation, entropy
  calculation geometries, and four small classifier regression examples.
- Experimental DLS, NMR and other source tables. This cleanup concerns raw
  theoretical outputs, not experimental observations.
- Source hashes, row/run indices and recorded verification results.

Some original source bundles contain copies of the same compact inputs. These
are retained where archived scripts or source mappings use those exact paths;
no calculation is presented as an independent replicate merely because a file
appears in more than one bundle.

## Excluded

Full event-by-event trajectories, production checkpoints, raw multi-block MOL2
streams and bulk per-run structure collections do not belong in the current
publication tree. Large raw pickles/checkpoints were already external.

The November 10 cleanup removes two batch snapshot ZIPs (3,688 structures) and
nine superseded large-assembly search outputs. All 11 exact files were copied
and hash-verified outside the publication worktree before removal. The retained
four classifier examples total 158,394 bytes and are linked to previously checked
raw-stream blocks. The exclusion list and hashes are in
[`data_retention.json`](../metadata/data_retention.json).

## What Can Be Reproduced

The normal lightweight plotters, entropy calculation, controlled-geometry
statistics, kinetic-boundary analysis and experimental fits still use packaged
inputs. The BTB motif audit verifies all 284 row/run mappings and 7,978 recorded
assignments/index relationships, but recalculates geometric classification only
for the four packaged regression examples. It does not claim a full-cohort
geometric replay without the external archive. The existing
`--full-source-archive` option retains that capability.

The original recovery manifests are frozen historical inventories. Entries for
excluded payloads are not missing-file errors: the explicit retention policy
matches their original hashes and the verifier rejects reintroduced bulk files.
An unlisted missing input still fails verification.

## Publication and History

No raw simulation deposit is required to run the documented lightweight entries.
An additional repository/archive for bulk outputs is a separate team decision,
subject to the final data-availability statement and journal requirements; this
policy does not establish compliance with a journal's requirements.

This is a normal deletion from the current branch, not a Git-history rewrite.
Earlier commits still contain the previously published files. Original research
copies on local/shared/server storage are unchanged. No Release, tag or Zenodo
deposit was created by this cleanup.
