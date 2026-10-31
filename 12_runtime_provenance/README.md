# Recovered Runtime Provenance

Read-only recovery from the 5090 project on 2026-10-31. This group records
**206 source locations**, with about 5.4 MiB of new source payload after
lossless compression and reuse of byte-identical files already in groups 07/09.
It is a provenance package, not a fresh KMC validation or a complete campaign archive.

## Start Here

- `source_manifest.json`: original project-relative paths, byte sizes, raw SHA-256,
  packaged paths, compression, stored hashes and observation times.
- `code_seed_version_links.csv`: 197 explicit references to exact packaged code,
  seed or frozen-input bytes. Repeated references are not distinct files.
- `checkpoint_lineage.csv`: 1327 continuation/paired-branch rows, each linked to
  its parent checkpoint and result, RNG seed and code manifest.
- `remote_checkpoint_verification.json`: read-only hash evidence for 975 checkpoint
  files and 975 associated result JSONs on the 5090. Their 545,387,724 bytes remain
  external; this is not a public download or a claim that they were replayed.
- `../metadata/runtime_table_audit_20261031.json`: recomputed family counts,
  seed metadata, ten executed-result matches and the remaining table-source limits.

## Families Must Remain Separate

| Family | What is now linked | Interpretation limit |
|---|---|---|
| Frozen seed ensemble v2 | 8 UiO-66, 8 unrestricted BTB-network and 8 historical BTB-planar seeds; generating RNG seed, engine/generator hash and original pickle bytes | Planar and unrestricted seeds are not interchangeable |
| Common response surface / seeded propagation | Original design manifests with 80,640 / 46,080 rows and per-row RNG seeds | Design rows are not evidence that all jobs completed |
| Competitive-bath pilots | Two 7680-row manifests; two sampled executed UiO-66 results match RNG seed, seed bytes, coordinates and caps | Not the newer explicit finite-pool defect campaign |
| SC1 seeded / surface campaigns | Remote manifests with 2688 / 1536 rows; reviewed BTB N=30, UiO-66 N=76 and host inputs; two sampled BTB results linked | Not a row-by-row completion audit of both campaigns |
| Table S7 UiO-66 growth | All six packaged result JSONs match exact manifest rows, RNG seeds, N=76 seed, coordinates, caps and executed engine/runner/selector | rho=10 uses replay_v2, not the earlier same-condition run; physical time calibration is not inferred |
| BTB high-rho / size extension | 32-row pilot, 16-row extension, reviewed N=30 seed and exact snapshot dependencies | Extension code describes deterministic replay/extension, not 16 new independent trajectories |
| Mixed-solvent grid | 27,648-row design plus 288-row pilot and their original model/runner code | Manifest/code recovery alone is not executed-version proof for every output |
| Fixed-event defect production | 1152 design rows, fresh reviewed N=76 seed, per-repeat RNG seed and all 37 manifest file hashes | Endpoint is fixed event count, not fixed size |
| Defect size follow-on | 943 continuation inputs, exact parent manifest/frozen versions, and 943 checkpoint/result pairs verified remotely | Same parent IDs, conditions and RNG seeds; not new independent repeats |
| Paired response | 384 branches from 32 N=1000 particles; branch RNG seeds and all 32 source checkpoint/result pairs verified remotely | 12 branches per shared particle; not 384 independent starting particles |

Windows/local and remote manifests, smoke tests, earlier replays, and production
versions keep their original relative paths. Do not add their row counts together.
The zero-byte legacy UiO-66 seed and quarantined Windows seed family are not used
as replacements for the recovered frozen-v2 or reviewed inputs.

## Exact Code Versions

The historical competitive-bath runner has SHA-256 beginning `eb6894cf`; the
SC1 sampled BTB runs record `d4824c72`, recovered from
`code/backups/pre_n1000_20260901_2347/seeded_molecular_runner.py`; Table S7 records
`b5354813`, already in group 07. These are different versions, not conflicting
copies of one supposedly identical file. Full hashes and their individual
references are in `code_seed_version_links.csv`.

The fixed-event manifest, size-followon manifest and paired manifest are linked
by their recorded hashes. Every file listed in those manifests has an exact-byte
copy in this package, including inherited frozen dependencies. This establishes
the recorded input chain, but does not independently establish the correctness
of all model assumptions or every reported statistical result.

## Verify Or Materialize

From the repository root:

```bash
python tools/audit_runtime_and_tables.py
python -m unittest discover -s tests -v
python verify_publication.py --report publication_verification.json
```

No network, simulation, archived-code import or pickle deserialization is used.
Local verification checks the saved remote hash evidence; it cannot prove that
the remote files are still available at a later date.

Large repetitive CSV/JSON manifests are gzip-compressed without changing their
uncompressed bytes. Files already present elsewhere in the package are reused
by exact SHA-256 rather than copied again. To reconstruct the selected source
tree for inspection:

```bash
python tools/audit_runtime_and_tables.py --materialize
```

This writes only under ignored `reproduced_local/runtime_source_tree/`, refuses
to overwrite differing files, and executes nothing. It does not download external
checkpoints or make every legacy absolute path portable. Never load an untrusted
pickle: these small seed pickles are retained for provenance and hashed only.

## Remaining Scope

Execution/checkpoint completeness for all other families, the historical
20-trajectory motif cohort and Fig. 5D row-to-trajectory mapping remain separate
questions. A full production replay and author selection of the public raw archive
are still pending. No MS/SI, remote calculation, access setting or permission was changed.
