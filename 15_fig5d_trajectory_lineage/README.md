# Fig. 5D: Recovered Trajectory-to-Motif Lineage

Recovered read-only from Jiageng on **2026-11-02**. This is the current Fig. 5D
provenance record; group 14 preserves the earlier, partial NAS/5090 search.
No manuscript, SI, classification table, structure ID or figure was changed.

## Result

- All **284 rows (71 conditions x 4 repeats)** map uniquely to processed and
  original run folders, using all eight condition/repeat keys.
- The remote classification CSV is byte-identical to the existing group 04 CSV.
- All **7,978 nonempty assignments at entity 1-29** replayed successfully against
  the recovered frozen representatives using the historical classification rule.
- The **90 representative structures at entity 1-13** already in group 04 are
  byte-identical to the remote copies. No replacement artwork is needed.
- All 284 original run directories, their runner/engine files, and the 3,240
  distinct required raw MOL2 chunks were present at the read-only check.
- Four entity-13 snapshots (one condition, repeats 0-3) were additionally compared
  with the actual blocks in their raw MOL2 streams: all four SHA-256 pairs match.
- The three NAS trajectory files with conflicting parent/filename indices match
  the original **Index_1, Index_2, Index_3** files by SHA-256. The NAS parent name
  `Index_0` is not their original repeat identity.

## Files and Lookup

`row_to_run.csv` uses one-based data-row numbers, excluding the CSV header.
`snapshot_assignments.csv` records every nonempty classification cell, its UID,
MOL2 hash, index, stored algorithm time, raw chunk and offset. To locate a source:

1. Look up the classification row in `row_to_run.csv` to obtain `folder`.
2. Select the entity in `snapshot_assignments.csv`.
3. The processed snapshot is `processed_root/folder/block_<block_index>.mol2`.
4. The raw stream is `raw_root/folder/<raw_chunk>`; `local_block_index` is zero-based.

Roots are recorded in `manifest.json` and `remote_lineage.json`:

```text
/public/home/yibinjiang/MOL_nuclear_plot_MOL_batch0_selected
/public/home/yibinjiang/workspace/out1/MOL_batch0_selected
```

The four small ZIPs contain ordinary files and can be opened without Python:

| Archive | Contents |
|---|---|
| `data/snapshots_entity01_07.zip` | Every available entity 1-7 snapshot, organized by table row |
| `data/snapshots_entity08_13.zip` | Every available entity 8-13 snapshot, organized by table row |
| `data/representatives.zip` | 352 historical class representatives covering entity 1-29 |
| `data/metadata.zip` | 284 original `level_idx.json` files, classification/extraction notebooks, 284 per-run runners, shared engine, and original morphology summary |

Together the Git payload contains **3,688 Fig. 5D snapshots**. The entity 14-29
snapshots remain in the external full recovery archive and on Jiageng; their
hashes, class assignments and source locations are still indexed here. The
external archive fingerprint is in `manifest.json`. Large trajectory pickles and
raw multi-block MOL2 streams are not copied into Git. This addition is about
34 MB including metadata, rather than the roughly 10 GB plot archive.

Within `metadata.zip`, `source/classification_test.ipynb` is the co-located
classifier reading `block_*.mol2`. The extractor notebooks retain their original
names. These differ from group 14's NAS classifier reading `entity_number*.mol2`;
neither version is silently substituted for the other. Notebook outputs and
execution metadata are removed, with original file hashes and unchanged source
cell fingerprints recorded. **Do not Run All:** archived code writes files and
uses paths and pickle inputs that are not part of the normal verifier.

## Selection and Classification Semantics

The extractor truncates each saved trajectory at its **first maximum entity
number**, then selects the **last occurrence of each size within that prefix**.
These are not first-passage structures or necessarily the last occurrence in a
longer, subsequently resumed simulation. The runner records the size before an
event and writes the structure after it; the extractor therefore uses
`block_index = level_idx - 1`. Four raw-stream spot checks confirm that indexing.
`raw_time_u` is the JSON's stored algorithm time; no physical-time rescaling is
applied in this package.

The classifier first checks Zr6/BTB counts, then compares element-wise radial
histograms (0-50 Angstrom, 50 bins) using mean cosine similarity > 0.8. It takes
the first matching class in numeric UID order. Our read-only replay uses the
recovered, frozen representatives and reproduces every recorded assignment;
it does not create new class IDs or depend on today's directory enumeration.
This is **not a graph-isomorphism or unique-topology proof**. Co-located source
and exact output replay establish reproducibility, not an unavailable historical
execution log or RNG replay. The recovered runner/engine source contains no
explicit `random.seed`/`np.random.seed` call; exact stochastic reruns are not claimed.

## Verify

From the release root:

```bash
python -m tools.audit_fig5d_lineage
```

This replays all 3,688 packaged figure-size assignments, checks all 284 row
keys, index arithmetic, source/ZIP hashes, representative identity and recorded
raw-stream checks. It does not reconnect to Jiageng. With the external full
recovery archive, replay all 7,978 assignments instead:

```bash
python -m tools.audit_fig5d_lineage --full-source-archive /path/to/classified_sources.tar.gz
```

Remaining limits: raw-stream byte comparisons were sampled, not performed for
all 7,978 blocks; complete pickle/RNG histories are not published here. Fig. 5C's
separate analysis-module gap is unaffected. S1/S2/S6/S8 are not reopened.
