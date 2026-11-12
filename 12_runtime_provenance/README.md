# Simulation Inputs and Run Lineage

This folder records 206 source locations, including versioned code, parameter
manifests, small starting structures and parent/continuation relationships.
Repeated source versions are stored once and referenced by hash.

## Contents

- `source_manifest.json`: original paths, packaged paths, compression and hashes.
- `code_seed_version_links.csv`: 197 references to exact code, seed or frozen-input bytes.
- `checkpoint_lineage.csv`: 1,327 continuation/paired-branch rows with parent, result, RNG seed and code manifest.
- `remote_checkpoint_verification.json`: recorded hashes for 975 external checkpoints and their result JSONs.
- `../metadata/runtime_table_audit_20261031.json`: computed source and table checks.

## Calculation Families

| Family | Inputs and relationship |
|---|---|
| Frozen seed ensemble v2 | Eight UiO-66, eight unrestricted BTB-network and eight BTB-planar seeds with generating RNG and engine versions |
| Common response / seeded propagation | Design manifests with 80,640 / 46,080 rows and per-row RNG seeds |
| Competitive-bath pilots | Two 7,680-row designs and their executed code versions |
| SC1 seeded / surface campaigns | 2,688 / 1,536 design rows and BTB/UiO-66 starting structures |
| UiO-66 seeded growth | Six result JSONs linked to manifest rows, RNG seeds, N = 76 seed and executed code; rho = 10 uses replay_v2 |
| BTB high-rho / size extension | 32-row pilot and 16-row replay/extension with N = 30 seed |
| Mixed-solvent grid | 27,648-row design and 288-row pilot with model and runner code |
| Fixed-event defect production | 1,152 design rows, N = 76 seed and 37 dependency hashes |
| Defect size follow-on | 943 continuations linked to parent results and frozen versions |
| Paired response | 384 branches from 32 N = 1000 particles, with 12 branches per shared particle |

Design rows describe requested calculations. Continuations and paired branches
are not additional independent starting particles. Planar and unrestricted
BTB seeds, fixed-event endpoints and fixed-size endpoints remain distinct.

## Code Versions

The competitive-bath runner hash begins `eb6894cf`; the sampled SC1 BTB
runner uses `d4824c72` from
`code/backups/pre_n1000_20260901_2347/seeded_molecular_runner.py`.
The six UiO-66 results use `b5354813`, stored in group 07.
Full hashes are in `code_seed_version_links.csv`.

## Check and Materialize

From the repository root:

```bash
python tools/audit_runtime_and_tables.py
python tools/audit_runtime_and_tables.py --materialize
```

The first command checks local source hashes and recorded run relationships.
The second reconstructs the selected source tree under
`reproduced_local/runtime_source_tree/`, decompressing manifests without
changing their original bytes. It refuses to overwrite differing files and
does not execute the scripts. Configure input/output paths for a new run;
continuations require their external checkpoints. Load serialized starting
structures only from trusted sources.
