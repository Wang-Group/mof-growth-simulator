# Verified 5090 launch

User authorization: follow-up calculations that are needed to support conclusions may be designed and launched directly on 5090. This campaign addresses the unmatched penalty/pool comparisons identified during manuscript v35 revision. It does not change the manuscript interpretation before results are available.

## Active production

- Remote directory: `/home/wave/MOL_growth_recalculation_20260830/defect_paired_response_20260916_v2`
- Supervisor PID: **3835966**.
- Started: **2026-09-15 21:13:21 UTC**, equivalent to **16 September 2026 05:13:21 Asia/Taipei**.
- Manifest SHA-256: `70c5fd17efec1c2cc8d29b55ea7f0bec299ce76f8f5aaa8749579667d8c9ebcb`.
- Confirmed after launch: 8 production workers active, 376 queued, no completed or failed production results yet, approximately 108.8 GiB available memory. The first active trajectory had N=1086 after 631 events and elapsed model time 0.2133. This is an operational check, not a scientific result.
- Queue ramps after two minutes, sharing a **64-growth-worker ceiling** with the retained pilot. Each numerical library is restricted to one thread. At most two shared measurement workers.
- Molecular execution uses the existing validated `/usr/bin/python3` environment on 5090. The separate existing SciPy analysis environment handles lattice filling. Local `pytorch_cuda12` was used only for syntax/AST checks; no molecular trajectory ran locally.

## Tests and source preservation

All 12 branch conditions passed initial state consistency, exact time-boundary sampling, and checkpoint replay tests. The saved snapshots passed the read-only filling-analysis pipeline. All 12 conditions also passed 1,000 molecular events and mass-balance checks in v1; AST comparison verified that v2's `initialize`, `worker`, `signature`, `setup`, `tests` and `analyze_trace` functions are identical to those tested functions. V2 additionally reran its own branch/replay/measurement tests under its new manifest. The only campaign changes are earlier observation times, extra file hashes, and sharing worker slots with the pilot.

Source checkpoints for all 32 starting particles and the complete frozen parent code/input inventory are hash-verified. No frozen molecular rules were changed. Remote/local hashes match:

- `campaign.py`: `6b07ad1eb2e153ba9af7d8a5c42fff80c09c508c1d17b5a4b6ebec824a56bc6c`
- `summarize.py`: `2af546e3fb121e1156780430888be5b1f025e4d883bfb2e17f8f48599748a393`

## Pilot retained separately

The v1 queue at `../defect_paired_response_20260916_v1` was held after exactly eight starts. Its 376 unsubmitted entries are not production results. Active pilot trajectories are allowed to reach their finite stopping bounds; their structures, logs and analyses are preserved. They are not additional independent replicates, because v2 uses the same source states and random-seed assignments. Production summaries read v2 only.

## Readout and limitations

See `PLAN.md` for factorial conditions, primary times, censoring rules and paired inference. This is a parameter-response test on established N=1000 crystals. It is not de novo defect formation or an explicit-geometry ablation. The initial state is deliberately not assumed to be equilibrated at the changed parameters.

Each completed trajectory automatically enters read-only lattice-filling analysis. Run `summarize.py` in this remote directory to compile outcome accounting, defect fractions, shared-position filling/retention, radial profiles and paired penalty contrasts. The summarizer is syntax-checked; complete-cohort results do not yet exist. No new monitor or recurring automation was created, and the existing manuscript/SI and student ZIP were left unchanged.
