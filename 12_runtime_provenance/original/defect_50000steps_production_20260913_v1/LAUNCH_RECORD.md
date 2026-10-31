# 50,000-step production restart

**Launched on 5090 at 04:34:19 Taiwan time, 13 September 2026.** Supervisor PID: 339545. Eight actual molecular worker processes were verified after launch. The guarded queue can expand to 64 growth workers, with up to two independent structure-analysis workers.

The new stopping rule is **exactly 50,000 attempted KMC events**, or a **3,600-second wall-time cap** if the step target has not yet been reached. There is **no N-based stopping condition**. N=5000, when crossed, is only a saved size milestone. Short checkpoint/export finalization is separate from the growth-loop wall cap.

The original 36-condition, 32-replicate grid is restarted from the same reviewed N76 seed, with identical replicate seeds, molecular models and event rules. The changed endpoint is not implemented by resuming partly completed old runs or reclassifying old outcomes.

Frozen manifest SHA-256: `a033a62f3c6ec3669748d39941e04cc3a8933f14b045117f8f8eacab6efa7674`.

## Verified checks

Eight wrapper checks passed, including exact step stopping, no termination at a deliberately tiny test N, wall-cap censoring, exact restart, and agreement of the complete output-enabled wrapper with 400 unobserved molecular events. All six saved test snapshots passed the independent central-filling analysis.

## Previous campaign

The old supervisor and all growth/analysis processes were confirmed absent before the restart. Its 1,024 unstarted trajectories were cancelled. All 128 already started trajectories have retained result files: 4 reached N5000, 60 hit the four-hour cap, and 64 ended under this user-requested stop. The latter retain the old handler's raw `resource_stop` label, explicitly identified as user cancellation in `USER_STOP_FINAL.json`, not as resource failure. No files were deleted.

## Locations

New remote root: `/home/wave/MOL_growth_recalculation_20260830/defect_50000steps_production_20260913_v1` on `5090`.

`status.json` is live queue state; `outcomes.json` records completed/capped trajectories. `traces/<trace>/snapshots/steps050000/` holds the completed 50,000-step structures; each trace also has a terminal snapshot, size snapshots, small entity-count curves and independent filling analysis. Model time, final N and maximum N accompany all event counts. Equal step count is not equal physical/model time.

`RESTART_PLAN.md` documents the complete unchanged grid and the new endpoints. Large coordinate/checkpoint data remain on 5090.
