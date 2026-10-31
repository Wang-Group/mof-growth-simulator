# N >= 1000 exact-checkpoint follow-on

Launched 2026-09-14, approximately 13:01 Taipei, following the author's explicit request to start now rather than wait for all previous jobs. Supervisor PID 822481 on 5090.

Remote directory: `/home/wave/MOL_growth_recalculation_20260830/defect_N1000_followon_20260914_v1`.

Manifest SHA-256: `360fa9d18433fc8e316139538f790678ee7fe226647a6400e8c5348be5af13b0`. Retrieved local manifest matches this hash.

Selection covers all 1152 original trajectory IDs: 943 settled terminal samples below N1000 selected for extension, 200 already-large endpoints retained unchanged, nine unsettled peer trajectories deferred. The deferred IDs must be reconciled after their current run terminates, never resumed from an older endpoint. Selection and source-result hashes are in dataset_source_index.json. Every selected source checkpoint was SHA-256 checked against its snapshot manifest.

N counts retained Zr6 cores plus BDC entities, excluding caps and solvent proxies. Stop at the first N >= 1000 in this extension, not a historical maximum. The original cumulative 50000-event stop is removed. Preserve chemistry, geometry, pools, RNG state, model time, cumulative events and original checkpoint provenance. No new seed construction is called by the production entrypoint. All old results remain untouched.

Finite budget: 86400 additional wall seconds per trace; periodic checkpoints every 1800 seconds. Eight initial growth workers, ramp after 120 seconds up to 64 minus the active peer growth workers, subject to 24 GiB reserve and observed-memory guards. Analysis capacity subtracts peer active analyses from the two-worker allowance at each dispatch. Since the peer supervisor independently dispatches analyses, short overlaps in analyzer concurrency remain possible; growth-worker sharing is conservative because the peer growth queue has no unsubmitted work. No automatic retries of capped samples.

Tests in verification/restart_tests.json passed: real N997 / 50000-event source, direct advancement and output-enabled continuation match after 1089 events to N998, post-checkpoint continuation matches another ten events, Python/NumPy/event RNG states match, already-large stop adds zero events, zero wall budget is correctly censored, source hashes unchanged. Read-only analyzer test completed for the new terminal checkpoint (verification/analysis_tests.json).

At 13:01:39 Taipei, 13 traces submitted, eight running, five already reached exactly N1000 at cumulative events 50009, 50021, 50249, 50257 and 50319. No failures or launch hold. This is a live campaign; consult remote status.json for current counts.

The original fixed-event/censored endpoint dataset and this unequal-time size-first-passage dataset are separate. Below-target capped or no-event results are retained. N1000 does not itself establish adequate interior-site eligibility. The existing read-only filling analysis evaluates terminal snapshots and sensitivity. Large checkpoints remain remote.
