# Author-requested replacement: 50,000 event steps, one-hour cap

The author changed the simulation endpoint on 13 September 2026: finish after **50,000 event attempts**, not at N=5000, and use a **3,600-second wall-time cap per trajectory**. N remains an output. There is no N-based stopping condition, including when N exceeds 5000.

## Unchanged model and grid

36 conditions × 32 replicate seeds = 1,152 freshly started trajectories. Each starts from the reviewed N=76 seed with the same condition-specific inputs and replicate random seeds as the earlier campaign. These are not continuations from the stopped intermediate configurations.

- Intermediate penalty delta = U_D/(RT): 0, 2, 4, 6, 8, 10.
- Per-bond removal coefficient rho0: 0.1, 1, 3.
- Fixed versus finite/depleting precursor pool.
- s_close = 1; solvent-cap prior = 0.5; acetate modulator.
- Initial free inventory: 1000 Zr6 entities and 6000 BDC molecules.
- Keep all twelve connection directions, explicit BDC aromatic hydrogens and acetate methyl hydrogens, the tested intermediate treatment, and the retained-oligomer inventory boundary.

The molecular engine and event-selection code are byte-for-byte copies of the tested release. Only run termination and output bookkeeping change. No extra energetic or geometric rule has been added.

## Endpoint and saved results

`step_target_reached` means exactly 50,000 attempts were executed. Attempts include accepted and rejected proposals, bond removal, internal closure, and acetate/solvent exchange; they are not a count of additions. `wall_time_cap` means the trajectory did not finish 50,000 attempts within one hour. Checkpoint/export finalization may require a short time after the growth loop stops.

Save restartable structures at steps 1000, 5000, 10000, 25000 and 50000, at size first passages N=250, 500, 1000, 2000 and 5000, and at termination. Small curves record every 100 attempts. The central-Zr6 filling analysis and boundary/tolerance sensitivity run independently after each trajectory ends.

Equal event counts do not imply equal model time. Always report model time tau, final N, maximum N, step-target completion, and eligible interior-site denominators alongside defect counts. A trajectory with insufficient interior coverage is unresolved for an interior-defect fraction, not necessarily defect-free.

## Resources

Use eight initial growth workers, increasing to at most 64 with the same memory checks and 24 GiB launch reserve, plus at most two separate analysis workers. The queue cannot exceed the user's prior 96-worker ceiling. No changes are made to other projects on 5090.

## Preservation of the old campaign

The supervisor and all 64 active growth processes in `defect_N5000_production_20260913_v2` were stopped; its unstarted queue was cancelled. All 128 available trajectory results remain: four N5000 completions, 60 four-hour-capped trajectories, and 64 trajectories terminated by this explicit user stop. Their raw `resource_stop` labels reflect the prior SIGTERM handler, not a resource failure; `USER_STOP_FINAL.json` records this distinction. No files were deleted. No old and new endpoints are silently merged.

Remote new campaign: `/home/wave/MOL_growth_recalculation_20260830/defect_50000steps_production_20260913_v1` on `5090`.
