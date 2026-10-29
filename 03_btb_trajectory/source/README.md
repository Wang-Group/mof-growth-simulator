# Recovered Fig. 5A run-folder source

These are byte-identical copies retrieved from historical run Index_2, dated
2025-08-18_22_52_29, in the MOL_batch0_selected campaign. They are archival source,
not a ready-to-run entry point: the main script retains its historical output
directory and does not contain a saved RNG state. Do not execute it directly.

The run's `entities_number_3124.8659233585645.pkl` is byte-identical to the figure
source renamed `entities_number_3124.8659233585645_index2.pkl`:
SHA-256 `e1a0c3f933be7be169f13c594b62e48ed9e2866e7aab356ab7769bc68a93b2d4`.
The figure source had been collected in a directory labelled Index_0; that
container label is not its actual run identity.

| Source | SHA-256 |
|---|---|
| mol_growth_main_20250811.py | a07480434d679249a4a353c108fc6a503678261dddb906fa41c3efa3b42957a2 |
| MOL_Assembly_Large_Correction_20250811.py | 27799313b3a11f16a6f99da210b5e1d58d20ee9ef5d641df3518a90352fe5490 |

The main script saves raw `timing` at line 156 and advances it by
`-log(U)/total_growth_rate` at line 192. The declared `exchange_rxn_time = 0.1`
is not applied to the stored trajectory. The figure plotter in group 07 multiplies
the stored coordinate by 0.2405 and displays 620-690. Changing the origin or
applying that multiplier alone does not validate the physical model clock.
