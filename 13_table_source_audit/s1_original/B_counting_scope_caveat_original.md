# Starting-cohort corrections found before condition extraction

These corrections are applied to the audit tables. They are not material for the manuscript narrative.

## DOI records versus distinct publications

The automated starting set contains 69 dose/presence DOI records. `10.1002/ange.201907074` and `10.1002/anie.201907074` are the German and International Edition records of the same article and experiment. They remain linked to preserve both citations, but they form one publication group. The starting set therefore contains 68 distinct publication groups before further duplicate-report adjudication.

## Resolved framework assignments

| DOI | Framework | Resolved family | Basis |
|---|---|---|---|
| [10.1016/j.micromeso.2020.110215](https://doi.org/10.1016/j.micromeso.2020.110215) | CID-4, Cd(bpndc)(bpy) | Cd-carboxylate | The node metal is Cd. |
| [10.1016/j.micromeso.2022.112379](https://doi.org/10.1016/j.micromeso.2022.112379) | 2D CuBDC | Cu-carboxylate | The node metal is Cu. |
| [10.1016/S1872-2067(20)63625-7](https://doi.org/10.1016/S1872-2067(20)63625-7) | MIL-53(Fe) | Fe-carboxylate | The node metal is Fe. |
| [10.1038/s41467-018-03979-2](https://doi.org/10.1038/s41467-018-03979-2) | DUT-49, Cu2(BBCDC) | Cu-carboxylate | The framework paddlewheel metal is Cu. |

No unresolved framework-family records remain in the 69-record starting set.

## Node metal separated from linker-bound and dopant metals

[10.1002/smll.202301933](https://doi.org/10.1002/smll.202301933) studies Fe-MOF-525, but its framework nodes are Zr6 clusters and Fe is bound within the porphyrin linker. It is classified under Zr/Hf node chemistry, with the Fe-metalloporphyrin linker stored separately. The same distinction will be applied to dopants, substrates, and composite additives.

After this correction, the 69 DOI records comprise 53 Zr/Hf records, 6 Cu-carboxylate, 4 Zn-carboxylate, 2 Fe-carboxylate, 2 ZIF, 1 Al-carboxylate, and 1 Cd-carboxylate. The companion-edition deduplication reduces the Zr/Hf set to 52 distinct publication groups.

## Automated direction-selection defect

The former paper-level script selected the first qualifying size outcome even when the paper also contained a same-acid dose trajectory. This mixed acid-identity and dose questions. It affects at least:

- [10.1002/smll.73795](https://doi.org/10.1002/smll.73795): the apparent smaller label comes from an acid-identity/topology comparison, not the formic-acid dose trajectory;
- [10.1021/acs.inorgchem.4c02028](https://doi.org/10.1021/acs.inorgchem.4c02028): within-acid AcOH and benzoic-acid trajectories must be evaluated separately and both show enlargement at the higher doses reported;
- [10.1016/j.memsci.2026.125701](https://doi.org/10.1016/j.memsci.2026.125701): identity and dose outcomes happen to have the same direction but remain separate comparison types.

All new counts will be derived from verified comparison rows, not the first outcome in each paper record.
