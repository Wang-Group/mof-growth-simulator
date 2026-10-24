# UiO-66 entity-13 structure uniqueness analysis

- Cases analyzed: `24`.
- Input snapshots: `assembly_last_entity_number_13.mol2` from the completed long-growth batch.
- Link assignment cutoff: `0.1 A` between BDC and Zr6 carboxylate centers.
- Important limitation: the saved entity-13 snapshots are mol2 files with zero bond records; no entity-13 pickle files are present. Therefore this analysis reconstructs SBU-linker connectivity from geometry rather than reading the original `connected_entities` objects.

## Topology distribution

- `T_c7f033d1c040`: n = `11`, repeats = `01,04,05,06,07,09,12,13,14,15,16`, Zr6/BDC = `5/8`, edges = `16`, mono/bridging BDC = `0/8`, cycle rank = `4`.
- `T_ecc1ba11d5f4`: n = `4`, repeats = `08,11,20,22`, Zr6/BDC = `5/8`, edges = `15`, mono/bridging BDC = `1/7`, cycle rank = `3`.
- `T_fea0200fb320`: n = `3`, repeats = `02,18,21`, Zr6/BDC = `5/8`, edges = `16`, mono/bridging BDC = `0/8`, cycle rank = `4`.
- `T_35e06fec9675`: n = `1`, repeats = `10`, Zr6/BDC = `4/9`, edges = `14`, mono/bridging BDC = `4/5`, cycle rank = `2`.
- `T_41c7dce37db4`: n = `1`, repeats = `23`, Zr6/BDC = `5/8`, edges = `15`, mono/bridging BDC = `1/7`, cycle rank = `3`.
- `T_63782e4fe2ca`: n = `1`, repeats = `19`, Zr6/BDC = `4/9`, edges = `14`, mono/bridging BDC = `4/5`, cycle rank = `2`.
- `T_aa3859fb03cd`: n = `1`, repeats = `17`, Zr6/BDC = `5/8`, edges = `14`, mono/bridging BDC = `2/6`, cycle rank = `2`.
- `T_aeee724067d1`: n = `1`, repeats = `00`, Zr6/BDC = `5/8`, edges = `15`, mono/bridging BDC = `1/7`, cycle rank = `3`.
- `T_c055a18b91ce`: n = `1`, repeats = `03`, Zr6/BDC = `4/9`, edges = `14`, mono/bridging BDC = `4/5`, cycle rank = `2`.

## Interpretation

The 24 entity-13 snapshots are not a single unique structure. They separate into multiple reconstructed SBU-linker topologies, and the stricter SBU-geometry and all-atom distance fingerprints split them further. Thus entity 13 should be described as a recurrent bottleneck or motif family, not as one deterministic nucleus.
