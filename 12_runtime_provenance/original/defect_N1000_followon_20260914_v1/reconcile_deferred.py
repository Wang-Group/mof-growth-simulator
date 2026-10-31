"""Read-only source verification; write separate reconciliation, never alter launch manifest."""
import json
from pathlib import Path
from datetime import datetime, timezone
from campaign_io import sha,write_json
base=Path(__file__).resolve().parent
index=json.loads((base/'dataset_source_index.json').read_text())
peer=base.parent/'defect_50000steps_continuation_20260913_v1'
status=json.loads((peer/'status.json').read_text())
assert status['state']=='completed' and status['analysis_completed']==110 and status['analysis_failed']==0
rows=[]
for tid,entry in index.items():
    if entry['selection']!='deferred_active_peer':continue
    rp=peer/'traces'/tid/'result.json'
    r=json.loads(rp.read_text());assert r['valid']
    cp=rp.parent/'snapshots/terminal/checkpoint.pkl'
    digest=sha(cp)
    assert digest==json.loads((cp.parent/'SHA256.json').read_text())['checkpoint.pkl']
    rows.append(dict(id=tid,source_result=str(rp),source_result_sha256=sha(rp),
                     source_checkpoint=str(cp),source_checkpoint_sha256=digest,
                     N=r['final_N'],events=r['events'],source_status=r['status'],
                     selection='skip_N_ge_1000' if r['final_N']>=1000 else 'needs_separate_extension'))
assert len(rows)==9
out=base/'deferred_reconciliation.json';assert not out.exists()
write_json(out,dict(utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(base/'manifest.json'),
                   rows=rows,all_already_large=all(r['N']>=1000 for r in rows)))
print(out.read_text())
