"""Freeze settled endpoints only; active peer trajectories are deferred, never duplicated."""
import json
from pathlib import Path
from datetime import datetime, timezone
from campaign_io import load_campaign, sha, write_json

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/wave/MOL_growth_recalculation_20260830')
PARENT=ROOT/'defect_50000steps_production_20260913_v1'
PEER=ROOT/'defect_50000steps_continuation_20260913_v1'
EXPECTED='a033a62f3c6ec3669748d39941e04cc3a8933f14b045117f8f8eacab6efa7674'

def main():
    assert not (BASE/'manifest.json').exists()
    assert sha(PARENT/'manifest.json')==EXPECTED
    _,original=load_campaign(PARENT/'manifest.json')
    peer_manifest=json.loads((PEER/'manifest.json').read_text())
    peer_ids={t['id'] for t in peer_manifest['traces']}
    settled=json.loads((PEER/'outcomes.json').read_text())
    traces=[]; index={}
    for t in original['traces']:
        tid=t['id']
        if tid in peer_ids and tid not in settled:
            index[tid]={'selection':'deferred_active_peer','peer':str(PEER)}
            continue
        origin=PEER if tid in peer_ids else PARENT
        rp=origin/'traces'/tid/'result.json'
        r=json.loads(rp.read_text())
        assert r['valid'], (tid,r['status'])
        index[tid]={'selection':'skip_N_ge_1000' if r['final_N']>=1000 else 'extend',
                    'source_result':str(rp),'source_result_sha256':sha(rp),
                    'source_status':r['status'],'source_N':r['final_N']}
        if r['final_N']>=1000:continue
        cp=rp.parent/'snapshots/terminal/checkpoint.pkl'
        assert sha(cp)==json.loads((cp.parent/'SHA256.json').read_text())['checkpoint.pkl']
        traces.append(dict(t,source_checkpoint=str(cp),source_checkpoint_sha256=sha(cp),
                    source_result=str(rp),source_result_sha256=sha(rp),source_events=r['events'],
                    source_final_N=r['final_N'],source_tau=r['tau'],
                    source_wall_seconds=r.get('continuation',{}).get('cumulative_compute_wall_seconds',r['wall_seconds'])))
    assert len(index)==1152 and traces
    traces.sort(key=lambda t:(-t['source_final_N'],t['id']))
    write_json(BASE/'dataset_source_index.json',index)
    m=dict(schema='acetate-size-followon-1',created_utc=datetime.now(timezone.utc).isoformat(),
           parent_manifest=str(PARENT/'manifest.json'),parent_manifest_sha256=EXPECTED,
           parent_frozen_files=original['files'],peer_status=str(PEER/'status.json'),
           target_N=1000,target_events=None,target_kind='entity_number',wall_seconds=86400,
           workers=original['workers'],analysis=original['analysis'],traces=traces,trace_count=len(traces),
           authorization='2026-09-14: start now alongside active peer, exclude its unsettled IDs',
           endpoint_policy='Separate size-first-passage dataset; keep original fixed-event endpoints',
           files={p.name:sha(p) for p in BASE.iterdir() if p.is_file()})
    write_json(BASE/'manifest.json',m)
    from collections import Counter
    print(json.dumps(dict(traces=len(traces),selection=dict(Counter(x['selection'] for x in index.values())),workers=m['workers'])))

if __name__=='__main__':main()
