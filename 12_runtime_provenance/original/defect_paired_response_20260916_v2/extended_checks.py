"""Exercise all branch treatments beyond initialization, without production data."""
from pathlib import Path
import json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
import campaign as c

if len(sys.argv)>1:
    i=int(sys.argv[1]);m=c.load();sim,prov,io=c.initialize(m,m['traces'][i])
    for _ in range(1000): assert sim.advance()=='event'
    sim.model.check_state();sim.model.ledger.check()
    if m['traces'][i]['mode']=='fixed':assert list(sim.model.ledger.free.values())==m['traces'][i]['free']
    c.save(c.B/'test_logs'/f'extended_{i}.json',dict(valid=True,events=sim.event_count,tau=sim.tau,N=len(sim.model.assembly.entities),pool=sim.model.ledger.record(),transitions=dict(sim.model.transition_counts)))
else:
    def one(i):
        with (c.B/'test_logs'/f'extended_{i}.log').open('xb') as f:
            p=subprocess.run([sys.executable,__file__,str(i)],stdout=f,stderr=subprocess.STDOUT,env=c.environment())
        return i,p.returncode
    with ThreadPoolExecutor(max_workers=4) as p: results=dict(p.map(one,range(12)))
    c.save(c.B/'EXTENDED_TEST_RESULTS.json',dict(passed=all(x==0 for x in results.values()),manifest_sha256=c.sha(c.B/'manifest.json'),script_sha256=c.sha(__file__),results=results,events_per_condition=1000))
    assert all(x==0 for x in results.values()),results
    print('1000 molecular events per treatment: all 12 passed')
