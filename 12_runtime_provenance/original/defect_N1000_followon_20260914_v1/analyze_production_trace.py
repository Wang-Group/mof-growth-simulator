"""Run unchanged parent filling analysis on new continuation snapshots only."""
import argparse
from pathlib import Path
import subprocess
import sys
from campaign_io import load_campaign, sha

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',type=Path,required=True)
ap.add_argument('--trace',type=Path,required=True)
args=ap.parse_args()
base,m=load_campaign(args.manifest)
parent=Path(m['parent_manifest']).parent
assert sha(parent/'manifest.json')==m['parent_manifest_sha256']
result=subprocess.run([sys.executable,str(parent/'analyze_production_trace.py'),
           '--manifest',str(parent/'manifest.json'),'--trace',str(args.trace)])
raise SystemExit(result.returncode)
