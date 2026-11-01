"""Read-only, bounded project search. Send to Python via stdin; no remote writes."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import zipfile

ROOTS = [Path('/home/wave') / x for x in (
    'MOL_growth_recalculation_20260830', 'MOL_growth_recalculation_20260831',
    'MOL_growth_reviewed_20260831', 'mof_kmc')]
TERMS = (b'analyze_figure6_current_parameter_phase_v1', b'def build_probability_cube',
         b'def phase_panel', b'unique_structures_result.csv')
report = {'read_only': True, 'roots': list(map(str, ROOTS)), 'files_seen': 0,
          'code_read': 0, 'matches': [], 'archives': [], 'errors': [], 'limits': []}
start = time.monotonic()
archives = []
for root in ROOTS:
    root_start = time.monotonic()
    root_seen = 0
    for directory, dirs, files in os.walk(root, onerror=lambda e: report['errors'].append(str(e))):
        dirs[:] = [d for d in dirs if not any(x in d.lower() for x in ('env', 'site-packages', 'node_modules', '__pycache__')) and d != '.git' and not (Path(directory)/d).is_symlink()]
        dirs.sort(key=lambda d: (d not in {'code', 'source_snapshot', 'figure_rebuild_20260901'}, d))
        if time.monotonic() - root_start > 150 or root_seen > 2_000_000:
            report['limits'].append(str(root)); break
        for name in files:
            report['files_seen'] += 1
            root_seen += 1
            p = Path(directory) / name
            try:
                if p.is_symlink(): continue
                if name.lower().endswith(('.tar.gz', '.tgz', '.tar', '.zip')):
                    archives.append(p)
                if p.suffix.lower() in {'.py', '.ipynb', '.md', '.sh'} and p.stat().st_size < 10_000_000:
                    raw = p.read_bytes(); report['code_read'] += 1
                    found = [x.decode() for x in TERMS if x in raw]
                    if found:
                        report['matches'].append({'path': str(p), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'terms': found})
            except OSError as exc:
                report['errors'].append({'path': str(p), 'error': str(exc)})
for p in archives:
    item = {'path': str(p), 'bytes': p.stat().st_size, 'matches': [], 'entries': 0}
    try:
        if item['bytes'] > 500_000_000:
            item['not_read'] = 'archive size exceeds 500 MB bound'
        elif p.suffix == '.zip':
            with zipfile.ZipFile(p) as archive:
                for entry in archive.infolist():
                    item['entries'] += 1
                    if any(x in entry.filename.lower() for x in ('figure6', 'phase_v1', 'unique_structures', '2025-08-18')):
                        item['matches'].append({'path': entry.filename, 'bytes': entry.file_size})
        else:
            with tarfile.open(p, 'r|*') as archive:
                for entry in archive:
                    item['entries'] += 1
                    if any(x in entry.name.lower() for x in ('figure6', 'phase_v1', 'unique_structures', '2025-08-18')):
                        item['matches'].append({'path': entry.name, 'bytes': entry.size})
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        item['error'] = str(exc)
    report['archives'].append(item)
print(json.dumps(report, sort_keys=True, indent=2))
