"""Read-only search within the supplied project; write local audit evidence only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path("Z:/\u5149\u50ac\u5316\u5c0f\u7ec4/0-papers-in-progress/2025-\u5b8b\u5b87\u822a-MOL_growth_mechanism")
OUT = Path(__file__).resolve().parent
FIELDS = ("Zr", "FA", "L", "Ratio", "Step", "Index", "SC", "KC")
PATTERN = re.compile(r"(" + "|".join(FIELDS) + r")_([^_/\\]+)")
TERMS = (b"analyze_figure6_current_parameter_phase_v1", b"def build_probability_cube", b"def phase_panel", b"unique_structures_result.csv")
report = {"root": str(ROOT), "read_only": True, "files_scanned": 0,
          "run_files": [], "code_matches": [], "archives": [], "errors": []}

def conditions(path):
    for segment in str(path).replace("\\", "/").split("/"):
        parsed = dict(PATTERN.findall(segment))
        if all(field in parsed for field in FIELDS):
            return parsed
    return None

for directory, dirs, files in os.walk(ROOT, onerror=lambda exc: report["errors"].append(str(exc))):
    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
    for name in files:
        report["files_scanned"] += 1
        p = Path(directory) / name
        relative = p.relative_to(ROOT).as_posix()
        try:
            cond = conditions(relative)
            if cond and p.suffix.lower() in {".mol2", ".pkl", ".csv", ".json", ".py", ".ipynb", ".log"}:
                report["run_files"].append({"path": relative, "conditions": cond, "bytes": p.stat().st_size})
            if p.suffix.lower() in {".py", ".ipynb"} and p.stat().st_size < 30_000_000:
                raw = p.read_bytes()
                found = [term.decode() for term in TERMS if term in raw]
                if found:
                    report["code_matches"].append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "terms": found})
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p) as archive:
                    entries = []
                    for entry in archive.infolist():
                        cond = conditions(entry.filename)
                        relevant_name = any(x in entry.filename.lower() for x in ("phase", "figure6", "unique_structures", "entity_number", "entities_number"))
                        if cond or relevant_name:
                            entries.append({"path": entry.filename, "bytes": entry.file_size, "conditions": cond, "directory": entry.is_dir()})
                    report["archives"].append({"path": relative, "bytes": p.stat().st_size, "entry_count": len(archive.infolist()), "matches": entries})
        except (OSError, zipfile.BadZipFile) as exc:
            report["errors"].append({"path": relative, "error": str(exc)})

destination = OUT / "nas_search.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files_scanned": report["files_scanned"], "run_files": len(report["run_files"]),
                  "code_matches": report["code_matches"], "archives": [{k:v for k,v in item.items() if k != "matches"} for item in report["archives"]], "errors": report["errors"]}, ensure_ascii=False, indent=2))
