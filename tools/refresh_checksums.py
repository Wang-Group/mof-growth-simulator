"""Refresh immutable-file fingerprints after intentional, reviewed edits."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".venv", "reproduced", "reproduced_local"} for part in relative.parts):
            continue
        if path.suffix == ".pyc" or path.name in {"CHECKSUMS.sha256", "verification_results.json", "verification_local.json"}:
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}")
    (ROOT / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Checksummed {len(lines)} immutable files")


if __name__ == "__main__":
    main()
