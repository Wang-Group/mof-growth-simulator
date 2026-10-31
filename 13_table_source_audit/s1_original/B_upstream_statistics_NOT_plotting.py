"""Reconstruct the carboxylic-acid size starting set from batch extractions.

This script does not adjudicate directions. It preserves every qualifying study
and outcome so that condition-level re-reading can replace the automated labels.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXTRACTION = ROOT / "analysis_workspace" / "systematic_review" / "extraction"
OUT = Path(__file__).resolve().parent

CARBOX = re.compile(
    r"formic|acetic|propionic|butyric|pivalic|benzoic|toluic|"
    r"trifluoroacet|dichloroacet|oxalic|malonic|succinic|citric|"
    r"salicylic|cinnamic|oleic|dodecanoic|lauric|stearic|acrylic|"
    r"methacrylic|valeric|heptanoic|HAc|FAc|AcOH|HCOOH|TFA\b|"
    r"HBA\b|HOAc|monocarboxylic",
    re.I,
)
NONCARBOX = re.compile(
    r"hydrofluoric|\bHF\b|hydrochloric|\bHCl\b|nitric|sulfuric|"
    r"phosphoric|sodium formate|sodium acetate|NaOAc|triethylamine|"
    r"\bTEA\b|TMAOH|NaOH|KOH|ammonia|imidazole|proline|"
    r"benzimidazole|glycine|ferric|Fe\(NO3|polymer|surfactant|"
    r"CTAB|PVP|F127|P123|ionic liquid|CNF",
    re.I,
)
VALID_DIRECTIONS = {"larger", "smaller", "non_monotonic", "no_change"}

EXPLICIT_FAMILY = {
    "10.1016/j.micromeso.2020.110215": "Cd-carboxylate",
    "10.1016/j.micromeso.2022.112379": "Cu-carboxylate",
    "10.1016/s1872-2067(20)63625-7": "Fe-carboxylate",
    "10.1038/s41467-018-03979-2": "Cu-carboxylate",
    # Fe is in the porphyrin linker; the framework nodes are Zr6 clusters.
    "10.1002/smll.202301933": "Zr/Hf",
}

PUBLICATION_GROUP = {
    # German and International Edition versions of the same Wiley article.
    "10.1002/ange.201907074": "publication::10.1002/anie.201907074",
    "10.1002/anie.201907074": "publication::10.1002/anie.201907074",
}


def normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.rstrip(" .;,)\n")


def read_records() -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in sorted(EXTRACTION.glob("batch_*.json")):
        if any(token in path.name for token in ("input", "manifest", "todo", "args")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict):
                records.append((path, item))
    return records


def qualifying_studies(paper: dict) -> list[tuple[int, dict, list[dict]]]:
    found: list[tuple[int, dict, list[dict]]] = []
    for study_index, study in enumerate(paper.get("studies") or [], start=1):
        axis = str(study.get("axis", ""))
        if not axis.startswith(("A_modulator", "B_modulator")):
            continue
        text = " ".join(
            [
                str(study.get("variable", "")),
                str(study.get("levels", "")),
                str(paper.get("system", "")),
            ]
        )
        if NONCARBOX.search(text) or not CARBOX.search(text):
            continue
        size_outcomes = [
            outcome
            for outcome in (study.get("outcomes") or [])
            if outcome.get("kind") == "size"
            and str(outcome.get("direction", "")) in VALID_DIRECTIONS
        ]
        if size_outcomes:
            found.append((study_index, study, size_outcomes))
    return found


def resolve_family(doi: str, extracted: str, systems: str) -> tuple[str, str]:
    if doi in EXPLICIT_FAMILY:
        return EXPLICIT_FAMILY[doi], "explicit DOI-level resolution"
    family = (extracted or "").strip()
    system = (systems or "").lower()
    if family.startswith("Zr/Hf"):
        return "Zr/Hf", "normalized from extracted Zr/Hf family"
    if family == "Zn/Co-imidazolate":
        return "ZIF", "normalized from imidazolate family"
    if family == "Cu":
        return "Cu-carboxylate", "normalized from extracted Cu family"
    if family == "Fe":
        return "Fe-carboxylate", "normalized from extracted Fe family"
    if family == "Al":
        return "Al-carboxylate", "normalized from extracted Al family"
    if family == "Zn-carboxylate":
        return "Zn-carboxylate", "retained extracted family"
    if "uio" in system or "mof-801" in system or "pcn-224" in system:
        return "Zr/Hf", "resolved from framework identity in extracted system"
    return family or "unresolved", "requires manual resolution"


def main() -> None:
    raw = read_records()
    study_rows: list[dict[str, str]] = []
    outcome_rows: list[dict[str, str]] = []
    sources_by_doi: dict[str, set[str]] = defaultdict(set)
    paper_records: dict[str, list[dict]] = defaultdict(list)

    for path, paper in raw:
        if paper.get("verdict") != "relevant_extracted":
            continue
        qualifying = qualifying_studies(paper)
        if not qualifying:
            continue
        doi = normalize_doi(str(paper.get("doi", "")))
        if not doi:
            continue
        rel_path = str(path.relative_to(ROOT))
        sources_by_doi[doi].add(rel_path)
        paper_records[doi].append(paper)
        for study_index, study, size_outcomes in qualifying:
            axis = str(study.get("axis", ""))
            comparison_type = "dose_or_presence" if axis.startswith("B_") else "identity"
            study_id = f"{doi}::s{study_index}::{comparison_type}"
            study_rows.append(
                {
                    "study_id": study_id,
                    "doi": doi,
                    "doi_url": f"https://doi.org/{doi}",
                    "comparison_type": comparison_type,
                    "axis": axis,
                    "family_extracted": str(paper.get("family", "")),
                    "system": str(paper.get("system", "")),
                    "paper_reason": str(paper.get("reason", "")),
                    "variable": str(study.get("variable", "")),
                    "levels": str(study.get("levels", "")),
                    "study_quote": str(study.get("quote", "")),
                    "quote_where": str(study.get("quote_where", "")),
                    "source_batch": rel_path,
                }
            )
            for outcome_index, outcome in enumerate(size_outcomes, start=1):
                outcome_rows.append(
                    {
                        "outcome_id": f"{study_id}::o{outcome_index}",
                        "study_id": study_id,
                        "doi": doi,
                        "comparison_type": comparison_type,
                        "axis": axis,
                        "family_extracted": str(paper.get("family", "")),
                        "system": str(paper.get("system", "")),
                        "variable": str(study.get("variable", "")),
                        "levels": str(study.get("levels", "")),
                        "automated_direction": str(outcome.get("direction", "")),
                        "numbers": str(outcome.get("numbers", "")),
                        "confounds": str(outcome.get("confounds", "")),
                        "study_quote": str(study.get("quote", "")),
                        "quote_where": str(study.get("quote_where", "")),
                        "source_batch": rel_path,
                    }
                )

    # De-duplicate exact repeated study/outcome records while preserving divergent extractions.
    def dedup(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, str]]:
        seen: set[tuple[str, ...]] = set()
        result: list[dict[str, str]] = []
        for row in rows:
            marker = tuple(row.get(key, "") for key in keys)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(row)
        return result

    study_rows = dedup(study_rows, ("doi", "axis", "variable", "levels", "study_quote"))
    outcome_rows = dedup(
        outcome_rows,
        ("doi", "axis", "variable", "levels", "automated_direction", "numbers", "study_quote"),
    )

    paper_rows: list[dict[str, str]] = []
    for doi in sorted(paper_records):
        rows = [row for row in study_rows if row["doi"] == doi]
        all_directions = [
            row["automated_direction"] for row in outcome_rows if row["doi"] == doi
        ]
        dose_directions = [
            row["automated_direction"]
            for row in outcome_rows
            if row["doi"] == doi and row["comparison_type"] == "dose_or_presence"
        ]
        identity_directions = [
            row["automated_direction"]
            for row in outcome_rows
            if row["doi"] == doi and row["comparison_type"] == "identity"
        ]
        family_values = sorted({row["family_extracted"].strip() for row in rows if row["family_extracted"].strip()})
        systems = sorted({row["system"].strip() for row in rows if row["system"].strip()})
        scope = "dose_or_presence" if dose_directions else "identity"
        extracted_family = " | ".join(family_values)
        joined_systems = " | ".join(systems)
        resolved_family, family_basis = resolve_family(doi, extracted_family, joined_systems)
        publication_group = PUBLICATION_GROUP.get(doi, f"publication::{doi}")
        paper_rows.append(
            {
                "study_id": f"study::{doi}",
                "publication_group_id": publication_group,
                "duplicate_relation": "companion edition" if doi in PUBLICATION_GROUP else "none identified",
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}",
                "starting_scope": scope,
                "family_extracted": extracted_family,
                "family_resolved": resolved_family,
                "family_resolution_basis": family_basis,
                "systems": joined_systems,
                "dose_directions_raw": " | ".join(dose_directions),
                "identity_directions_raw": " | ".join(identity_directions),
                "all_directions_raw": " | ".join(all_directions),
                "source_batches": " | ".join(sorted(sources_by_doi[doi])),
                "access_state": "pending",
                "verification_state": "pending",
            }
        )

    for filename, rows in (
        ("seed_studies.csv", paper_rows),
        ("seed_extracted_comparisons.csv", study_rows),
        ("seed_size_outcomes.csv", outcome_rows),
    ):
        path = OUT / filename
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    dose_papers = [row for row in paper_rows if row["starting_scope"] == "dose_or_presence"]
    identity_only = [row for row in paper_rows if row["starting_scope"] == "identity"]
    raw_record_counts = Counter()
    for _path, paper in raw:
        if paper.get("verdict") != "relevant_extracted":
            continue
        q = qualifying_studies(paper)
        if q:
            raw_record_counts["qualifying_records"] += 1
            raw_record_counts[
                "dose_records" if any(str(s.get("axis", "")).startswith("B_") for _, s, _ in q) else "identity_records"
            ] += 1
    summary = {
        "unique_papers": len(paper_rows),
        "unique_dose_or_presence_papers": len(dose_papers),
        "unique_identity_only_papers": len(identity_only),
        "distinct_publication_groups": len({row["publication_group_id"] for row in paper_rows}),
        "distinct_dose_or_presence_publication_groups": len(
            {row["publication_group_id"] for row in dose_papers}
        ),
        "qualifying_study_rows_after_exact_deduplication": len(study_rows),
        "qualifying_size_outcomes_after_exact_deduplication": len(outcome_rows),
        "raw_record_counts_before_doi_deduplication": dict(raw_record_counts),
        "papers_with_multiple_batch_records": {
            doi: len(records) for doi, records in paper_records.items() if len(records) > 1
        },
        "unresolved_family_dois": [
            row["doi"] for row in paper_rows if row["family_resolved"] == "unresolved"
        ],
        "dose_family_counts_by_doi_record": dict(
            Counter(row["family_resolved"] for row in dose_papers)
        ),
    }
    (OUT / "seed_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
