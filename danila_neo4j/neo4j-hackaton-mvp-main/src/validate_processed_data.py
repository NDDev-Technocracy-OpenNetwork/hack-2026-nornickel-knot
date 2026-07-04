#!/usr/bin/env python3
"""Validate processed JSONL graph import files.

The validator checks referential integrity and data quality for the
`data/processed/*.jsonl` contract used by `src/load_jsonl_graph.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CRITICAL = "CRITICAL"
WARNING = "WARNING"

DOCUMENTS_FILE = "documents.jsonl"
CHUNKS_FILE = "chunks.jsonl"
ENTITIES_FILE = "entities.jsonl"
MENTIONS_FILE = "entity_mentions.jsonl"
RELATIONS_FILE = "relations.jsonl"

REQUIRED_FILES = (
    DOCUMENTS_FILE,
    CHUNKS_FILE,
    ENTITIES_FILE,
    MENTIONS_FILE,
    RELATIONS_FILE,
)

CSV_FIELDS = ("severity", "file", "row_id", "error_type", "message")


@dataclass(frozen=True)
class Record:
    file_name: str
    line_number: int
    data: dict[str, Any]


@dataclass(frozen=True)
class Issue:
    severity: str
    file: str
    row_id: str
    error_type: str
    message: str


class Validator:
    def __init__(self, data_dir: Path, reports_dir: Path, strict_warnings: bool) -> None:
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.strict_warnings = strict_warnings
        self.issues: list[Issue] = []
        self.records: dict[str, list[Record]] = {file_name: [] for file_name in REQUIRED_FILES}
        self.file_stats: dict[str, dict[str, Any]] = {
            file_name: {"exists": False, "rows": 0} for file_name in REQUIRED_FILES
        }

    def run(self) -> int:
        self._load_files()
        indexes = self._validate_primary_ids()
        self._validate_confidence_values()
        self._validate_references(indexes)
        self._validate_warnings(indexes)
        exit_code = self._exit_code()
        self._write_reports(exit_code)
        self._print_summary(exit_code)
        return exit_code

    def _load_files(self) -> None:
        for file_name in REQUIRED_FILES:
            path = self.data_dir / file_name
            if not path.exists():
                self._add_issue(
                    CRITICAL,
                    file_name,
                    "",
                    "missing_file",
                    f"Required file is missing: {path}",
                )
                continue
            if not path.is_file():
                self._add_issue(
                    CRITICAL,
                    file_name,
                    "",
                    "missing_file",
                    f"Required path is not a file: {path}",
                )
                continue

            self.file_stats[file_name]["exists"] = True
            self.records[file_name] = self._read_jsonl(path, file_name)
            self.file_stats[file_name]["rows"] = len(self.records[file_name])

    def _read_jsonl(self, path: Path, file_name: str) -> list[Record]:
        records: list[Record] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        self._add_issue(
                            CRITICAL,
                            file_name,
                            f"line {line_number}",
                            "invalid_jsonl",
                            f"Invalid JSON at line {line_number}: {exc.msg}",
                        )
                        continue
                    if not isinstance(row, dict):
                        self._add_issue(
                            CRITICAL,
                            file_name,
                            f"line {line_number}",
                            "invalid_jsonl",
                            "JSONL row must be a JSON object",
                        )
                        continue
                    records.append(Record(file_name=file_name, line_number=line_number, data=row))
        except UnicodeDecodeError as exc:
            self._add_issue(
                CRITICAL,
                file_name,
                "",
                "file_read_error",
                f"File is not valid UTF-8: {exc}",
            )
        except OSError as exc:
            self._add_issue(
                CRITICAL,
                file_name,
                "",
                "file_read_error",
                f"Could not read file: {exc}",
            )
        return records

    def _validate_primary_ids(self) -> dict[str, dict[str, Record]]:
        return {
            "documents": self._validate_unique_id(DOCUMENTS_FILE, "document_id"),
            "chunks": self._validate_unique_id(CHUNKS_FILE, "chunk_id"),
            "entities": self._validate_unique_id(ENTITIES_FILE, "entity_id"),
            "mentions": self._validate_unique_id(MENTIONS_FILE, "mention_id"),
            "relations": self._validate_unique_id(RELATIONS_FILE, "relation_id"),
        }

    def _validate_unique_id(self, file_name: str, id_field: str) -> dict[str, Record]:
        seen: dict[str, Record] = {}
        for record in self.records[file_name]:
            value = self._string_value(record.data.get(id_field))
            if not value:
                self._add_issue(
                    CRITICAL,
                    file_name,
                    self._fallback_row_id(record),
                    "missing_id",
                    f"{id_field} is missing or empty",
                )
                continue
            if value in seen:
                first = seen[value]
                self._add_issue(
                    CRITICAL,
                    file_name,
                    value,
                    "duplicate_id",
                    f"Duplicate {id_field}; first seen at line {first.line_number}, repeated at line {record.line_number}",
                )
                continue
            seen[value] = record
        return seen

    def _validate_confidence_values(self) -> None:
        for file_name in REQUIRED_FILES:
            for record in self.records[file_name]:
                if "confidence" not in record.data:
                    continue
                value = record.data.get("confidence")
                if not self._is_valid_confidence(value):
                    self._add_issue(
                        CRITICAL,
                        file_name,
                        self._row_id(record),
                        "invalid_confidence",
                        "confidence must be a number from 0 to 1",
                    )

    def _validate_references(self, indexes: dict[str, dict[str, Record]]) -> None:
        document_ids = indexes["documents"]
        chunk_ids = indexes["chunks"]
        entity_ids = indexes["entities"]

        for record in self.records[CHUNKS_FILE]:
            self._require_reference(
                record,
                "document_id",
                document_ids,
                "missing_reference",
                "chunk.document_id must exist in documents",
            )

        for record in self.records[MENTIONS_FILE]:
            self._require_reference(
                record,
                "chunk_id",
                chunk_ids,
                "missing_reference",
                "mention.chunk_id must exist in chunks",
            )
            self._require_reference(
                record,
                "entity_id",
                entity_ids,
                "missing_reference",
                "mention.entity_id must exist in entities",
            )

        for record in self.records[RELATIONS_FILE]:
            self._require_reference(
                record,
                "chunk_id",
                chunk_ids,
                "missing_reference",
                "relation.chunk_id must exist in chunks",
            )
            self._require_reference(
                record,
                "source_entity_id",
                entity_ids,
                "missing_reference",
                "relation.source_entity_id must exist in entities",
            )
            self._require_reference(
                record,
                "target_entity_id",
                entity_ids,
                "missing_reference",
                "relation.target_entity_id must exist in entities",
            )
            if not self._string_value(record.data.get("relation_type")):
                self._add_issue(
                    CRITICAL,
                    RELATIONS_FILE,
                    self._row_id(record),
                    "empty_relation_type",
                    "relation_type must not be empty",
                )

    def _validate_warnings(self, indexes: dict[str, dict[str, Record]]) -> None:
        chunk_records = indexes["chunks"]
        chunk_text_by_id = {
            chunk_id: self._text_value(record.data.get("text"))
            for chunk_id, record in chunk_records.items()
        }

        for record in self.records[CHUNKS_FILE]:
            if record.data.get("page_start") is None or record.data.get("page_end") is None:
                self._add_issue(
                    WARNING,
                    CHUNKS_FILE,
                    self._row_id(record),
                    "missing_page_range",
                    "chunk.page_start or chunk.page_end is missing or null",
                )
            chunk_text = self._text_value(record.data.get("text"))
            if len(chunk_text.strip()) < 30:
                self._add_issue(
                    WARNING,
                    CHUNKS_FILE,
                    self._row_id(record),
                    "short_chunk_text",
                    "chunk.text is empty or shorter than 30 characters",
                )

        for record in self.records[MENTIONS_FILE]:
            surface_text = self._text_value(record.data.get("surface_text"))
            chunk_id = self._string_value(record.data.get("chunk_id"))
            chunk_text = chunk_text_by_id.get(chunk_id, "")

            if not surface_text.strip():
                self._add_issue(
                    WARNING,
                    MENTIONS_FILE,
                    self._row_id(record),
                    "empty_surface_text",
                    "mention.surface_text is empty",
                )
            elif chunk_text and surface_text not in chunk_text:
                self._add_issue(
                    WARNING,
                    MENTIONS_FILE,
                    self._row_id(record),
                    "surface_text_not_found",
                    "mention.surface_text was not found in the referenced chunk.text",
                )

            if record.data.get("char_start") is None or record.data.get("char_end") is None:
                self._add_issue(
                    WARNING,
                    MENTIONS_FILE,
                    self._row_id(record),
                    "missing_char_range",
                    "mention.char_start or mention.char_end is missing or null",
                )

        for record in self.records[RELATIONS_FILE]:
            evidence_text = self._text_value(record.data.get("evidence_text"))
            chunk_id = self._string_value(record.data.get("chunk_id"))
            chunk_text = chunk_text_by_id.get(chunk_id, "")

            if not evidence_text.strip():
                self._add_issue(
                    WARNING,
                    RELATIONS_FILE,
                    self._row_id(record),
                    "empty_evidence_text",
                    "relation.evidence_text is empty",
                )
            elif chunk_text and evidence_text not in chunk_text:
                self._add_issue(
                    WARNING,
                    RELATIONS_FILE,
                    self._row_id(record),
                    "evidence_text_not_found",
                    "relation.evidence_text was not found in the referenced chunk.text",
                )

    def _require_reference(
        self,
        record: Record,
        field_name: str,
        target_index: dict[str, Record],
        error_type: str,
        message: str,
    ) -> None:
        value = self._string_value(record.data.get(field_name))
        if not value or value not in target_index:
            detail = f"{message}: {field_name}={value or '<empty>'}"
            self._add_issue(CRITICAL, record.file_name, self._row_id(record), error_type, detail)

    def _write_reports(self, exit_code: int) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_dir / "validation_report.json"
        csv_path = self.reports_dir / "validation_errors.csv"

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "data_dir": str(self.data_dir),
            "reports_dir": str(self.reports_dir),
            "strict_warnings": self.strict_warnings,
            "files": self.file_stats,
            "summary": {
                "critical": self._count(CRITICAL),
                "warning": self._count(WARNING),
                "total": len(self.issues),
                "exit_code": exit_code,
            },
            "issues": [asdict(issue) for issue in self.issues],
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for issue in self.issues:
                writer.writerow(asdict(issue))

    def _print_summary(self, exit_code: int) -> None:
        critical_count = self._count(CRITICAL)
        warning_count = self._count(WARNING)
        print("Processed data validation")
        print(f"Data dir: {self.data_dir}")
        print(f"Reports dir: {self.reports_dir}")
        print("Files:")
        for file_name in REQUIRED_FILES:
            stats = self.file_stats[file_name]
            status = "found" if stats["exists"] else "missing"
            print(f"  {file_name}: {status}, rows={stats['rows']}")
        print("Issues:")
        print(f"  CRITICAL: {critical_count}")
        print(f"  WARNING: {warning_count}")
        print(f"Report: {self.reports_dir / 'validation_report.json'}")
        print(f"Errors CSV: {self.reports_dir / 'validation_errors.csv'}")
        if exit_code == 0 and warning_count:
            print("Result: PASSED_WITH_WARNINGS")
        elif exit_code == 0:
            print("Result: PASSED")
        else:
            print("Result: FAILED")

    def _exit_code(self) -> int:
        if self._count(CRITICAL):
            return 1
        if self.strict_warnings and self._count(WARNING):
            return 1
        return 0

    def _count(self, severity: str) -> int:
        return sum(1 for issue in self.issues if issue.severity == severity)

    def _add_issue(
        self,
        severity: str,
        file_name: str,
        row_id: str,
        error_type: str,
        message: str,
    ) -> None:
        self.issues.append(
            Issue(
                severity=severity,
                file=file_name,
                row_id=row_id,
                error_type=error_type,
                message=message,
            )
        )

    def _row_id(self, record: Record) -> str:
        id_fields_by_file = {
            DOCUMENTS_FILE: ("document_id",),
            CHUNKS_FILE: ("chunk_id",),
            ENTITIES_FILE: ("entity_id",),
            MENTIONS_FILE: ("mention_id",),
            RELATIONS_FILE: ("relation_id",),
        }
        for field_name in id_fields_by_file.get(record.file_name, ()):  # pragma: no branch
            value = self._string_value(record.data.get(field_name))
            if value:
                return value
        return self._fallback_row_id(record)

    @staticmethod
    def _fallback_row_id(record: Record) -> str:
        return f"line {record.line_number}"

    @staticmethod
    def _string_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _text_value(value: Any) -> str:
        return value if isinstance(value, str) else ""

    @staticmethod
    def _is_valid_confidence(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value)) and 0 <= float(value) <= 1


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed JSONL graph data.")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed JSONL files.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for validation reports.")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return exit code 1 when warnings are present.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    validator = Validator(
        data_dir=Path(args.data_dir),
        reports_dir=Path(args.reports_dir),
        strict_warnings=args.strict_warnings,
    )
    return validator.run()


if __name__ == "__main__":
    sys.exit(main())
