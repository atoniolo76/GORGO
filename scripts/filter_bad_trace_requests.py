"""Remove known-bad requests from a replayable Mooncake JSONL trace.

The script reads one source trace JSONL and one or more workload result JSON /
proxy request-trace JSONL files, collects request IDs with bad HTTP statuses,
then writes a cleaned trace with matching ``request_id`` or ``uuid`` rows
removed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {e}") from e
            if isinstance(row, dict):
                yield row


def _status_is_bad(status: object, statuses: set[int], status_min: int, status_max: int) -> bool:
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    if statuses and code in statuses:
        return True
    return status_min <= code <= status_max


def _matches_target(row: dict, target_substrings: list[str]) -> bool:
    if not target_substrings:
        return True
    target = str(row.get("target") or "")
    return any(s in target for s in target_substrings)


def _matches_error(row: dict, error_substrings: list[str]) -> bool:
    if not error_substrings:
        return True
    error = str(row.get("error") or "")
    return any(s in error for s in error_substrings)


def _add_bad_id(row: dict, bad_ids: set[str]) -> None:
    for key in ("request_id", "uuid"):
        value = row.get(key)
        if value:
            bad_ids.add(str(value))


def _collect_from_workload_json(
    path: Path,
    *,
    statuses: set[int],
    status_min: int,
    status_max: int,
    error_substrings: list[str],
    bad_ids: set[str],
) -> int:
    doc = json.loads(path.read_text())
    requests = doc.get("requests") if isinstance(doc, dict) else None
    if not isinstance(requests, list):
        return 0
    before = len(bad_ids)
    for row in requests:
        if not isinstance(row, dict):
            continue
        if not _status_is_bad(row.get("status"), statuses, status_min, status_max):
            continue
        if not _matches_error(row, error_substrings):
            continue
        _add_bad_id(row, bad_ids)
    return len(bad_ids) - before


def _collect_from_request_jsonl(
    path: Path,
    *,
    statuses: set[int],
    status_min: int,
    status_max: int,
    target_substrings: list[str],
    error_substrings: list[str],
    bad_ids: set[str],
) -> int:
    before = len(bad_ids)
    for row in _iter_jsonl(path):
        if row.get("kind") not in (None, "request"):
            continue
        if not _status_is_bad(row.get("status"), statuses, status_min, status_max):
            continue
        if not _matches_target(row, target_substrings):
            continue
        if not _matches_error(row, error_substrings):
            continue
        _add_bad_id(row, bad_ids)
    return len(bad_ids) - before


def _load_bad_id_file(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def _write_json_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def filter_trace(
    *,
    trace_path: Path,
    output_path: Path,
    bad_ids: set[str],
    dry_run: bool,
) -> dict:
    rows = 0
    kept = 0
    removed = 0
    removed_ids: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    out = None if dry_run else tmp_path.open("w")
    try:
        for row in _iter_jsonl(trace_path):
            rows += 1
            row_ids = {str(row[k]) for k in ("request_id", "uuid") if row.get(k)}
            if row_ids & bad_ids:
                removed += 1
                removed_ids.extend(sorted(row_ids & bad_ids))
                continue
            kept += 1
            if out is not None:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        if out is not None:
            out.close()

    if not dry_run:
        os.replace(tmp_path, output_path)

    return {
        "trace_path": str(trace_path),
        "output_path": str(output_path),
        "rows": rows,
        "kept": kept,
        "removed": removed,
        "bad_ids": len(bad_ids),
        "removed_ids_sample": removed_ids[:50],
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Input Mooncake JSONL trace.")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned output JSONL trace.")
    parser.add_argument(
        "--workload-result",
        action="append",
        default=[],
        type=Path,
        help="Workload result JSON containing a requests[] array. May be repeated.",
    )
    parser.add_argument(
        "--request-trace",
        action="append",
        default=[],
        type=Path,
        help="Proxy saved request trace JSONL. May be repeated.",
    )
    parser.add_argument(
        "--bad-id-file",
        action="append",
        default=[],
        type=Path,
        help="Text file with request_id/uuid values to remove, one per line.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        type=int,
        help="Exact bad HTTP status. Repeatable. Defaults to 400 if no range is set.",
    )
    parser.add_argument("--status-min", type=int, default=0, help="Inclusive bad status range min.")
    parser.add_argument("--status-max", type=int, default=0, help="Inclusive bad status range max.")
    parser.add_argument(
        "--target-contains",
        action="append",
        default=[],
        help="Only collect proxy-trace bad IDs whose target URL contains this substring.",
    )
    parser.add_argument(
        "--error-contains",
        action="append",
        default=[],
        help="Only collect bad IDs whose error string contains this substring.",
    )
    parser.add_argument("--summary", type=Path, help="Optional summary JSON path.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Count removals without writing output."
    )
    args = parser.parse_args()

    statuses = set(args.status)
    status_min = args.status_min
    status_max = args.status_max
    if not statuses and status_min == 0 and status_max == 0:
        statuses = {400}
    elif (status_min == 0) != (status_max == 0):
        raise SystemExit("--status-min and --status-max must be provided together")

    bad_ids: set[str] = set()
    sources: list[dict] = []
    for path in args.bad_id_file:
        before = len(bad_ids)
        bad_ids.update(_load_bad_id_file(path))
        sources.append({"path": str(path), "type": "bad-id-file", "added": len(bad_ids) - before})
    for path in args.workload_result:
        added = _collect_from_workload_json(
            path,
            statuses=statuses,
            status_min=status_min,
            status_max=status_max,
            error_substrings=args.error_contains,
            bad_ids=bad_ids,
        )
        sources.append({"path": str(path), "type": "workload-result", "added": added})
    for path in args.request_trace:
        added = _collect_from_request_jsonl(
            path,
            statuses=statuses,
            status_min=status_min,
            status_max=status_max,
            target_substrings=args.target_contains,
            error_substrings=args.error_contains,
            bad_ids=bad_ids,
        )
        sources.append({"path": str(path), "type": "request-trace", "added": added})

    if not bad_ids:
        raise SystemExit(
            "no bad request IDs found; pass a workload result, request trace, or bad-id file"
        )

    summary = filter_trace(
        trace_path=args.trace,
        output_path=args.output,
        bad_ids=bad_ids,
        dry_run=args.dry_run,
    )
    summary["sources"] = sources
    summary["statuses"] = sorted(statuses)
    summary["status_range"] = [status_min, status_max] if status_min or status_max else None
    if args.summary:
        _write_json_atomic(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
