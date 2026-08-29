from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmarks"

RECORD_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,63})\Z")
SYSTEM_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,127})\Z")
SOURCE_DOCUMENT_RE = re.compile(
    r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\.json\Z"
)
SCENARIOS = frozenset({"boundary", "browsing", "buying", "intent_override"})
METRIC_KEYS = frozenset(
    {
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
    }
)
SCENARIO_METRIC_KEYS = frozenset(
    {"sample_count", "hit_rate_at_10", "mrr", "mttc"}
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "record_id",
        "source_document",
        "system",
        "sample_count",
        "metrics",
        "scenario_metrics",
    }
)
REQUIRED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS - {"scenario_metrics"}


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _exact_keys(
    value: dict[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    try:
        numeric_value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name} must be finite and between {minimum} and {maximum}"
        ) from error
    if not math.isfinite(numeric_value) or not minimum <= numeric_value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


def _sample_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_metrics(
    metrics: object,
    name: str,
    *,
    scenario: bool,
) -> int | None:
    values = _object(metrics, name)
    _exact_keys(
        values,
        SCENARIO_METRIC_KEYS if scenario else METRIC_KEYS,
        name,
    )
    scenario_sample_count = None
    if scenario:
        scenario_sample_count = _sample_count(
            values["sample_count"],
            f"{name}.sample_count",
        )
    for key in ("hit_rate_at_10", "mrr"):
        _number(values[key], f"{name}.{key}", minimum=0.0, maximum=1.0)
    _number(values["mttc"], f"{name}.mttc", minimum=1.0, maximum=11.0)
    if not scenario:
        for key in ("efficiency", "recommended_technical_score"):
            _number(values[key], f"{name}.{key}", minimum=0.0, maximum=1.0)
    return scenario_sample_count


def validate_record(record: object) -> dict[str, object]:
    """Validate the aggregate-only benchmark record schema without mutation."""

    value = _object(record, "record")
    actual_keys = set(value)
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - actual_keys)
    unexpected = sorted(actual_keys - TOP_LEVEL_KEYS)
    if missing or unexpected:
        raise ValueError(
            "record keys do not match the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    record_id = value["record_id"]
    if not isinstance(record_id, str) or RECORD_ID_RE.fullmatch(record_id) is None:
        raise ValueError("record_id must be a safe lowercase file stem")

    source_document = value["source_document"]
    if (
        not isinstance(source_document, str)
        or len(source_document) > 256
        or SOURCE_DOCUMENT_RE.fullmatch(source_document) is None
    ):
        raise ValueError("source_document must be a relative JSON path")
    source_path = PurePosixPath(source_document)
    if (
        source_path.is_absolute()
        or ".." in source_path.parts
        or source_path.suffix.lower() != ".json"
    ):
        raise ValueError("source_document must be a safe relative JSON path")

    system = value["system"]
    if (
        not isinstance(system, str)
        or SYSTEM_ID_RE.fullmatch(system) is None
    ):
        raise ValueError("system must be a compact identifier")

    sample_count = _sample_count(value["sample_count"], "sample_count")
    _validate_metrics(value["metrics"], "metrics", scenario=False)

    scenario_metrics = value.get("scenario_metrics")
    if scenario_metrics is not None:
        scenarios = _object(scenario_metrics, "scenario_metrics")
        if not scenarios or not set(scenarios).issubset(SCENARIOS):
            raise ValueError("scenario_metrics contains unsupported scenario names")
        scenario_total = 0
        for scenario_name, metrics in scenarios.items():
            scenario_sample_count = _validate_metrics(
                metrics,
                f"scenario_metrics.{scenario_name}",
                scenario=True,
            )
            if scenario_sample_count is None:  # Kept explicit for static type checkers.
                raise AssertionError("scenario metrics must include sample_count")
            scenario_total += scenario_sample_count
        if scenario_total != sample_count:
            raise ValueError("scenario sample counts must sum to sample_count")
    return value


def deterministic_json(record: object) -> bytes:
    value = validate_record(record)
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def append_record(
    record: object,
    directory: str | Path = DEFAULT_BENCHMARK_DIRECTORY,
) -> Path:
    """Atomically publish one new record, refusing every existing destination."""

    value = validate_record(record)
    encoded = deterministic_json(value)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{value['record_id']}.json"
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite benchmark record: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{value['record_id']}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite benchmark record: {destination}"
            ) from error
        temporary.unlink()
        _fsync_directory(root)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append one validated aggregate benchmark record"
    )
    parser.add_argument("input", help="JSON file conforming to benchmarks/schema.json")
    parser.add_argument(
        "--directory",
        default=DEFAULT_BENCHMARK_DIRECTORY,
        help="benchmark record directory (default: repository benchmarks/)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark record {args.input}: {error}") from error
    destination = append_record(record, args.directory)
    print(json.dumps({"record": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
