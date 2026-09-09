"""Verify and expand the bundled catalog without network access."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
ARCHIVE_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_catalog(
    archive: Path,
    destination: Path,
    *,
    archive_sha256: str = ARCHIVE_SHA256,
    catalog_sha256: str = CATALOG_SHA256,
) -> Path:
    """Publish only a fully verified catalog; leave existing files untouched."""

    if destination.exists():
        if file_sha256(destination) != catalog_sha256:
            raise ValueError(f"existing catalog checksum mismatch: {destination}")
        return destination
    if file_sha256(archive) != archive_sha256:
        raise ValueError(f"catalog archive checksum mismatch: {archive}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".catalog-", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            with gzip.open(archive, "rb") as source:
                shutil.copyfileobj(source, output)
        if file_sha256(temporary) != catalog_sha256:
            raise ValueError("expanded catalog checksum mismatch")
        # An exclusive link also prevents concurrent preparation from replacing
        # a catalog another process has just written.
        try:
            destination.hardlink_to(temporary)
        except FileExistsError:
            if file_sha256(destination) != catalog_sha256:
                raise ValueError(f"concurrent catalog checksum mismatch: {destination}")
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    catalog = prepare_catalog(ROOT / "data/catalog.jsonl.gz", ROOT / "data/catalog.jsonl")
    print(json.dumps({"catalog": str(catalog), "sha256": CATALOG_SHA256, "ready": True}))


if __name__ == "__main__":
    main()
