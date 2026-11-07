"""Verify and atomically install a manager data migration archive."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/var/lib/rk3568-manager/manager.db"),
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        default=Path("/var/lib/rk3568-manager/events"),
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("/var/backups/rk3568-manager"),
    )
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = args.backup_dir / f"data-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    args.database.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names or "manager.db" not in names:
            raise SystemExit("archive is missing manifest.json or manager.db")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("version") != 1:
            raise SystemExit("unsupported migration archive version")
        members = manifest.get("members")
        if not isinstance(members, dict):
            raise SystemExit("invalid migration manifest")
        for name, expected in members.items():
            if name not in names or sha256(archive.read(name)) != expected:
                raise SystemExit(f"checksum mismatch: {name}")
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise SystemExit(f"unsafe archive path: {name}")

        with tempfile.TemporaryDirectory(
            dir=args.database.parent
        ) as temporary:
            staging = Path(temporary)
            staged_db = staging / "manager.db"
            staged_db.write_bytes(archive.read("manager.db"))
            staged_events = staging / "events"
            staged_events.mkdir()
            for name in members:
                if not name.startswith("events/"):
                    continue
                target = staged_events / Path(name).name
                target.write_bytes(archive.read(name))

            if args.database.exists():
                shutil.copy2(args.database, backup / "manager.db")
            if args.event_dir.exists():
                os.replace(args.event_dir, backup / "events")
            os.replace(staged_db, args.database)
            os.replace(staged_events, args.event_dir)

    print(
        f"imported users={manifest.get('users')} "
        f"events={manifest.get('events')} "
        f"snapshots={manifest.get('snapshots')}; backup={backup}"
    )


if __name__ == "__main__":
    main()
