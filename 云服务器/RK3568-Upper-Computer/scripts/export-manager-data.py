"""Create a consistent, checksummed cloud migration archive."""

import argparse
import hashlib
import json
import posixpath
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

import config


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "manager-data.zip",
    )
    parser.add_argument(
        "--target-event-dir",
        default="/var/lib/rk3568-manager/events",
    )
    args = parser.parse_args()
    source_db = Path(config.DATABASE_PATH)
    if not source_db.is_file():
        raise SystemExit(f"database not found: {source_db}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        snapshot_db = Path(temporary) / "manager.db"
        source = sqlite3.connect(source_db)
        destination = sqlite3.connect(snapshot_db)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()

        conn = sqlite3.connect(snapshot_db)
        conn.row_factory = sqlite3.Row
        try:
            users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            rows = conn.execute(
                "SELECT id, snapshot_path FROM events ORDER BY id"
            ).fetchall()
            snapshots = []
            for row in rows:
                path = Path(row["snapshot_path"])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"event {row['id']} snapshot is missing: {path}"
                    )
                archive_name = f"events/{path.name}"
                snapshots.append((path, archive_name))
                conn.execute(
                    "UPDATE events SET snapshot_path = ? WHERE id = ?",
                    (
                        posixpath.join(args.target_event_dir, path.name),
                        row["id"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        members = {}
        with zipfile.ZipFile(
            args.output,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            database_bytes = snapshot_db.read_bytes()
            archive.writestr("manager.db", database_bytes)
            members["manager.db"] = sha256(database_bytes)
            for path, archive_name in snapshots:
                data = path.read_bytes()
                archive.writestr(archive_name, data)
                members[archive_name] = sha256(data)
            manifest = {
                "version": 1,
                "created_at_ms": int(time.time() * 1000),
                "users": users,
                "events": len(rows),
                "snapshots": len(snapshots),
                "members": members,
            }
            archive.writestr(
                "manifest.json",
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
    print(
        f"exported users={users} events={len(rows)} "
        f"snapshots={len(snapshots)} to {args.output}"
    )


if __name__ == "__main__":
    main()
