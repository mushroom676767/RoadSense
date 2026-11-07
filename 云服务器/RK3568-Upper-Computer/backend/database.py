import os
import sqlite3
from contextlib import contextmanager

import config


@contextmanager
def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(config.DATABASE_PATH)), exist_ok=True)
    os.makedirs(config.EVENT_SNAPSHOT_DIR, exist_ok=True)
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        user_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "role" not in user_columns:
            conn.execute(
                "ALTER TABLE users "
                "ADD COLUMN role TEXT NOT NULL DEFAULT 'viewer'"
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at_ms INTEGER NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                color TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                external_agent_id TEXT,
                external_event_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        event_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "external_agent_id" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN external_agent_id TEXT")
        if "external_event_id" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN external_event_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_time "
            "ON events(occurred_at_ms DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_mode_type "
            "ON events(mode, event_type)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_external "
            "ON events(external_agent_id, external_event_id) "
            "WHERE external_agent_id IS NOT NULL "
            "AND external_event_id IS NOT NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_event_outbox (
                external_event_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_event_outbox_created "
            "ON agent_event_outbox(created_at_ms)"
        )
