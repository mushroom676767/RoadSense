"""Durable Agent event queue backed by the local SQLite database."""

import json

from backend.database import get_db


class EventOutbox:
    def next_pending(self):
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT external_event_id, payload_json, snapshot_path,
                       attempts, created_at_ms
                FROM agent_event_outbox
                ORDER BY created_at_ms, external_event_id
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            payload = {}
        return {
            "external_event_id": row["external_event_id"],
            "payload": payload,
            "snapshot_path": row["snapshot_path"],
            "attempts": row["attempts"],
            "created_at_ms": row["created_at_ms"],
        }

    def acknowledge(self, external_event_id):
        with get_db() as conn:
            deleted = conn.execute(
                """
                DELETE FROM agent_event_outbox
                WHERE external_event_id = ?
                """,
                (str(external_event_id),),
            ).rowcount
        return bool(deleted)

    def fail(self, external_event_id, error):
        with get_db() as conn:
            conn.execute(
                """
                UPDATE agent_event_outbox
                SET attempts = attempts + 1, last_error = ?
                WHERE external_event_id = ?
                """,
                (str(error or "")[:512], str(external_event_id)),
            )

    def count(self):
        with get_db() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM agent_event_outbox"
                ).fetchone()[0]
            )
