import os
import tempfile
import unittest
from unittest import mock

import config
from backend.database import get_db, init_db
from backend.event_outbox import EventOutbox
from backend.events import cleanup_events, save_event, save_external_event


JPEG = b"\xff\xd8event-snapshot\xff\xd9"


class AgentEventOutboxTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.tempdir.name, "manager.db")
        self.snapshot_dir = os.path.join(self.tempdir.name, "events")
        self.patches = [
            mock.patch.object(config, "DATABASE_PATH", self.database_path),
            mock.patch.object(
                config,
                "EVENT_SNAPSHOT_DIR",
                self.snapshot_dir,
            ),
            mock.patch.object(config, "AGENT_ID", "agent-test"),
        ]
        for patch in self.patches:
            patch.start()
        init_db()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.tempdir.cleanup()

    def test_agent_event_and_outbox_are_created_together(self):
        with mock.patch.object(config, "DEPLOYMENT_ROLE", "agent"):
            event_id = save_event(
                "plate_ai",
                "plate",
                "粤B12345",
                "蓝牌",
                0.96,
                {"stream_id": "plate_ai"},
                JPEG,
                manager_received_at_ms=1_900_000_000_000,
            )

        outbox = EventOutbox()
        pending = outbox.next_pending()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["payload"]["label"], "粤B12345")
        self.assertTrue(os.path.isfile(pending["snapshot_path"]))
        with get_db() as conn:
            row = conn.execute(
                "SELECT external_event_id FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        self.assertEqual(
            row["external_event_id"],
            pending["external_event_id"],
        )
        self.assertTrue(outbox.acknowledge(pending["external_event_id"]))
        self.assertEqual(outbox.count(), 0)
        self.assertTrue(os.path.isfile(pending["snapshot_path"]))

    def test_cloud_import_is_idempotent(self):
        payload = {
            "external_event_id": "event-0001",
            "mode": "people_redlight",
            "event_type": "redlight_violation",
            "occurred_at_ms": 1_700_000_000_000,
            "label": "行人闯红灯",
            "color": "red",
            "confidence": 0.91,
            "metadata": {"frame_id": 42},
        }
        with mock.patch.object(config, "DEPLOYMENT_ROLE", "cloud"):
            first = save_external_event("agent-test", payload, JPEG)
            second = save_external_event("agent-test", payload, JPEG)

        self.assertEqual(first, second)
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT occurred_at_ms, metadata_json
                FROM events
                WHERE external_agent_id = ?
                  AND external_event_id = ?
                """,
                ("agent-test", "event-0001"),
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["occurred_at_ms"], payload["occurred_at_ms"])
        self.assertEqual(len(os.listdir(self.snapshot_dir)), 1)

    def test_pending_snapshot_is_exempt_from_retention_cleanup(self):
        with mock.patch.object(config, "DEPLOYMENT_ROLE", "agent"):
            save_event(
                "plate_serial",
                "plate",
                "粤B54321",
                "蓝牌",
                0.94,
                {"stream_id": "plate_serial"},
                JPEG,
                manager_received_at_ms=1,
            )
            cleanup_events()
        pending = EventOutbox().next_pending()
        self.assertTrue(os.path.isfile(pending["snapshot_path"]))


if __name__ == "__main__":
    unittest.main()
