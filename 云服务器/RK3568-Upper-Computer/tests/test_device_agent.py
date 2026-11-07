import queue
import threading
import unittest
from unittest import mock

from backend.device_agent import DeviceAgent


class FakePublisher:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.started = 0
        self.stopped = 0
        self.stats = {
            "running": True,
            "ready": True,
            "encoded_frames": 12,
            "encode_ms": 1.5,
            "upload_mbps": 4.0,
            "error": "",
        }

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def bare_agent():
    agent = DeviceAgent.__new__(DeviceAgent)
    agent._meta_lock = threading.Lock()
    agent._latest_meta = None
    agent._publisher_lock = threading.RLock()
    agent._media_wakeup = threading.Event()
    agent._latest_jpeg_stream = None
    agent._latest_jpeg_at = 0.0
    agent._last_media_error = ""
    agent._publisher = None
    agent._publisher_stream = None
    return agent


class DeviceAgentDesiredStateTests(unittest.TestCase):
    def test_serial_roi_metadata_is_relayed_without_coordinate_rewrite(self):
        agent = bare_agent()
        meta = {
            "version": 1,
            "stream_id": "plate_serial",
            "mode": "plate_serial",
            "overlays": [{"type": "plate", "source": "rk-roi"}],
            "serial_data": {
                "online": True,
                "sequence": 8,
                "received_at_ms": 123456,
                "age_ms": 1.5,
                "raw_hex": "AA 01 64 00 C8 00 90 01 2C 01",
                "count": 1,
                "rois": [
                    {"x1": 100, "y1": 200, "x2": 400, "y2": 300}
                ],
            },
        }

        agent._on_ingest("plate_serial", meta, b"jpeg", [])

        self.assertEqual(agent._latest_meta["stream_id"], "plate_serial")
        self.assertEqual(agent._latest_meta["meta"], meta)
        self.assertIsNot(agent._latest_meta["meta"], meta)
        self.assertEqual(
            agent._latest_meta["meta"]["serial_data"]["rois"][0],
            {"x1": 100, "y1": 200, "x2": 400, "y2": 300},
        )

    def test_count_zero_replaces_previous_serial_results(self):
        agent = bare_agent()
        agent._on_ingest(
            "plate_serial",
            {
                "overlays": [{"type": "plate", "label": "粤B12345"}],
                "serial_data": {"sequence": 1, "count": 1, "rois": [{}]},
            },
            b"jpeg",
            [],
        )
        cleared = {
            "overlays": [],
            "results": [],
            "serial_data": {
                "online": True,
                "sequence": 2,
                "count": 0,
                "rois": [],
                "raw_hex": "AA 00",
            },
        }

        agent._on_ingest("plate_serial", cleared, b"jpeg", [])

        self.assertEqual(agent._latest_meta["meta"], cleared)
        self.assertEqual(agent._latest_meta["meta"]["overlays"], [])

    def test_heartbeat_ack_requeues_newer_desired_state(self):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent._generation = 7
        agent._desired_queue = queue.Queue(maxsize=1)

        agent._receive_heartbeat_ack(
            {
                "ok": True,
                "desired": {
                    "generation": 8,
                    "mode": None,
                    "source": "hdmi",
                    "running": False,
                },
            }
        )

        desired = agent._desired_queue.get_nowait()
        self.assertEqual(desired["generation"], 8)
        self.assertFalse(desired["running"])

    def test_heartbeat_ack_ignores_stale_or_invalid_state(self):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent._generation = 7
        agent._desired_queue = queue.Queue(maxsize=1)

        agent._receive_heartbeat_ack(
            {
                "ok": True,
                "desired": {
                    "generation": 7,
                    "running": True,
                },
            }
        )
        agent._receive_heartbeat_ack({"ok": False})

        self.assertTrue(agent._desired_queue.empty())

    @mock.patch("backend.device_agent.config.MEDIA_MTX_INGEST_TRANSPORT", "srt")
    @mock.patch("backend.device_agent.NvencSrtPublisher", FakePublisher)
    def test_first_jpeg_starts_one_publisher_for_terminal_launch_too(self):
        agent = bare_agent()
        with mock.patch("backend.device_agent.time.monotonic", return_value=10.0):
            agent._on_ingest("plate_ai", {}, b"jpeg", [])
        self.assertTrue(agent._reconcile_media(now=10.0))
        publisher = agent._publisher
        self.assertEqual(agent._publisher_stream, "plate_ai")
        self.assertEqual(publisher.started, 1)

        agent._on_ingest("plate_ai", {}, b"same-jpeg", [])
        self.assertTrue(agent._reconcile_media(now=10.1))
        self.assertIs(agent._publisher, publisher)
        self.assertEqual(publisher.started, 1)

    @mock.patch("backend.device_agent.config.MEDIA_MTX_INGEST_TRANSPORT", "srt")
    @mock.patch("backend.device_agent.NvencSrtPublisher", FakePublisher)
    def test_latest_stream_switches_publisher_and_timeout_stops_it(self):
        agent = bare_agent()
        with mock.patch("backend.device_agent.time.monotonic", return_value=20.0):
            agent._on_ingest("plate_ai", {}, b"jpeg", [])
        agent._reconcile_media(now=20.0)
        old = agent._publisher

        with mock.patch("backend.device_agent.time.monotonic", return_value=21.0):
            agent._on_ingest("people_redlight", {}, b"jpeg", [])
        agent._reconcile_media(now=21.0)
        self.assertEqual(old.stopped, 1)
        self.assertEqual(agent._publisher_stream, "people_redlight")
        current = agent._publisher

        self.assertFalse(agent._reconcile_media(now=34.0))
        self.assertEqual(current.stopped, 1)
        self.assertIsNone(agent._publisher)


if __name__ == "__main__":
    unittest.main()
