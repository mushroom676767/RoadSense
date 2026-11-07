import os
import threading
import time
import unittest
from unittest import mock

import config
from backend.cloud_media import (
    CloudMediaCapacityError,
    CloudMediaService,
    NvencRtspPublisher,
)
from backend.ssh_manager import SSHManager


class _ReadyPublisher:
    last_error = ""

    @staticmethod
    def wait_ready(_timeout):
        return True


class _SocketRecorder:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


class CloudMediaTests(unittest.TestCase):
    def setUp(self):
        self.service = CloudMediaService()
        self.service.ensure_tunnel = mock.Mock()
        self.service._publisher = mock.Mock(
            return_value=_ReadyPublisher()
        )

    def tearDown(self):
        self.service.stop()

    def test_whep_answer_and_delete_are_proxied(self):
        self.service._whep_request = mock.Mock(
            side_effect=[
                (
                    "v=0\r\nanswer",
                    {
                        "Location": "/plate_ai/whep/session?token=opaque",
                        "ETag": "session-etag",
                    },
                ),
                ("", {}),
            ]
        )

        answer = self.service.create_answer(
            "plate_ai", "v=0\r\noffer", "offer"
        )

        self.assertEqual(answer["type"], "answer")
        self.assertEqual(answer["sdp"], "v=0\r\nanswer")
        self.assertEqual(self.service.peer_count, 1)
        self.assertTrue(self.service.touch_peer(answer["peer_id"]))
        self.assertTrue(self.service.close_peer(answer["peer_id"]))
        self.assertEqual(self.service.peer_count, 0)
        self.assertEqual(
            self.service._whep_request.call_args_list,
            [
                mock.call(
                    "POST",
                    stream_id="plate_ai",
                    sdp="v=0\r\noffer",
                ),
                mock.call(
                    "DELETE",
                    location=(
                        "/plate_ai/whep/session?token=opaque"
                    ),
                    etag="session-etag",
                ),
            ],
        )

    def test_capacity_counts_in_progress_negotiations(self):
        original_limit = config.WEBRTC_MAX_VIEWERS
        config.WEBRTC_MAX_VIEWERS = 1
        entered = threading.Event()
        release = threading.Event()
        result = {}

        def whep(method, **_kwargs):
            if method == "DELETE":
                return "", {}
            entered.set()
            release.wait(timeout=2)
            return "v=0\r\nanswer", {
                "Location": "/plate_ai/whep/session",
                "ETag": "etag",
            }

        self.service._whep_request = whep

        def negotiate():
            try:
                result["answer"] = self.service.create_answer(
                    "plate_ai", "v=0\r\nfirst", "offer"
                )
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=negotiate)
        worker.start()
        self.assertTrue(entered.wait(timeout=2))
        try:
            with self.assertRaises(CloudMediaCapacityError):
                self.service.create_answer(
                    "plate_ai", "v=0\r\nsecond", "offer"
                )
        finally:
            release.set()
            worker.join(timeout=3)
            config.WEBRTC_MAX_VIEWERS = original_limit

        self.assertFalse(worker.is_alive())
        self.assertNotIn("error", result)
        self.assertEqual(self.service.peer_count, 1)
        self.service.close_peer(result["answer"]["peer_id"])

    def test_whep_session_url_keeps_opaque_query(self):
        with mock.patch.object(
            config, "MEDIA_MTX_LOCAL_WHEP_PORT", 18889
        ):
            self.assertEqual(
                self.service._whep_url(
                    location=(
                        "http://server/path/whep/session?token=abc"
                    )
                ),
                (
                    "http://127.0.0.1:18889"
                    "/path/whep/session?token=abc"
                ),
            )

    def test_ssh_tunnel_ignores_user_ssh_forwarding_rules(self):
        command = self.service._tunnel_command()
        self.assertEqual(command[command.index("-F") + 1], os.devnull)
        self.assertEqual(command.count("-L"), 2)
        self.assertIn(
            f"ServerAliveInterval={config.MEDIA_MTX_SSH_KEEPALIVE_SECONDS}",
            command,
        )
        self.assertIn(
            f"ServerAliveCountMax={config.MEDIA_MTX_SSH_KEEPALIVE_COUNT}",
            command,
        )

    def test_expired_whep_sessions_do_not_consume_capacity(self):
        original_limit = config.WEBRTC_MAX_VIEWERS
        original_timeout = config.WEBRTC_SESSION_TIMEOUT_SECONDS
        config.WEBRTC_MAX_VIEWERS = 1
        config.WEBRTC_SESSION_TIMEOUT_SECONDS = 20
        self.service._sessions["expired"] = {
            "location": "/plate_ai/whep/expired",
            "etag": "old",
            "last_seen": time.monotonic() - 21,
        }
        self.service._delete_sessions_async = mock.Mock()
        self.service._whep_request = mock.Mock(
            return_value=(
                "v=0\r\nanswer",
                {"Location": "/plate_ai/whep/current", "ETag": "new"},
            )
        )
        try:
            answer = self.service.create_answer(
                "plate_ai", "v=0\r\noffer", "offer"
            )
        finally:
            config.WEBRTC_MAX_VIEWERS = original_limit
            config.WEBRTC_SESSION_TIMEOUT_SECONDS = original_timeout

        self.assertNotEqual(answer["peer_id"], "expired")
        self.assertEqual(self.service.peer_count, 1)
        self.service._delete_sessions_async.assert_called_once()
        self.assertFalse(self.service.touch_peer("expired"))

    def test_touch_peer_refreshes_whep_session_ttl(self):
        self.service._sessions["peer"] = {
            "location": "/plate_ai/whep/peer",
            "etag": "etag",
            "last_seen": 1.0,
        }
        with mock.patch("backend.cloud_media.time.monotonic", return_value=8.0):
            self.assertTrue(self.service.touch_peer("peer"))
        self.assertEqual(self.service._sessions["peer"]["last_seen"], 8.0)


class NvencPublisherTests(unittest.TestCase):
    def test_decode_error_reuses_last_frame_during_source_grace(self):
        publisher = NvencRtspPublisher("plate_ai", mock.Mock())
        frame = mock.Mock()
        frame.width = config.WEBRTC_VIDEO_WIDTH
        frame.height = config.WEBRTC_VIDEO_HEIGHT
        frame.format.name = "yuv420p"
        fallback = mock.Mock()
        fallback.width = config.WEBRTC_VIDEO_WIDTH
        fallback.height = config.WEBRTC_VIDEO_HEIGHT
        fallback.format.name = "yuv420p"
        now = time.monotonic()
        publisher._last_good_jpeg = b"good"
        publisher._last_good_received = now - config.STREAM_STALE_SECONDS - 0.1
        publisher._decode = mock.Mock(
            side_effect=[ValueError("damaged jpeg"), fallback]
        )

        with mock.patch(
            "backend.cloud_media.time.monotonic", return_value=now
        ):
            result = publisher._prepare_frame(
                {"jpeg": b"bad", "received_monotonic": now}, 7
            )

        self.assertIs(result, fallback)
        self.assertEqual(result.pts, 7)

    def test_single_publish_failure_does_not_kill_healthy_tunnel(self):
        invalidate = mock.Mock()
        publisher = NvencRtspPublisher(
            "plate_ai",
            mock.Mock(),
            invalidate,
            tunnel_online=lambda: True,
        )
        publisher._publish_session = mock.Mock(
            side_effect=[
                BrokenPipeError("rtsp disconnected"),
                KeyboardInterrupt(),
            ]
        )
        publisher._stop.wait = mock.Mock()

        with mock.patch(
            "backend.cloud_media.get_latest",
            return_value={"jpeg": b"frame"},
        ):
            with self.assertRaises(KeyboardInterrupt):
                publisher._run()

        invalidate.assert_not_called()

    def test_repeated_publish_failures_rebuild_tunnel(self):
        invalidate = mock.Mock()
        publisher = NvencRtspPublisher(
            "plate_ai",
            mock.Mock(),
            invalidate,
            tunnel_online=lambda: True,
        )
        publisher._publish_session = mock.Mock(
            side_effect=[
                BrokenPipeError("rtsp disconnected")
                for _ in range(5)
            ]
            + [KeyboardInterrupt()]
        )
        publisher._stop.wait = mock.Mock()

        with mock.patch(
            "backend.cloud_media.get_latest",
            return_value={"jpeg": b"frame"},
        ):
            with self.assertRaises(KeyboardInterrupt):
                publisher._run()

        invalidate.assert_called_once()


class SharedLogTests(unittest.TestCase):
    def test_new_viewers_can_receive_bounded_log_history(self):
        socket = _SocketRecorder()
        manager = SSHManager(socket)
        original_limit = config.SHARED_LOG_MAX_CHARS
        config.SHARED_LOG_MAX_CHARS = 12
        try:
            manager._emit_log("\x1b[31mfirst\x1b[0m\n")
            manager._emit_log("second-line\n")
        finally:
            config.SHARED_LOG_MAX_CHARS = original_limit

        self.assertEqual(manager.get_log_snapshot(), "second-line\n")
        self.assertNotIn("\x1b", manager.get_log_snapshot())
        self.assertEqual(socket.events[-1][0], "log_output")


if __name__ == "__main__":
    unittest.main()
