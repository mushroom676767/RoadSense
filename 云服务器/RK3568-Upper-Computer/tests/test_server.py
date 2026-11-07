import json
import os
import struct
import tempfile
import time
import unittest
from unittest import mock

import config
from app import create_app
from backend.database import get_db
from backend import events as events_module
from backend.events import event_detector
from backend import ingest as ingest_module
from backend.ssh_manager import _REMOTE_PROCESS_CLEANUP
from backend.webrtc import aiortc_h264, webrtc_service


JPEG = b"\xff\xd8\xff\xd9"


class ServerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        config.DATABASE_PATH = os.path.join(cls.tempdir.name, "manager.db")
        config.EVENT_SNAPSHOT_DIR = os.path.join(cls.tempdir.name, "events")
        config.ADMIN_USERNAME = "testadmin"
        config.ADMIN_PASSWORD = "test-password"
        config.VISITOR_USERNAME = "visitor"
        config.VISITOR_PASSWORD = "123456"
        config.PUBLISH_TOKEN = "publish-secret"
        config.PUBLISH_URL = "http://127.0.0.1:5000"
        config.PLATE_STABLE_FRAMES = 3
        config.EVENT_COOLDOWN_SECONDS = 0
        event_detector._plates.clear()
        event_detector._last_emitted.clear()
        event_detector._redlight_active.clear()
        event_detector._explicit_ids.clear()
        cls.app, cls.socketio = create_app({"TESTING": True})
        cls.client = cls.app.test_client()
        login = cls.client.post(
            "/api/login",
            json={"username": "testadmin", "password": "test-password"},
        )
        assert login.status_code == 200, login.get_json()
        cls.headers = {
            "Authorization": f"Bearer {login.get_json()['token']}"
        }
        visitor_login = cls.client.post(
            "/api/login",
            json={"username": "visitor", "password": "123456"},
        )
        assert visitor_login.status_code == 200, visitor_login.get_json()
        cls.visitor_headers = {
            "Authorization": f"Bearer {visitor_login.get_json()['token']}"
        }

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def setUp(self):
        with get_db() as conn:
            rows = conn.execute("SELECT snapshot_path FROM events").fetchall()
            conn.execute("DELETE FROM events")
        for row in rows:
            try:
                os.remove(row["snapshot_path"])
            except FileNotFoundError:
                pass
        event_detector._plates.clear()
        event_detector._last_emitted.clear()
        event_detector._redlight_active.clear()
        event_detector._explicit_ids.clear()
        with ingest_module._lock:
            ingest_module._latest.clear()

    def _ingest(self, frame_id, overlays=None, token="publish-secret", **meta):
        metadata = {
            "version": 1,
            "stream_id": "plate_ai",
            "mode": "plate_ai",
            "frame_id": frame_id,
            "sent_at_ms": int(time.time() * 1000),
            "overlays": overlays or [],
            "results": [],
            "events": [],
            "metrics": {"capture_fps": 59.4},
            **meta,
        }
        encoded = json.dumps(metadata).encode("utf-8")
        body = struct.pack(">I", len(encoded)) + encoded + JPEG
        return self.client.post(
            "/api/ingest",
            data=body,
            headers={
                "X-Publish-Token": token,
                "Content-Type": "application/octet-stream",
            },
        )

    def _ingest_serial_plate_and_redlight(
        self,
        serial_start_frame=1,
        redlight_frame=1,
    ):
        serial_plate = {
            "type": "plate",
            "source": "rk-roi",
            "label": "粤B54321",
            "score": 0.91,
            "rec_score": 0.96,
            "bbox": [10, 20, 100, 60],
        }
        for frame_id in range(
            serial_start_frame,
            serial_start_frame + config.PLATE_STABLE_FRAMES,
        ):
            self.assertEqual(
                self._ingest(
                    frame_id,
                    overlays=[serial_plate],
                    stream_id="plate_serial",
                    mode="plate_serial",
                ).status_code,
                204,
            )

        violation = {
            "type": "redlight_violation",
            "source": "rk",
            "label": "行人闯红灯",
            "score": 0.93,
            "status": "violation",
            "bbox": [20, 30, 90, 180],
        }
        self.assertEqual(
            self._ingest(
                redlight_frame,
                overlays=[violation],
                stream_id="people_redlight",
                mode="people_redlight",
            ).status_code,
            204,
        )

    def test_protected_endpoints_and_modes(self):
        self.assertEqual(self.client.get("/api/streams").status_code, 401)
        self.assertEqual(self.client.get("/api/webrtc/config").status_code, 401)
        response = self.client.get("/api/modes", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.get_json()),
            {"plate_serial", "plate_ai", "people_redlight"},
        )

        rtc = self.client.get(
            "/api/webrtc/config", headers=self.headers
        ).get_json()
        self.assertEqual(rtc["transport"], config.VIDEO_TRANSPORT)
        self.assertEqual(rtc["fallback"], "jpeg")
        self.assertIsInstance(rtc["iceServers"], list)
        self.assertEqual(rtc["video_width"], 1920)
        self.assertEqual(rtc["video_height"], 1080)
        self.assertEqual(rtc["video_bitrate"], config.WEBRTC_VIDEO_BITRATE)
        self.assertEqual(
            rtc["video_max_bitrate"], config.WEBRTC_VIDEO_MAX_BITRATE
        )
        self.assertEqual(rtc["video_encoder"], "h264_nvenc")
        self.assertTrue(rtc["shared_encoding"])
        self.assertEqual(rtc["max_viewers"], 12)
        if aiortc_h264 is not None:
            self.assertEqual(
                aiortc_h264.DEFAULT_BITRATE, config.WEBRTC_VIDEO_BITRATE
            )
            self.assertEqual(
                aiortc_h264.MAX_BITRATE,
                config.WEBRTC_VIDEO_MAX_BITRATE,
            )

    def test_hls_media_authorization_accepts_login_cookie_only(self):
        self.assertEqual(
            self.client.get("/api/webrtc/media-authorize").status_code,
            204,
        )
        anonymous = self.app.test_client()
        self.assertEqual(
            anonymous.get("/api/webrtc/media-authorize").status_code,
            401,
        )

    def test_webrtc_heartbeat_is_authenticated_and_reports_liveness(self):
        self.assertEqual(
            self.client.post(
                "/api/webrtc/heartbeat", json={"peer_id": "peer"}
            ).status_code,
            401,
        )
        service = mock.Mock()
        service.touch_peer.side_effect = [True, False]
        with mock.patch("backend.webrtc._active_service", return_value=service):
            alive = self.client.post(
                "/api/webrtc/heartbeat",
                headers=self.headers,
                json={"peer_id": "peer"},
            )
            missing = self.client.post(
                "/api/webrtc/heartbeat",
                headers=self.headers,
                json={"peer_id": "missing"},
            )
        self.assertEqual(alive.status_code, 200)
        self.assertTrue(alive.get_json()["alive"])
        self.assertEqual(missing.status_code, 200)
        self.assertFalse(missing.get_json()["alive"])

    def test_serial_pose_metrics_are_preserved_without_server_rewrite(self):
        metrics = {
            "detect_rga_ms": 1.2,
            "detect_npu_ms": 7.2,
            "detect_output_ms": 0.2,
            "detect_decode_ms": 0.2,
            "detect_total_ms": 7.6,
        }
        response = self._ingest(
            1,
            stream_id="plate_serial",
            mode="plate_serial",
            metrics=metrics,
        )
        self.assertEqual(response.status_code, 204)
        latest = ingest_module.get_latest("plate_serial")
        self.assertEqual(latest["meta"]["metrics"], metrics)

    def test_visitor_can_view_but_cannot_control_devices(self):
        user = self.client.get(
            "/api/user",
            headers=self.visitor_headers,
        )
        self.assertEqual(user.status_code, 200)
        self.assertEqual(user.get_json()["role"], "viewer")
        self.assertFalse(user.get_json()["can_control"])

        self.assertEqual(
            self.client.get(
                "/api/mode/status",
                headers=self.visitor_headers,
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                "/api/events",
                headers=self.visitor_headers,
            ).status_code,
            200,
        )

        start = self.client.post(
            "/api/mode/start",
            headers=self.visitor_headers,
            json={"mode": "plate_ai", "source": "hdmi"},
        )
        self.assertEqual(start.status_code, 403)
        self.assertIn("不能进行设备控制", start.get_json()["error"])

        stop = self.client.post(
            "/api/mode/stop",
            headers=self.visitor_headers,
        )
        self.assertEqual(stop.status_code, 403)

    def test_force_terminate_endpoint_is_removed(self):
        response = self.client.post(
            "/api/mode/terminate",
            headers=self.headers,
        )
        self.assertIn(response.status_code, (404, 405))

    def test_socket_connect_receives_shared_state_snapshots(self):
        ssh = self.app.extensions["ssh_manager"]
        ssh._emit_log("shared-board-log\n")
        token = self.visitor_headers["Authorization"].removeprefix(
            "Bearer "
        )
        socket_client = self.socketio.test_client(
            self.app,
            auth={"token": token},
        )
        try:
            self.assertTrue(socket_client.is_connected())
            received = {
                item["name"]: item["args"][0]
                for item in socket_client.get_received()
            }
            self.assertIn("shared-board-log", received["log_snapshot"]["text"])
            self.assertIn("running", received["mode_status"])
        finally:
            socket_client.disconnect()

    def test_public_security_headers(self):
        response = self.client.get(
            "/api/config",
            headers={"X-Forwarded-Proto": "https"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["X-Content-Type-Options"],
            "nosniff",
        )
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn(
            "default-src 'self'",
            response.headers["Content-Security-Policy"],
        )
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertIn(
            "max-age=31536000",
            response.headers["Strict-Transport-Security"],
        )

    def test_frontend_html_is_revalidated_after_updates(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "no-cache, must-revalidate",
        )
        response.close()

    def test_damaged_password_hash_is_rejected_without_server_error(self):
        with get_db() as conn:
            current = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                ("testadmin",),
            ).fetchone()["password_hash"]
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                ("not-a-bcrypt-hash", "testadmin"),
            )
        try:
            response = self.client.post(
                "/api/login",
                json={"username": "testadmin", "password": "test-password"},
            )
            self.assertEqual(response.status_code, 401)
        finally:
            with get_db() as conn:
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (current, "testadmin"),
                )

    def test_ingest_validation_and_stream_status(self):
        self.assertEqual(self._ingest(1, token="wrong").status_code, 401)
        self.assertEqual(self._ingest(1, version=2).status_code, 400)
        self.assertEqual(self._ingest(1).status_code, 204)
        streams = self.client.get(
            "/api/streams", headers=self.headers
        ).get_json()
        self.assertTrue(streams["plate_ai"]["online"])
        self.assertEqual(streams["plate_ai"]["frame_id"], 1)

    def test_stale_stream_is_not_a_live_webrtc_source(self):
        self.assertEqual(self._ingest(1).status_code, 204)
        self.assertIsNotNone(
            ingest_module.get_latest(
                "plate_ai",
                max_age_seconds=config.STREAM_STALE_SECONDS,
            )
        )
        with ingest_module._lock:
            ingest_module._latest["plate_ai"]["received_monotonic"] -= (
                config.STREAM_STALE_SECONDS + 1
            )
        self.assertIsNone(
            ingest_module.get_latest(
                "plate_ai",
                max_age_seconds=config.STREAM_STALE_SECONDS,
            )
        )
        self.assertFalse(ingest_module.is_stream_online("plate_ai"))

    @unittest.skipUnless(webrtc_service.available, "aiortc is not installed")
    def test_webrtc_offer_rejects_offline_stream(self):
        response = self.client.post(
            "/api/webrtc/offer",
            headers=self.headers,
            json={
                "stream_id": "plate_ai",
                "type": "offer",
                "sdp": "v=0\r\n",
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["code"], "stream_offline")

    def test_new_publish_session_can_restart_frame_sequence(self):
        first_session = "a" * 32
        second_session = "b" * 32
        self.assertEqual(
            self._ingest(100, session_id=first_session).status_code,
            204,
        )
        self.assertEqual(
            self._ingest(1, session_id=first_session).status_code,
            204,
        )
        streams = self.client.get(
            "/api/streams", headers=self.headers
        ).get_json()
        self.assertEqual(streams["plate_ai"]["frame_id"], 100)

        self.assertEqual(
            self._ingest(1, session_id=second_session).status_code,
            204,
        )
        streams = self.client.get(
            "/api/streams", headers=self.headers
        ).get_json()
        self.assertEqual(streams["plate_ai"]["frame_id"], 1)
        self.assertEqual(
            streams["plate_ai"]["session_id"],
            second_session,
        )

    def test_plate_event_is_saved_once_until_target_disappears(self):
        plate = {
            "type": "plate",
            "source": "rk",
            "label": "粤B12345",
            "score": 0.91,
            "rec_score": 0.95,
            "bbox": [10, 20, 100, 60],
        }
        for frame_id in range(10, 16):
            self.assertEqual(
                self._ingest(frame_id, overlays=[plate]).status_code, 204
            )
        events = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()
        matching = [
            item for item in events["items"] if item["label"] == "粤B12345"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["color"], "")

        self._ingest(16, overlays=[])
        for frame_id in range(17, 20):
            self._ingest(frame_id, overlays=[plate])
        events = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()
        matching = [
            item for item in events["items"] if item["label"] == "粤B12345"
        ]
        self.assertEqual(len(matching), 2)

    def test_rk_old_clock_uses_manager_receive_time_for_archive(self):
        rk_old_time_ms = 1_718_669_455_692
        before_ms = int(time.time() * 1000)
        violation = {
            "type": "redlight_violation",
            "source": "rk",
            "label": "行人闯红灯",
            "score": 0.93,
            "status": "violation",
            "bbox": [20, 30, 90, 180],
        }
        response = self._ingest(
            1,
            overlays=[violation],
            stream_id="people_redlight",
            mode="people_redlight",
            sent_at_ms=rk_old_time_ms,
        )
        after_ms = int(time.time() * 1000)

        self.assertEqual(response.status_code, 204)
        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertGreaterEqual(items[0]["occurred_at_ms"], before_ms)
        self.assertLessEqual(items[0]["occurred_at_ms"], after_ms)
        self.assertNotEqual(items[0]["occurred_at_ms"], rk_old_time_ms)
        detail = self.client.get(
            f"/api/events/{items[0]['id']}", headers=self.headers
        ).get_json()
        self.assertEqual(detail["metadata"]["sent_at_ms"], rk_old_time_ms)
        self.assertEqual(
            detail["metadata"]["manager_received_at_ms"],
            items[0]["occurred_at_ms"],
        )
        with get_db() as conn:
            snapshot_path = conn.execute(
                "SELECT snapshot_path FROM events WHERE id = ?",
                (items[0]["id"],),
            ).fetchone()["snapshot_path"]
        self.assertTrue(os.path.isfile(snapshot_path))

    def test_event_write_failure_retries_without_breaking_ingest(self):
        violation = {
            "type": "redlight_violation",
            "source": "rk",
            "label": "行人闯红灯",
            "score": 0.93,
            "status": "violation",
            "bbox": [20, 30, 90, 180],
        }
        original_save = events_module.save_event
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("simulated archive failure")
            return original_save(*args, **kwargs)

        with mock.patch(
            "backend.events.save_event", side_effect=fail_once
        ), mock.patch("backend.events.logger.exception"):
            first = self._ingest(
                1,
                overlays=[violation],
                stream_id="people_redlight",
                mode="people_redlight",
            )
            second = self._ingest(
                2,
                overlays=[violation],
                stream_id="people_redlight",
                mode="people_redlight",
            )

        self.assertEqual(first.status_code, 204)
        self.assertEqual(second.status_code, 204)
        self.assertEqual(attempts, 2)
        self.assertEqual(
            self.client.get(
                "/api/events", headers=self.headers
            ).get_json()["total"],
            1,
        )

    def test_invalid_explicit_events_and_scores_fall_back_to_overlays(self):
        plate = {
            "type": "plate",
            "source": "rk",
            "label": "粤B54321",
            "score": "invalid",
            "rec_score": "invalid",
            "bbox": [10, 20, 100, 60],
        }
        for frame_id in range(1, config.PLATE_STABLE_FRAMES + 1):
            response = self._ingest(
                frame_id,
                overlays=[plate],
                events=[{}, "invalid"],
            )
            self.assertEqual(response.status_code, 204)

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "粤B54321")
        self.assertEqual(items[0]["confidence"], 0)

    def test_serial_archive_accepts_the_rk_roi_source(self):
        plate = {
            "type": "plate",
            "source": "rk-roi",
            "label": "粤B00001",
            "score": 0.91,
            "rec_score": 0.95,
            "bbox": [10, 20, 100, 60],
        }
        for frame_id in range(1, config.PLATE_STABLE_FRAMES + 1):
            self.assertEqual(
                self._ingest(
                    frame_id,
                    overlays=[plate],
                    stream_id="plate_serial",
                    mode="plate_serial",
                ).status_code,
                204,
            )

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertTrue(all(item["mode"] == "plate_serial" for item in items))
        self.assertTrue(all("source" not in item for item in items))

    def test_serial_archive_groups_every_plate_into_one_snapshot(self):
        overlays = [
            {
                "type": "plate",
                "source": "rk-roi",
                "label": "粤B10001",
                "score": 0.94,
                "rec_score": 0.96,
                "bbox": [10, 20, 100, 60],
            },
            {
                "type": "plate",
                "source": "rk-roi",
                "label": "粤B10002",
                "score": 0.92,
                "rec_score": 0.95,
                "bbox": [200, 30, 310, 75],
            },
        ]
        for frame_id in range(1, config.PLATE_STABLE_FRAMES + 1):
            self.assertEqual(
                self._ingest(
                    frame_id,
                    overlays=overlays,
                    stream_id="plate_serial",
                    mode="plate_serial",
                ).status_code,
                204,
            )

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "粤B10001、粤B10002")
        self.assertTrue(all(item["mode"] == "plate_serial" for item in items))
        detail = self.client.get(
            f"/api/events/{items[0]['id']}", headers=self.headers
        ).get_json()
        self.assertEqual(
            detail["metadata"]["plate_labels"],
            ["粤B10001", "粤B10002"],
        )
        self.assertEqual(detail["metadata"]["plate_count"], 2)

    def test_serial_archive_groups_intermittent_secondary_plate(self):
        primary = {
            "type": "plate",
            "source": "rk-roi",
            "label": "粤B20001",
            "score": 0.95,
            "rec_score": 0.97,
            "bbox": [10, 20, 100, 60],
        }
        secondary = {
            "type": "plate",
            "source": "rk-roi",
            "label": "粤B20002",
            "score": 0.86,
            "rec_score": 0.89,
            "bbox": [200, 30, 310, 75],
        }
        frames = (
            [primary, secondary],
            [primary],
            [primary, secondary],
            [primary],
            [primary, secondary],
        )
        for frame_id, overlays in enumerate(frames, 1):
            self.assertEqual(
                self._ingest(
                    frame_id,
                    overlays=overlays,
                    stream_id="plate_serial",
                    mode="plate_serial",
                ).status_code,
                204,
            )

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "粤B20001、粤B20002")

    def test_ai_archive_groups_every_plate_into_one_snapshot(self):
        overlays = [
            {
                "type": "plate",
                "source": "rk",
                "label": "粤A10001",
                "score": 0.94,
                "rec_score": 0.96,
                "bbox": [10, 20, 100, 60],
            },
            {
                "type": "plate",
                "source": "rk",
                "label": "粤A10002",
                "score": 0.92,
                "rec_score": 0.94,
                "bbox": [200, 30, 310, 75],
            },
        ]
        for frame_id in range(1, config.PLATE_STABLE_FRAMES + 1):
            self.assertEqual(
                self._ingest(
                    frame_id,
                    overlays=overlays,
                    stream_id="plate_ai",
                    mode="plate_ai",
                ).status_code,
                204,
            )

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "粤A10001、粤A10002")

    def test_redlight_archive_lists_total_and_violation_counts(self):
        overlays = [
            {
                "type": "person",
                "source": "rk",
                "label": "person",
                "score": 0.88,
                "status": "normal",
                "bbox": [10, 20, 50, 180],
            },
            {
                "type": "person",
                "source": "rk",
                "label": "person",
                "score": 0.86,
                "status": "normal",
                "bbox": [80, 20, 120, 180],
            },
            {
                "type": "redlight_violation",
                "source": "rk",
                "label": "run_red_light",
                "score": 0.95,
                "status": "violation",
                "bbox": [150, 20, 190, 180],
            },
        ]
        self.assertEqual(
            self._ingest(
                1,
                overlays=overlays,
                stream_id="people_redlight",
                mode="people_redlight",
            ).status_code,
            204,
        )

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "总行人数：3，闯红灯人数：1")
        detail = self.client.get(
            f"/api/events/{items[0]['id']}", headers=self.headers
        ).get_json()
        self.assertEqual(detail["metadata"]["person_count"], 3)
        self.assertEqual(detail["metadata"]["redlight_count"], 1)

    def test_database_failure_removes_uncommitted_snapshot(self):
        before = set(os.listdir(config.EVENT_SNAPSHOT_DIR))
        with mock.patch(
            "backend.events.get_db",
            side_effect=OSError("simulated database failure"),
        ):
            with self.assertRaises(OSError):
                events_module.save_event(
                    "plate_ai",
                    "plate",
                    "粤B12345",
                    "蓝牌",
                    0.95,
                    {
                        "sent_at_ms": 1_718_669_455_692,
                        "manager_received_at_ms": int(time.time() * 1000),
                    },
                    JPEG,
                    manager_received_at_ms=int(time.time() * 1000),
                )
        self.assertEqual(
            set(os.listdir(config.EVENT_SNAPSHOT_DIR)),
            before,
        )

    def test_serial_plate_and_redlight_events_are_saved(self):
        self._ingest_serial_plate_and_redlight()

        items = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(
            {(item["mode"], item["event_type"]) for item in items},
            {
                ("plate_serial", "plate"),
                ("people_redlight", "redlight_violation"),
            },
        )

    def test_only_admin_can_clear_event_archive_and_snapshots(self):
        self.test_serial_plate_and_redlight_events_are_saved()
        with get_db() as conn:
            snapshot_paths = [
                row["snapshot_path"]
                for row in conn.execute(
                    "SELECT snapshot_path FROM events"
                ).fetchall()
            ]
        self.assertTrue(snapshot_paths)
        self.assertTrue(all(os.path.isfile(path) for path in snapshot_paths))

        self.assertEqual(
            self.client.post("/api/events/clear").status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/api/events/clear", headers=self.visitor_headers
            ).status_code,
            403,
        )
        response = self.client.post(
            "/api/events/clear", headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["deleted"],
            len(snapshot_paths),
        )
        self.assertFalse(any(os.path.exists(path) for path in snapshot_paths))
        self.assertEqual(
            self.client.get(
                "/api/events", headers=self.headers
            ).get_json()["total"],
            0,
        )
        self.assertFalse(event_detector._plates)
        self.assertFalse(event_detector._last_emitted)
        self.assertFalse(event_detector._redlight_active)
        self.assertFalse(event_detector._explicit_ids)

        self._ingest_serial_plate_and_redlight(
            serial_start_frame=4,
            redlight_frame=2,
        )
        recreated = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        self.assertEqual(
            {(item["mode"], item["event_type"]) for item in recreated},
            {
                ("plate_serial", "plate"),
                ("people_redlight", "redlight_violation"),
            },
        )

    def test_event_snapshot_and_csv_are_authenticated(self):
        events = self.client.get(
            "/api/events", headers=self.headers
        ).get_json()["items"]
        if not events:
            self.test_plate_event_is_saved_once_until_target_disappears()
            events = self.client.get(
                "/api/events", headers=self.headers
            ).get_json()["items"]
        event_id = events[0]["id"]
        self.assertEqual(
            self.client.get(f"/api/events/{event_id}/snapshot").status_code,
            401,
        )
        snapshot = self.client.get(
            f"/api/events/{event_id}/snapshot", headers=self.headers
        )
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.data, JPEG)
        snapshot.close()
        csv_response = self.client.get(
            "/api/events/export.csv", headers=self.headers
        )
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("text/csv", csv_response.content_type)
        csv_response.close()

    def test_ssh_command_quotes_values_and_masks_token(self):
        ssh = self.app.extensions["ssh_manager"]
        command, display, source_name = ssh._build_command(
            "people_redlight", "hdmi"
        )
        self.assertNotIn("--source", command)
        self.assertIn(config.PUBLISH_TOKEN, command)
        self.assertNotIn(config.PUBLISH_TOKEN, display)
        self.assertIn("********", display)
        self.assertEqual(source_name, "HDMI采集卡")

    def test_serial_mode_runs_with_preserved_sudo_environment(self):
        ssh = self.app.extensions["ssh_manager"]
        command, display, _ = ssh._build_command("plate_serial", "hdmi")
        expected = "exec sudo -E python3 -u fpga_serial.py"
        self.assertIn(expected, command)
        self.assertIn(expected, display)

    def test_ssh_cleanup_is_limited_to_known_entrypoints(self):
        self.assertIn("fpga_ai", _REMOTE_PROCESS_CLEANUP)
        self.assertIn("fpga_serial", _REMOTE_PROCESS_CLEANUP)
        self.assertIn("people_redlight", _REMOTE_PROCESS_CLEANUP)
        self.assertIn("$2 ~ /^python3/", _REMOTE_PROCESS_CLEANUP)
        self.assertNotIn("pkill python", _REMOTE_PROCESS_CLEANUP)


if __name__ == "__main__":
    unittest.main()
