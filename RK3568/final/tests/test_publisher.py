import argparse
import json
import struct
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.publisher import (
    FramePublisher,
    add_publish_args,
    people_overlays,
    plate_overlays,
    validate_publish_args,
)


class _IngestHandler(BaseHTTPRequestHandler):
    payload = None
    token = None
    received = threading.Event()

    def do_POST(self):
        """执行_IngestHandler的“do POST”职责，保持原有输入输出约定。"""
        length = int(self.headers["Content-Length"])
        type(self).payload = self.rfile.read(length)
        type(self).token = self.headers.get("X-Publish-Token")
        type(self).received.set()
        self.send_response(204)
        self.end_headers()

    def log_message(self, _format, *_args):
        """执行_IngestHandler的“log message”职责，保持原有输入输出约定。"""
        pass


class PublisherTests(unittest.TestCase):
    def test_publish_arguments(self):
        """验证“publish arguments”行为契约在重构后保持不变。"""
        parser = argparse.ArgumentParser()
        add_publish_args(parser)
        args = parser.parse_args(
            [
                "--publish",
                "http://127.0.0.1:5000",
                "--publish-token",
                "secret",
                "--publish-stream",
                "plate_ai",
            ]
        )
        validate_publish_args(args)
        self.assertEqual(args.publish_fps, 30)
        self.assertEqual(args.publish_quality, 50)

    def test_overlay_mappings(self):
        """验证“overlay mappings”行为契约在重构后保持不变。"""
        plates = plate_overlays(
            [
                {
                    "plate_no": "粤B12345",
                    "plate_color": "蓝牌",
                    "bbox": [1, 2, 3, 4],
                    "rec_conf": 0.9,
                    "geometry_source": "rk",
                    "recognition_source": "rk",
                }
            ],
            "rk",
        )
        self.assertEqual(plates[0]["type"], "plate")
        self.assertEqual(plates[0]["label"], "粤B12345")
        self.assertNotIn("plate_color", plates[0])
        self.assertNotIn("color_score", plates[0])
        self.assertNotIn("match_state", plates[0])
        self.assertNotIn("serial_iou", plates[0])
        self.assertEqual(plates[0]["color"], "#ef4444")

        people = people_overlays(
            [{"class_id": 7, "bbox": [1, 2, 3, 4], "score": 0.8}],
            [str(index) for index in range(8)],
            {7},
        )
        self.assertEqual(people[0]["type"], "redlight_violation")
        self.assertEqual(people[0]["status"], "violation")

    def test_binary_publish_protocol(self):
        """验证“binary publish protocol”行为契约在重构后保持不变。"""
        _IngestHandler.payload = None
        _IngestHandler.received.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _IngestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        jpeg = b"\xff\xd8unit-test\xff\xd9"

        publisher = FramePublisher(
            f"http://127.0.0.1:{server.server_port}",
            "secret",
            "plate_ai",
            "plate_ai",
            fps=30,
            jpeg_provider=lambda last: (
                None if last == 8 else (jpeg, 8, time.monotonic_ns())
            ),
            printer=lambda _message: None,
        )
        publisher.update(
            overlays=[{"type": "plate", "label": "粤B12345"}],
            serial_data={
                "online": True,
                "sequence": 7,
                "raw_hex": "AA 01 00",
                "count": 1,
                "rois": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}],
            },
        )
        publisher.start()
        self.assertTrue(_IngestHandler.received.wait(2.0))
        publisher.stop()
        server.shutdown()
        server.server_close()

        payload = _IngestHandler.payload
        json_length = struct.unpack(">I", payload[:4])[0]
        meta = json.loads(payload[4 : 4 + json_length].decode("utf-8"))
        self.assertEqual(_IngestHandler.token, "secret")
        self.assertEqual(meta["version"], 1)
        self.assertEqual(len(meta["session_id"]), 32)
        int(meta["session_id"], 16)
        self.assertEqual(meta["frame_id"], 8)
        self.assertNotIn("handover", meta)
        self.assertEqual(meta["serial_data"]["sequence"], 7)
        self.assertEqual(
            meta["serial_data"]["rois"][0],
            {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        )
        self.assertEqual(payload[4 + json_length :], jpeg)


if __name__ == "__main__":
    unittest.main()
