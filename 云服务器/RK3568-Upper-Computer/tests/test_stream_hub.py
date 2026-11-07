import unittest
from unittest import mock

from backend.stream_hub import StreamHub


class FakeSocket:
    def __init__(self):
        self.emitted = []

    def emit(self, event, payload, **kwargs):
        self.emitted.append((event, payload, kwargs))


class StreamHubTests(unittest.TestCase):
    def test_metadata_subscription_does_not_send_jpeg(self):
        socket = FakeSocket()
        hub = StreamHub()
        hub.init(socket)
        hub.connect("browser")
        hub.subscribe("browser", "plate_ai", include_jpeg=False)

        hub.publish("plate_ai", {"frame_id": 7}, b"large-jpeg")

        event, payload, _kwargs = socket.emitted[0]
        self.assertEqual(event, "frame_meta")
        self.assertEqual(payload, {"meta": {"frame_id": 7}})
        self.assertNotIn("jpeg", payload)

    def test_jpeg_subscription_keeps_compatibility_payload(self):
        socket = FakeSocket()
        hub = StreamHub()
        hub.init(socket)
        hub.connect("browser")
        hub.subscribe("browser", "plate_ai", include_jpeg=True)

        hub.publish("plate_ai", {"frame_id": 8}, b"jpeg")

        event, payload, _kwargs = socket.emitted[0]
        self.assertEqual(event, "frame")
        self.assertEqual(payload["jpeg"], b"jpeg")

    def test_jpeg_fallback_is_limited_to_five_fps(self):
        socket = FakeSocket()
        hub = StreamHub()
        hub.init(socket)
        hub.connect("browser")
        hub.subscribe(
            "browser",
            "plate_ai",
            include_jpeg=True,
            jpeg_fps=5,
        )

        with mock.patch(
            "backend.stream_hub.time.monotonic",
            side_effect=[1.0, 1.05, 1.21],
        ):
            hub.publish("plate_ai", {"frame_id": 1}, b"one")
            hub.publish("plate_ai", {"frame_id": 2}, b"two")
            hub.publish("plate_ai", {"frame_id": 3}, b"three")

        self.assertEqual(len(socket.emitted), 1)
        socket.emitted[0][2]["callback"]()
        self.assertEqual(len(socket.emitted), 2)
        self.assertEqual(socket.emitted[1][1]["jpeg"], b"three")


if __name__ == "__main__":
    unittest.main()
