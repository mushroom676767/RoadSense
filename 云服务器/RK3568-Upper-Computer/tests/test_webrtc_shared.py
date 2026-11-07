import asyncio
import unittest

from backend.webrtc import RTCPeerConnection


@unittest.skipUnless(RTCPeerConnection is not None, "aiortc is not installed")
class SharedEncoderQueueTests(unittest.TestCase):
    def setUp(self):
        from backend.webrtc import SharedH264Broadcaster

        self.source = SharedH264Broadcaster("test", 30, 1920, 1080)

    def tearDown(self):
        self.source._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def packet(keyframe=False):
        class Packet:
            is_keyframe = keyframe

        return Packet()

    def test_new_subscriber_waits_for_keyframe(self):
        queue = asyncio.Queue(maxsize=3)
        self.source._subscribers[queue] = True

        self.source._broadcast_packet(self.packet(False))
        self.assertTrue(queue.empty())

        keyframe = self.packet(True)
        self.source._broadcast_packet(keyframe)
        self.assertIs(queue.get_nowait(), keyframe)
        self.assertFalse(self.source._subscribers[queue])

    def test_slow_subscriber_drops_old_gop_and_requests_keyframe(self):
        queue = asyncio.Queue(maxsize=2)
        self.source._subscribers[queue] = False
        queue.put_nowait(self.packet(True))
        queue.put_nowait(self.packet(False))

        self.source._broadcast_packet(self.packet(False))

        self.assertTrue(queue.empty())
        self.assertTrue(self.source._subscribers[queue])
        self.assertTrue(self.source._force_keyframe)
