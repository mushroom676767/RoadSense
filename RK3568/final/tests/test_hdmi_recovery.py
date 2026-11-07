import queue
import unittest

from core.native_capture import (
    NativeMppHdmiReceiver,
    NativeNv12Frame,
    NativeNv12Info,
)


class _FakeCapture:
    def __init__(self):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.released = []

    def _release_native_frame(self, token):
        """通过帧token减少Native引用计数，使空闲槽可被解码器复用。"""
        self.released.append(token)


class HdmiRecoveryTests(unittest.TestCase):
    def test_usb_disconnect_errors_are_recoverable(self):
        """验证“usb disconnect errors are recoverable”行为契约在重构后保持不变。"""
        recoverable = (
            "VIDIOC_DQBUF failed: No such device",
            "poll V4L2 failed: Input/output error",
            "poll V4L2 device event 0x18",
            "V4L2 frame timeout for 3 consecutive seconds",
            "open /dev/video9 failed: No such file or directory",
            "VIDIOC_QBUF failed: No such device",
        )
        for message in recoverable:
            with self.subTest(message=message):
                self.assertTrue(
                    NativeMppHdmiReceiver._is_recoverable_capture_error(
                        RuntimeError(message)
                    )
                )

        self.assertFalse(
            NativeMppHdmiReceiver._is_recoverable_capture_error(
                RuntimeError("RGA model preprocessing failed")
            )
        )

    def test_reconnect_discards_owned_frames_before_capture_close(self):
        """验证“reconnect discards owned frames before capture close”行为契约在重构后保持不变。"""
        frame_queue = queue.Queue()
        receiver = NativeMppHdmiReceiver(frame_queue)
        capture = _FakeCapture()
        info = NativeNv12Info()
        info.token = 42
        frame_queue.put(NativeNv12Frame(capture, info))

        receiver._discard_queued_frames()

        self.assertTrue(frame_queue.empty())
        self.assertEqual(capture.released, [42])


if __name__ == "__main__":
    unittest.main()
