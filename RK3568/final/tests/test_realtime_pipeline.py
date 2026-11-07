import queue
import unittest

from core.realtime_pipeline import (
    LatestValue,
    OwnedLatestFrame,
    put_latest,
)


class FakeFrame:
    def __init__(self, token, owner=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.token = token
        self.owner = owner or {"refs": 1, "released": 0}
        self.released = False

    def retain(self):
        """增加当前帧的Native引用计数，供另一个异步使用者安全持有。"""
        self.owner["refs"] += 1
        return FakeFrame(self.token, self.owner)

    def release(self):
        """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
        if not self.released:
            self.released = True
            self.owner["released"] += 1


class RealtimePipelineTests(unittest.TestCase):
    def test_owned_slot_releases_replaced_and_lent_frames(self):
        """验证“owned slot releases replaced and lent frames”行为契约在重构后保持不变。"""
        slot = OwnedLatestFrame()
        first = FakeFrame(1)
        second = FakeFrame(2)
        slot.publish(first)
        lent = slot.acquire()
        slot.publish(second)
        self.assertEqual(first.owner["released"], 1)
        lent.release()
        self.assertEqual(first.owner["released"], 2)
        slot.clear()
        self.assertEqual(second.owner["released"], 1)

    def test_put_latest_releases_displaced_frame(self):
        """验证“put latest releases displaced frame”行为契约在重构后保持不变。"""
        target = queue.Queue(maxsize=1)
        stale = FakeFrame(1)
        current = FakeFrame(2)
        target.put(stale)
        self.assertEqual(put_latest(target, current, release_dropped=True), 1)
        self.assertTrue(stale.released)
        self.assertIs(target.get_nowait(), current)

    def test_latest_value_ttl(self):
        """验证“latest value ttl”行为契约在重构后保持不变。"""
        value = LatestValue({"_ts": 5.0, "result": 1})
        self.assertIsNotNone(value.get_fresh(100, now=5.05))
        self.assertIsNone(value.get_fresh(100, now=5.2))


if __name__ == "__main__":
    unittest.main()
