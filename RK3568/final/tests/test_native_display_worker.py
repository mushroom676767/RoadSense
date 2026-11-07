import queue
import threading
import time
import unittest
from unittest import mock

from plate_det.ai_runtime import (
    NpuExecutionArbiter,
    native_display_worker,
    run_with_rknn_recovery,
)


class _Frame:
    token = 1

    def __init__(self):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.release_count = 0

    def release(self):
        """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
        self.release_count += 1


class _FailingDisplay:
    def __init__(self):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.closed = False

    def show(self, _frame):
        """提交一帧到显示后端，并保持原有Page Flip和缓冲复用语义。"""
        raise RuntimeError("injected atomic failure")

    def close(self):
        """按原有依赖顺序停止工作并释放本对象持有的硬件或系统资源。"""
        self.closed = True


class _RepeatingDisplay:
    def __init__(self, stop):
        self.stop = stop
        self.tokens = []
        self.closed = False

    def show(self, frame):
        self.tokens.append(frame.token)
        if len(self.tokens) >= 2:
            self.stop.set()
        return 0

    def close(self):
        self.closed = True


class _Counters:
    def inc(self, *_args):
        """执行_Counters的“inc”职责，保持原有输入输出约定。"""
        pass

    def add_time(self, *_args):
        """执行_Counters的“add time”职责，保持原有输入输出约定。"""
        pass


class _Model:
    def __init__(self, name):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.name = name
        self.released = False

    def release(self):
        """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
        self.released = True


class NativeDisplayWorkerTests(unittest.TestCase):
    def test_atomic_failure_is_reported_and_frame_released(self):
        """验证“atomic failure is reported and frame released”行为契约在重构后保持不变。"""
        display = _FailingDisplay()
        frame = _Frame()
        frames = queue.Queue()
        frames.put(frame)
        stop = threading.Event()
        status = queue.Queue()

        native_display_worker(display, frames, stop, _Counters(), status)

        self.assertTrue(stop.is_set())
        self.assertTrue(display.closed)
        self.assertEqual(frame.release_count, 1)
        self.assertEqual(
            status.get_nowait(),
            ("error", "atomic-display-worker", "injected atomic failure"),
        )

    def test_atomic_worker_repeats_latest_frame_at_vblank(self):
        stop = threading.Event()
        display = _RepeatingDisplay(stop)
        frame = _Frame()
        frames = queue.Queue()
        frames.put(frame)

        native_display_worker(
            display, frames, stop, _Counters(), queue.Queue()
        )

        self.assertEqual(display.tokens, [frame.token, frame.token])
        self.assertEqual(frame.release_count, 1)
        self.assertTrue(display.closed)

    def test_rknn_operation_rebuilds_once(self):
        """验证“rknn operation rebuilds once”行为契约在重构后保持不变。"""
        original = _Model("original")
        replacement = _Model("replacement")
        status = queue.Queue()

        def operation(model):
            """执行“operation”流程，保持原有输入输出和副作用不变。"""
            if model is original:
                raise RuntimeError("injected NPU failure")
            return "ok"

        with mock.patch(
            "plate_det.ai_runtime.load_rknn_model", return_value=replacement
        ) as loader:
            result, active = run_with_rknn_recovery(
                original, "model.rknn", "detector", operation, status
            )

        self.assertEqual(result, "ok")
        self.assertIs(active, replacement)
        self.assertTrue(original.released)
        self.assertFalse(replacement.released)
        loader.assert_called_once()
        self.assertEqual(status.get_nowait()[0], "recovered")

    def test_rknn_operation_obeys_shared_execution_lock(self):
        model = _Model("detector")
        gate = threading.Lock()
        status = queue.Queue()
        started = threading.Event()
        finished = threading.Event()

        gate.acquire()

        def invoke():
            started.set()
            run_with_rknn_recovery(
                model,
                "model.rknn",
                "detector",
                lambda active: active.name,
                status,
                execution_lock=gate,
            )
            finished.set()

        worker = threading.Thread(target=invoke)
        worker.start()
        self.assertTrue(started.wait(1.0))
        self.assertFalse(finished.wait(0.05))
        gate.release()
        self.assertTrue(finished.wait(1.0))
        worker.join(timeout=1.0)

    def test_npu_arbiter_alternates_detection_and_recognition(self):
        """双方等待时识别紧跟检测运行，检测不能连续抢占单核NPU。"""
        arbiter = NpuExecutionArbiter()
        detect_lock = arbiter.role_lock("detect")
        recognition_lock = arbiter.role_lock("recognition")
        order = []

        # 先让一次检测持有NPU，再同时排入下一次检测和识别。释放后应由
        # 识别先运行，随后才轮到下一次检测。
        detect_lock.acquire()

        def contend(role, role_lock):
            """记录调度器实际授予两个角色的先后顺序。"""
            role_lock.acquire()
            try:
                order.append(role)
            finally:
                role_lock.release()

        detect_thread = threading.Thread(
            target=contend, args=("detect", detect_lock)
        )
        recognition_thread = threading.Thread(
            target=contend, args=("recognition", recognition_lock)
        )
        detect_thread.start()
        recognition_thread.start()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with arbiter._condition:
                if (
                    arbiter._waiting["detect"] == 1
                    and arbiter._waiting["recognition"] == 1
                ):
                    break
            time.sleep(0.001)
        else:
            self.fail("NPU contenders did not both enter the wait queue")

        detect_lock.release()
        detect_thread.join(timeout=1.0)
        recognition_thread.join(timeout=1.0)

        self.assertFalse(detect_thread.is_alive())
        self.assertFalse(recognition_thread.is_alive())
        self.assertEqual(order, ["recognition", "detect"])

    def test_npu_arbiter_reserves_recognition_before_worker_wakes(self):
        """识别任务入队后即使Worker尚未运行，下一帧检测也不能抢占。"""
        arbiter = NpuExecutionArbiter()
        detect_lock = arbiter.role_lock("detect")
        recognition_lock = arbiter.role_lock("recognition")
        order = []
        detection_finished = threading.Event()

        detect_lock.acquire()
        detect_lock.release()
        detect_lock.reserve_next("recognition")

        def run_detection():
            """尝试执行预约之后到达的下一帧检测。"""
            detect_lock.acquire()
            try:
                order.append("detect")
            finally:
                detect_lock.release()
                detection_finished.set()

        detection_thread = threading.Thread(target=run_detection)
        detection_thread.start()
        self.assertFalse(detection_finished.wait(0.05))

        recognition_lock.acquire()
        try:
            order.append("recognition")
        finally:
            recognition_lock.release()

        self.assertTrue(detection_finished.wait(1.0))
        detection_thread.join(timeout=1.0)
        self.assertEqual(order, ["recognition", "detect"])


if __name__ == "__main__":
    unittest.main()
