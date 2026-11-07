import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from people_det import app as people_redlight
from core.hdmi_io import HDMI_DEVICE
from core.display_drm_atomic import (
    AtomicNv12Display,
    ITEM_FULL,
    ITEM_GENERIC,
    ITEM_LABEL,
    MAX_OVERLAY_ITEMS,
)
from core.rga_preprocess import letterbox_geometry
from core.native_capture import NativeNv12Frame, NativeNv12Info
from people_det.people_detector import RKNNDetector
from people_det.runtime import _inference_worker, run_video
from core.realtime_pipeline import LatestValue


class PeopleNativePipelineTests(unittest.TestCase):
    def test_native_defaults(self):
        """验证“native defaults”行为契约在重构后保持不变。"""
        args = people_redlight.build_args([])
        self.assertEqual(args.rknn_backend, "native-c")
        self.assertEqual(args.draw_mode, "fast")
        self.assertEqual(args.hdmi_device, HDMI_DEVICE)
        self.assertTrue(args.show_fps)
        self.assertEqual(
            people_redlight.get_model_path().name,
            "people_redlight_yolov5_384x640_int8.rknn",
        )

    def test_fps_can_be_disabled_explicitly(self):
        """验证“fps can be disabled explicitly”行为契约在重构后保持不变。"""
        args = people_redlight.build_args(["--no-fps"])
        self.assertFalse(args.show_fps)

    def test_video_keeps_cli_compatibility(self):
        """验证“video keeps cli compatibility”行为契约在重构后保持不变。"""
        args = people_redlight.build_args(["sample.mp4", "--show"])
        self.assertEqual(args.video, "sample.mp4")
        self.assertTrue(args.show)

    def test_video_runs_detection_on_first_frame(self):
        """验证“video runs detection on first frame”行为契约在重构后保持不变。"""
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        class Capture:
            def __init__(self):
                """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
                self.calls = 0

            def isOpened(self):
                """执行Capture的“isOpened”职责，保持原有输入输出约定。"""
                return True

            def read(self):
                """执行Capture的“read”职责，保持原有输入输出约定。"""
                self.calls += 1
                return (True, frame.copy()) if self.calls == 1 else (False, None)

            def release(self):
                """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
                pass

        class Detector:
            def __init__(self):
                """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
                self.infer_calls = 0

            def infer(self, _frame):
                """执行Detector的“infer”职责，保持原有输入输出约定。"""
                self.infer_calls += 1
                return []

            def draw(self, image, _detections, draw_mode="fast"):
                """执行Detector的“draw”职责，保持原有输入输出约定。"""
                return image, {"person": 0, "run_red_light": 0}

        detector = Detector()
        args = SimpleNamespace(
            skip=5, draw_mode="fast", result_log_interval=60
        )
        with mock.patch(
            "people_det.runtime.cv2.VideoCapture",
            return_value=Capture(),
        ):
            run_video(detector, None, "sample.mp4", False, args)
        self.assertEqual(detector.infer_calls, 1)

    def test_overlay_colors_and_filtering(self):
        """验证“overlay colors and filtering”行为契约在重构后保持不变。"""
        detector = RKNNDetector.__new__(RKNNDetector)
        detector.class_names = ["person", "run_red_light", "car"]
        detector.person_ids = {0}
        detector.run_red_ids = {1}
        values = detector.overlay_detections([
            {"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0},
            {"bbox": [1, 1, 9, 9], "score": 0.8, "class_id": 1},
            {"bbox": [2, 2, 8, 8], "score": 0.7, "class_id": 2},
        ])
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]["rgba"], (0, 255, 0, 255))
        self.assertEqual(values[1]["rgba"], (255, 0, 0, 255))

    def test_postprocess_mapping_uses_asymmetric_padding(self):
        """验证“postprocess mapping uses asymmetric padding”行为契约在重构后保持不变。"""
        detector = RKNNDetector.__new__(RKNNDetector)
        detector.class_names = ["person"]
        detector.conf_thres = 0.35
        detector.iou_thres = 0.45
        prediction = np.asarray(
            [[192.0, 320.0, 100.0, 100.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        result = detector.postprocess(
            prediction, scale=0.2, pad_left=0, pad_top=212,
            orig_w=1920, orig_h=1080,
        )
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["bbox"][1], 0)

    def test_1080p_to_model_letterbox_geometry(self):
        """验证“1080p to model letterbox geometry”行为契约在重构后保持不变。"""
        ratio, padding = letterbox_geometry(1920, 1080, 384, 640)
        self.assertAlmostEqual(ratio, 0.2)
        self.assertEqual(padding, (0, 212))

    def test_generic_overlay_is_bounded_and_clamps_rgba(self):
        """验证“generic overlay is bounded and clamps rgba”行为契约在重构后保持不变。"""
        display = AtomicNv12Display.__new__(AtomicNv12Display)
        captured = {}

        def capture(array, count, ttl_ms):
            """执行“capture”流程，保持原有输入输出和副作用不变。"""
            captured["items"] = array
            captured["count"] = count
            captured["ttl"] = ttl_ms

        display._submit_items = capture
        display.submit_detections(
            [
                {
                    "bbox": [1, 2, 3, 4],
                    "class_id": 0,
                    "label": "person",
                    "score": 0.75,
                    "rgba": (-1, 300, 10, 999),
                }
                for _ in range(MAX_OVERLAY_ITEMS + 3)
            ],
            draw_mode="full",
            ttl_ms=123,
        )
        self.assertEqual(captured["count"], MAX_OVERLAY_ITEMS)
        first = captured["items"][0]
        self.assertEqual(
            first.flags, ITEM_GENERIC | ITEM_LABEL | ITEM_FULL
        )
        self.assertEqual(
            (first.red, first.green, first.blue, first.alpha),
            (0, 255, 10, 255),
        )
        self.assertEqual(captured["ttl"], 123)

    def test_timed_overlay_forwards_source_timestamp(self):
        """验证“timed overlay forwards source timestamp”行为契约在重构后保持不变。"""
        display = AtomicNv12Display.__new__(AtomicNv12Display)
        captured = {}

        def capture(array, count, ttl_ms, source_timestamp_ns=0,
                    ready_timestamp_ns=0):
            """执行“capture”流程，保持原有输入输出和副作用不变。"""
            captured["count"] = count
            captured["ttl"] = ttl_ms
            captured["source_timestamp_ns"] = source_timestamp_ns
            captured["ready_timestamp_ns"] = ready_timestamp_ns

        display._submit_items = capture
        display.submit_detections(
            [],
            ttl_ms=222,
            source_timestamp_ns=1_234_567_890,
            ready_timestamp_ns=1_345_678_901,
        )
        self.assertEqual(captured["count"], 0)
        self.assertEqual(captured["ttl"], 222)
        self.assertEqual(
            captured["source_timestamp_ns"], 1_234_567_890
        )
        self.assertEqual(
            captured["ready_timestamp_ns"], 1_345_678_901
        )

    def test_native_frame_exposes_capture_and_decode_timestamps(self):
        """验证“native frame exposes capture and decode timestamps”行为契约在重构后保持不变。"""
        field_names = [name for name, _ctype in NativeNv12Info._fields_]
        self.assertEqual(
            field_names[-2:], ["timestamp_ns", "decoded_timestamp_ns"]
        )

    def test_native_frame_retains_processing_timestamp(self):
        """验证Native帧retain后仍携带同一DQBUF处理起点。"""
        class Capture:
            def _retain_native_frame(self, _token):
                """模拟增加Native帧引用。"""

        info = NativeNv12Info()
        info.token = 7
        info.timestamp_ns = 1_000_000_000
        info.decoded_timestamp_ns = 1_022_000_000
        frame = NativeNv12Frame(Capture(), info, 1_016_700_000)
        retained = frame.retain()
        self.assertEqual(frame.processing_timestamp_ns, 1_016_700_000)
        self.assertEqual(retained.processing_timestamp_ns, 1_016_700_000)

    def test_inference_worker_rebuilds_once(self):
        """Native输入准备失败时重建一次并释放原帧。"""
        class Frame:
            def __init__(self):
                self.release_calls = 0

            def release(self):
                self.release_calls += 1

        class Detector:
            def __init__(self):
                self.calls = 0
                self.rebuilds = 0

            def prepare_nv12(self, _frame):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary RGA error")
                return (0.2, 0, 212, 1920, 1080)

            def infer_prepared(self, *_geometry):
                return [{
                    "bbox": [0, 0, 1, 1],
                    "score": 1.0,
                    "class_id": 0,
                }]

            def rebuild(self):
                self.rebuilds += 1

        detector = Detector()
        frame = Frame()
        work = queue.Queue(maxsize=1)
        results = LatestValue()
        stop = threading.Event()
        errors = queue.Queue(maxsize=1)
        stats = {}
        work.put({
            "frame": frame,
            "source_ts": time.monotonic(),
        })
        thread = threading.Thread(
            target=_inference_worker,
            args=(detector, work, results, stop, errors, stats),
        )
        thread.start()
        deadline = time.monotonic() + 1.0
        while results.get() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(timeout=1.0)
        self.assertEqual(detector.rebuilds, 1)
        self.assertEqual(frame.release_calls, 1)
        result = results.get()
        self.assertIsNotNone(result)
        self.assertGreaterEqual(
            result["detection_latency_ms"], result["infer_ms"]
        )
        self.assertGreater(result["source_timestamp_ns"], 0)
        self.assertTrue(errors.empty())

    def test_native_worker_releases_frame_before_npu_run(self):
        """验证“native worker releases frame before npu run”行为契约在重构后保持不变。"""
        class Frame:
            def __init__(self):
                """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
                self.released = False
                self.release_calls = 0

            def release(self):
                """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
                self.released = True
                self.release_calls += 1

        frame = Frame()
        completed = threading.Event()

        class Detector:
            def prepare_nv12(self, source):
                """执行Detector的“prepare nv12”职责，保持原有输入输出约定。"""
                self.source = source
                return (0.2, 0, 212, 1920, 1080)

            def infer_prepared(self, *_geometry):
                """执行Detector的“infer prepared”职责，保持原有输入输出约定。"""
                self.released_before_run = self.source.released
                completed.set()
                return []

            def rebuild(self):
                """释放失败的RKNN上下文并按原配置重建一次。"""
                raise AssertionError("rebuild must not be called")

        detector = Detector()
        work = queue.Queue(maxsize=1)
        results = LatestValue()
        stop = threading.Event()
        errors = queue.Queue(maxsize=1)
        stats = {}
        work.put({"frame": frame, "source_ts": time.monotonic()})
        thread = threading.Thread(
            target=_inference_worker,
            args=(detector, work, results, stop, errors, stats),
        )
        thread.start()
        self.assertTrue(completed.wait(timeout=1.0))
        stop.set()
        thread.join(timeout=1.0)
        self.assertTrue(detector.released_before_run)
        self.assertEqual(frame.release_calls, 1)
        self.assertTrue(errors.empty())


if __name__ == "__main__":
    unittest.main()
