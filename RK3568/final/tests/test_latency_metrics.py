import unittest
from types import SimpleNamespace

from core.latency_metrics import (
    DisplayedLatencyTracker,
    DisplayedStageTracker,
    ai_latency_values,
    frame_e2e_source_timestamp_ns,
    people_latency_values,
    serial_latency_values,
)


class LatencyFormulaTests(unittest.TestCase):
    def test_e2e_prefers_dequeue_processing_timestamp(self):
        """验证端到端从DQBUF完成时刻开始，排除摄像头采集周期。"""
        frame = SimpleNamespace(
            timestamp_ns=1_000_000_000,
            processing_timestamp_ns=1_016_700_000,
        )
        self.assertEqual(
            frame_e2e_source_timestamp_ns(frame), 1_016_700_000
        )

    def test_e2e_falls_back_for_legacy_native_frame(self):
        """验证旧Native帧缺少处理时间戳时仍保持原有兼容行为。"""
        frame = SimpleNamespace(timestamp_ns=1_000_000_000)
        self.assertEqual(
            frame_e2e_source_timestamp_ns(frame), 1_000_000_000
        )

    def test_ai_uses_only_native_npu_timings(self):
        """AI检测和识别时延只采用模型的纯NPU耗时。"""
        detection, recognition = ai_latency_values({
            "native_detection_timing": {"npu_ms": 20.3},
            "recognition_npu_ms": 1.4,
            "detect_queue_ms": 200.0,
            "detect_ms": 300.0,
            "recognition_queue_ms": 400.0,
            "plate_ms": 500.0,
        })
        self.assertEqual(detection, 20.3)
        self.assertEqual(recognition, 1.4)

    def test_serial_uses_only_detector_and_ocr_npu_timings(self):
        """串口检测和识别时延不混入组图、排队或后处理。"""
        detection, recognition = serial_latency_values({
            "detection_npu_ms": 7.2,
            "recognition_npu_ms": 1.3,
            "detection_latency_ms": 10.0,
            "recognition_queue_ms": 100.0,
            "recognition_compute_ms": 200.0,
        })
        self.assertEqual(detection, 7.2)
        self.assertEqual(recognition, 1.3)

    def test_missing_pure_npu_timings_do_not_fall_back(self):
        """旧阶段字段存在时也不得混入新的纯NPU口径。"""
        self.assertEqual(
            ai_latency_values({"detect_ms": 20.0, "plate_ms": 2.0}),
            (None, None),
        )
        self.assertEqual(
            serial_latency_values({"detection_latency_ms": 10.0}),
            (None, None),
        )

    def test_people_uses_queue_prepare_and_compute(self):
        """验证“people uses queue prepare and compute”行为契约在重构后保持不变。"""
        detection = people_latency_values({
            "source_timestamp_ns": 100,
            "inference_queue_ms": 3.0,
            "prepare_ms": 4.0,
            "inference_compute_ms": 22.0,
        })
        self.assertEqual(detection, 29.0)


class DisplayedLatencyTrackerTests(unittest.TestCase):
    def setUp(self):
        """为当前测试用例准备隔离的初始状态。"""
        self.tracker = DisplayedLatencyTracker(capacity=4)
        self.tracker.record(
            950_000_000, 1_000_000_000, 33.5, 13.5
        )

    def test_resolves_only_matching_page_flip_source(self):
        """验证“resolves only matching page flip source”行为契约在重构后保持不变。"""
        mismatched = SimpleNamespace(
            last_overlay_source_timestamp_ns=949_000_000,
            last_overlay_e2e_ms=51.0,
        )
        self.assertIsNone(
            self.tracker.resolve(
                mismatched, 500, now_ns=1_100_000_000
            )["total_latency_ms"]
        )

        matched = SimpleNamespace(
            last_overlay_source_timestamp_ns=950_000_000,
            last_overlay_e2e_ms=51.0,
        )
        resolved = self.tracker.resolve(
            matched, 500, now_ns=1_100_000_000
        )
        self.assertEqual(resolved["detection_latency_ms"], 33.5)
        self.assertEqual(resolved["recognition_latency_ms"], 13.5)
        self.assertEqual(resolved["total_latency_ms"], 50.0)

    def test_ttl_clears_stale_values(self):
        """验证“ttl clears stale values”行为契约在重构后保持不变。"""
        stats = SimpleNamespace(
            last_overlay_source_timestamp_ns=950_000_000,
            last_overlay_e2e_ms=51.0,
        )
        resolved = self.tracker.resolve(
            stats, 50, now_ns=1_100_000_000
        )
        self.assertEqual(
            resolved,
            {
                "detection_latency_ms": None,
                "recognition_latency_ms": None,
                "total_latency_ms": None,
            },
        )

    def test_new_tracker_does_not_reuse_previous_mode(self):
        """验证“new tracker does not reuse previous mode”行为契约在重构后保持不变。"""
        stats = SimpleNamespace(
            last_overlay_source_timestamp_ns=950_000_000,
            last_overlay_e2e_ms=51.0,
        )
        fresh_tracker = DisplayedLatencyTracker()
        self.assertIsNone(
            fresh_tracker.resolve(
                stats, 500, now_ns=1_100_000_000
            )["detection_latency_ms"]
        )


class DisplayedStageTrackerTests(unittest.TestCase):
    """验证调试阶段耗时与实际Page Flip来源帧严格关联。"""

    def test_returns_only_the_displayed_source_sample(self):
        tracker = DisplayedStageTracker(capacity=4)
        tracker.record(
            900_000_000, 1_000_000_000,
            {"detect_ms": 15.0, "recognition_ms": None},
        )
        tracker.record(
            910_000_000, 1_010_000_000,
            {"detect_ms": 18.0, "recognition_ms": 14.0},
        )
        displayed = SimpleNamespace(
            last_overlay_source_timestamp_ns=900_000_000
        )

        resolved = tracker.resolve(
            displayed, 500, now_ns=1_100_000_000
        )

        self.assertEqual(
            resolved["source_timestamp_ns"], 900_000_000
        )
        self.assertEqual(resolved["algorithm_e2e_ms"], 100.0)
        self.assertEqual(resolved["detect_ms"], 15.0)
        self.assertIsNone(resolved["recognition_ms"])

    def test_returns_none_for_unknown_or_expired_source(self):
        tracker = DisplayedStageTracker()
        tracker.record(
            900_000_000, 1_000_000_000, {"detect_ms": 15.0}
        )
        unknown = SimpleNamespace(
            last_overlay_source_timestamp_ns=899_000_000
        )
        expired = SimpleNamespace(
            last_overlay_source_timestamp_ns=900_000_000
        )

        self.assertIsNone(
            tracker.resolve(unknown, 500, now_ns=1_100_000_000)
        )
        self.assertIsNone(
            tracker.resolve(expired, 50, now_ns=1_100_000_000)
        )


if __name__ == "__main__":
    unittest.main()
