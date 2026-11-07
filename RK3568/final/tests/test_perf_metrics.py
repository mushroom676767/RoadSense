import unittest

from tools.perf_metrics import (
    parse_capture_sample,
    parse_display_sample,
    percentile,
    summarize,
)


CAPTURE_LINE = (
    "[采集 7200] 采集帧率:59.8 FPS | V4L2:16.7ms | mpp:5.4ms | "
    "V4L2丢帧:0 | MPP丢帧:1 | 输出丢帧:0 | 队列丢帧:2"
)
DISPLAY_LINE = (
    "[调试 7200] 显示:59.6FPS | DQBUF→解码:5.4ms | "
    "检测排队:1.2ms | 检测NPU等待:0.3ms | 检测计算:17.6ms | "
    "检测RGA:0.8ms | 检测NPU:13.2ms | 检测输出:0.5ms | "
    "检测解码:2.4ms | 检测NMS:0.7ms | 识别排队:0.0ms | "
    "识别NPU等待:0.0ms | 识别计算:0.0ms | 识别候选:0 | "
    "结果调度:0.1ms | Overlay:0.08ms | 就绪→显示:8.4ms | "
    "端到端:33.7ms | DRM错误:0"
)


class PerfMetricsTests(unittest.TestCase):
    def test_parse_current_capture_log(self):
        sample = parse_capture_sample(CAPTURE_LINE)
        self.assertEqual(sample.frame, 7200)
        self.assertEqual(sample.fps, 59.8)
        self.assertEqual(sample.v4l2_ms, 16.7)
        self.assertEqual(sample.mpp_drops, 1)
        self.assertEqual(sample.queue_drops, 2)

    def test_parse_current_display_log(self):
        sample = parse_display_sample(DISPLAY_LINE)
        self.assertEqual(sample.frame, 7200)
        self.assertEqual(sample.detect_compute_ms, 17.6)
        self.assertEqual(sample.detect_npu_ms, 13.2)
        self.assertEqual(sample.recognition_candidates, 0)
        self.assertEqual(sample.overlay_ms, 0.08)
        self.assertEqual(sample.e2e_ms, 33.7)

    def test_stale_log_formats_are_rejected(self):
        self.assertIsNone(parse_capture_sample(
            "[采集 60] 采集帧率:60.0 FPS | V4L2:1ms | mpp:2ms | 丢帧:0"
        ))
        self.assertIsNone(parse_display_sample(
            "[调试 60] 显示帧率:60 FPS | 检测:10ms | 总时延:20ms"
        ))

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([], 99), 0.0)
        self.assertEqual(percentile([1, 2, 3], 50), 2.0)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_schema_v2_summary(self):
        capture = parse_capture_sample(CAPTURE_LINE)
        display = parse_display_sample(DISPLAY_LINE)
        result = summarize([capture], [display])
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["fps"]["display"]["min"], 59.6)
        self.assertEqual(result["drops"], {
            "v4l2": 0, "mpp": 1, "output": 0, "queue": 2, "drm": 0,
        })
        self.assertEqual(result["stages_ms"]["detect_npu"]["p95"], 13.2)
        for stage in result["stages_ms"].values():
            self.assertEqual(set(stage), {"avg", "p50", "p95", "p99"})


if __name__ == "__main__":
    unittest.main()
