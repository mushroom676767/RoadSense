import unittest
from contextlib import redirect_stderr
from io import StringIO

from people_det import app as people_redlight
from plate_det import ai_runtime as fpga_ai
from plate_det import serial_runtime as fpga_serial


class EntrypointDefaultTests(unittest.TestCase):
    def test_all_realtime_entrypoints_are_fixed_to_native(self):
        """所有实时入口固定使用Native流水线，不再暴露视频或显示后端。"""
        ai_args = fpga_ai.build_args([])
        self.assertEqual(ai_args.draw_mode, "fast")
        self.assertFalse(hasattr(ai_args, "rknn_backend"))
        serial_args = fpga_serial.build_args([])
        self.assertFalse(hasattr(serial_args, "rknn_backend"))
        self.assertEqual(serial_args.draw_mode, "fast")
        people_args = people_redlight.build_args([])
        self.assertEqual(people_args.rknn_backend, "native-c")
        self.assertEqual(people_args.draw_mode, "fast")
        removed_attrs = (
            "_".join(("pipeline", "backend")),
            "_".join(("display", "backend")),
        )
        for args in (
            ai_args,
            fpga_serial.build_args([]),
            people_redlight.build_args([]),
        ):
            for attribute in removed_attrs:
                self.assertFalse(hasattr(args, attribute))

    def test_all_realtime_entrypoints_reject_removed_backends(self):
        """已删除的视频和显示后端参数在所有入口都必须报未知参数。"""
        pipeline_option = "--" + "-".join(("pipeline", "backend"))
        display_option = "--" + "-".join(("display", "backend"))
        removed_pipeline = "-".join(("legacy", "bgr"))
        removed_display = "".join(("fb", "dev"))
        for name, build_args in (
            ("ai", fpga_ai.build_args),
            ("serial", fpga_serial.build_args),
            ("people", people_redlight.build_args),
        ):
            for option, value in (
                (pipeline_option, removed_pipeline),
                (display_option, removed_display),
            ):
                with self.subTest(entrypoint=name, option=option):
                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            build_args([option, value])

    def test_ai_removes_all_legacy_detector_switches(self):
        """AI模式必须固定Pose26，不再暴露旧模型、尺寸或NMS参数。"""
        defaults = fpga_ai.build_args([])
        for name in ("det_model", "det_img_size", "det_iou", "rknn_backend"):
            self.assertFalse(hasattr(defaults, name))
        for option, value in (
            ("--det-model", "/tmp/old.rknn"),
            ("--det-img-size", "640"),
            ("--det-iou", "0.5"),
            ("--rknn-backend", "python-toolkit"),
        ):
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    fpga_ai.build_args([option, value])

    def test_all_realtime_entrypoints_reject_removed_rate_override(self):
        """所有实时模式均不得再暴露或接受已删除的显示帧率覆盖参数。"""
        removed_name = "_".join(("fake", "fps"))
        for name, build_args in (
            ("ai", fpga_ai.build_args),
            ("serial", fpga_serial.build_args),
            ("people", people_redlight.build_args),
        ):
            with self.subTest(entrypoint=name):
                self.assertFalse(hasattr(build_args([]), removed_name))
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        build_args([f"--{removed_name}", "21"])

    def test_serial_mode_rejects_legacy_detector_and_handover_switches(self):
        """串口模式固定使用ROI Pose模型和Native后端。"""
        defaults = fpga_serial.build_args([])
        self.assertAlmostEqual(defaults.det_threshold, 0.25)
        for name in (
            "det_model", "det_img_size", "layout_model", "fallback_policy",
            "rk_probe_fps", "fallback_max_plates", "fallback_miss_ms",
            "rknn_backend",
        ):
            self.assertFalse(hasattr(defaults, name))
        for option, value in (
            ("--det-model", "/tmp/old.rknn"),
            ("--det-img-size", "640"),
            ("--layout-model", "/tmp/layout.rknn"),
            ("--fallback-policy", "auto"),
            ("--rk-probe-fps", "2"),
            ("--fallback-max-plates", "4"),
            ("--fallback-miss-ms", "300"),
            ("--rknn-backend", "python-toolkit"),
        ):
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    fpga_serial.build_args([option, value])


if __name__ == "__main__":
    unittest.main()
