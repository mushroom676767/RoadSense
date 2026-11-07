"""Production Native Pose detection ABI tests."""

import unittest

from core.rknn_native import NativeDetectionTiming, NativePlateDetection, NativeRknnModel


class NativePlatePostprocessTests(unittest.TestCase):
    def test_ctypes_detection_abi_is_stable(self):
        self.assertEqual(
            [name for name, _ctype in NativePlateDetection._fields_],
            ["bbox", "landmarks", "landmark_visibility", "confidence", "class_id"],
        )
        self.assertEqual(
            [name for name, _ctype in NativeDetectionTiming._fields_],
            [
                "preprocess_ms", "npu_ms", "output_ms", "decode_ms",
                "nms_ms", "total_ms", "anchors_scanned", "candidates_decoded",
            ],
        )

    def test_legacy_methods_are_removed(self):
        for name in (
            "inference_nv12", "detect_nv12", "detect_pose26_nv12",
            "copy_last_input", "copy_pose26_outputs",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(NativeRknnModel, name))

    def test_production_methods_remain(self):
        for name in (
            "inference", "prepare_nv12", "inference_prepared",
            "prepare_pose26_nv12", "detect_pose26_prepared",
            "prepare_roi_pose26_nv12", "detect_roi_pose26_prepared",
            "recognize_plate_bgr", "last_run_duration_ms",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(NativeRknnModel, name))


if __name__ == "__main__":
    unittest.main()
