"""车牌四角点贴合框的统一行为测试。"""

import unittest

import numpy as np

from plate_det.serial_runtime import serial_pose_confirmed_roi_overlay
from plate_det.ai_runtime import merge_draw_results


class PlatePolygonOverlayTests(unittest.TestCase):
    """确保AI矩形和串口红色多边形使用各自约定。"""

    def test_ai_results_keep_original_bbox_rectangle(self):
        """AI模式保留原bbox结果，不写入多边形绘制标志。"""
        candidate = {
            "bbox": np.asarray([10, 10, 100, 50]),
            "landmarks": np.asarray(
                [[10, 10], [100, 12], [98, 50], [12, 48]],
                dtype=np.float32,
            ),
            "conf": 0.9,
            "class": 0,
        }
        overlay = merge_draw_results([candidate], [])[0]
        self.assertNotIn("polygon", overlay)
        self.assertNotIn("polygon_landmarks", overlay)
        self.assertEqual(list(overlay["bbox"]), [10, 10, 100, 50])

    def test_serial_results_use_pose_confirmed_source_roi(self):
        """Pose命中后使用对应的原始串口ROI，不使用扩张ROI或Pose框。"""
        points = [[10, 10], [100, 12], [98, 50], [12, 48]]
        source_rois = [
            {"x1": 10, "y1": 20, "x2": 110, "y2": 60},
            {"x1": 300, "y1": 200, "x2": 500, "y2": 280},
        ]
        overlay = serial_pose_confirmed_roi_overlay(
            {
                "bbox": [320, 210, 470, 270],
                "landmarks": points,
                "roi_index": 1,
                "polygon": True,
                "polygon_landmarks": [
                    points[0], points[1], points[3], points[2]
                ],
            },
            source_rois,
        )
        self.assertNotIn("polygon", overlay)
        self.assertNotIn("polygon_landmarks", overlay)
        self.assertEqual(overlay["bbox"], [300, 200, 500, 280])
        self.assertEqual(overlay["geometry_source"], "fpga-pose")

    def test_serial_result_without_matching_pose_slot_is_not_drawn(self):
        """无对应Pose槽位时禁止直接绘制串口ROI。"""
        self.assertIsNone(serial_pose_confirmed_roi_overlay(
            {"bbox": [1, 2, 3, 4], "roi_index": 2},
            [{"x1": 10, "y1": 20, "x2": 110, "y2": 60}],
        ))

if __name__ == "__main__":
    unittest.main()
