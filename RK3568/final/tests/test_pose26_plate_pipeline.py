import unittest

import numpy as np

from plate_det.plate_pipeline import (
    pose26_landmarks_are_valid,
    prepare_plate_image,
)


class Pose26PlatePipelineTests(unittest.TestCase):
    def _detection(self, visibility):
        return {
            "bbox": np.asarray([20, 30, 100, 60], dtype=np.float32),
            "landmarks": np.asarray(
                [[25, 32], [95, 32], [95, 58], [25, 58]],
                dtype=np.float32,
            ),
            "landmark_visibility": np.asarray(visibility, dtype=np.float32),
            "class": 0,
            "conf": 0.9,
        }

    def test_visible_pose26_corners_use_perspective_geometry(self):
        detection = self._detection([0.9, 0.9, 0.9, 0.9])
        self.assertTrue(pose26_landmarks_are_valid(detection))
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        roi = prepare_plate_image(image, detection)
        self.assertGreaterEqual(roi.shape[1], 65)
        self.assertGreaterEqual(roi.shape[0], 24)

    def test_low_visibility_falls_back_to_bbox_crop(self):
        detection = self._detection([0.9, 0.2, 0.9, 0.9])
        self.assertFalse(pose26_landmarks_are_valid(detection))
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        roi = prepare_plate_image(image, detection)
        self.assertEqual(roi.shape, (30, 80, 3))


if __name__ == "__main__":
    unittest.main()
