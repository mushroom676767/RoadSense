import unittest

from plate_det.result_stabilizer import PlateTextStabilizer


def plate(text, bbox, *, roi_index=None, confidence=0.9):
    result = {
        "plate_no": text,
        "bbox": list(bbox),
        "landmarks": [],
        "rec_conf": confidence,
        "detect_conf": 0.8,
    }
    if roi_index is not None:
        result["roi_index"] = roi_index
    return result


class PlateTextStabilizerTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = PlateTextStabilizer(
            iou_threshold=0.5,
            first_confirmations=2,
            switch_confirmations=3,
            track_ttl_ms=1200,
        )

    def test_first_text_requires_two_matching_results(self):
        self.assertEqual(
            self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.0),
            [],
        )
        current = self.stabilizer.update(
            [plate("粤B12345", [2, 1, 102, 41])], 0.1
        )
        self.assertEqual([item["plate_no"] for item in current], ["粤B12345"])

    def test_one_bad_read_does_not_replace_stable_text(self):
        self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.0)
        self.stabilizer.update([plate("粤B12345", [1, 0, 101, 40])], 0.1)
        current = self.stabilizer.update(
            [plate("粤B1234S", [4, 2, 104, 42])], 0.2
        )
        self.assertEqual(current[0]["plate_no"], "粤B12345")
        self.assertEqual(current[0]["bbox"], [4, 2, 104, 42])

    def test_stable_text_switches_after_three_matching_reads(self):
        self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.0)
        self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.1)
        for index in range(2):
            current = self.stabilizer.update(
                [plate("粤B12346", [0, 0, 100, 40])], 0.2 + index * 0.1
            )
            self.assertEqual(current[0]["plate_no"], "粤B12345")
        current = self.stabilizer.update(
            [plate("粤B12346", [0, 0, 100, 40])], 0.4
        )
        self.assertEqual(current[0]["plate_no"], "粤B12346")

    def test_empty_batch_keeps_text_until_track_ttl(self):
        self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.0)
        self.stabilizer.update([plate("粤B12345", [0, 0, 100, 40])], 0.1)
        self.assertEqual(len(self.stabilizer.update([], 1.29)), 1)
        self.assertEqual(self.stabilizer.current(1.31), [])

    def test_multiple_targets_are_matched_one_to_one_while_moving(self):
        first = [
            plate("粤B11111", [0, 0, 100, 40]),
            plate("粤B22222", [200, 0, 300, 40]),
        ]
        moved = [
            plate("粤B22222", [205, 2, 305, 42]),
            plate("粤B11111", [5, 2, 105, 42]),
        ]
        self.stabilizer.update(first, 0.0)
        current = self.stabilizer.update(moved, 0.1)
        self.assertEqual(
            {item["plate_no"] for item in current},
            {"粤B11111", "粤B22222"},
        )

    def test_fast_motion_with_no_iou_keeps_one_updated_track(self):
        self.stabilizer.update([plate("豫AH1308", [0, 0, 100, 40])], 0.0)
        current = self.stabilizer.update(
            [plate("豫AH1308", [120, 5, 220, 45])], 0.1
        )
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["plate_no"], "豫AH1308")
        self.assertEqual(current[0]["bbox"], [120, 5, 220, 45])
        current = self.stabilizer.update(
            [plate("豫AH1308", [240, 10, 340, 50])], 0.2
        )
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["bbox"], [240, 10, 340, 50])

    def test_two_targets_with_same_text_remain_distinct(self):
        first = [
            plate("粤B11111", [0, 0, 100, 40]),
            plate("粤B11111", [500, 0, 600, 40]),
        ]
        moved = [
            plate("粤B11111", [120, 0, 220, 40]),
            plate("粤B11111", [620, 0, 720, 40]),
        ]
        self.stabilizer.update(first, 0.0)
        current = self.stabilizer.update(moved, 0.1)
        self.assertEqual(len(current), 2)
        self.assertEqual(
            {tuple(item["bbox"]) for item in current},
            {(120, 0, 220, 40), (620, 0, 720, 40)},
        )

    def test_missing_target_retires_after_two_nonempty_updates(self):
        self.stabilizer.update(
            [
                plate("粤B11111", [0, 0, 100, 40]),
                plate("粤B22222", [300, 0, 400, 40]),
            ],
            0.0,
        )
        self.stabilizer.update(
            [
                plate("粤B11111", [1, 0, 101, 40]),
                plate("粤B22222", [301, 0, 401, 40]),
            ],
            0.1,
        )
        current = self.stabilizer.update(
            [plate("粤B11111", [2, 0, 102, 40])], 0.2
        )
        self.assertEqual(len(current), 2)
        current = self.stabilizer.update(
            [plate("粤B11111", [3, 0, 103, 40])], 0.3
        )
        self.assertEqual([item["plate_no"] for item in current], ["粤B11111"])

    def test_serial_group_prevents_cross_slot_matching_and_clear_resets(self):
        serial = PlateTextStabilizer(
            iou_threshold=0.35,
            group_key="roi_index",
            track_ttl_ms=1200,
        )
        serial.update([plate("粤B11111", [0, 0, 100, 40], roi_index=0)], 0.0)
        self.assertEqual(
            serial.update(
                [plate("粤B11111", [0, 0, 100, 40], roi_index=1)], 0.1
            ),
            [],
        )
        serial.clear()
        self.assertEqual(
            serial.update(
                [plate("粤B33333", [0, 0, 100, 40], roi_index=1)], 0.2
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
