import ctypes
import queue
import threading
import unittest
from unittest import mock

from plate_det import serial_runtime as fpga_serial
from core.realtime_pipeline import LatestValue
from core.rknn_native import (
    NativePlateDetection,
    NativeRoiPlateDetection,
    NativeRoiRect,
    NativeRoiSlotInfo,
)
from plate_det.result_stabilizer import PlateTextStabilizer


class FakeFrame:
    def __init__(self):
        self.refs = 1

    def retain(self):
        self.refs += 1
        return self

    def release(self):
        self.refs -= 1


class FakeLatestFrame:
    def __init__(self, frame=None):
        self.frame = frame

    def acquire(self):
        return None if self.frame is None else self.frame.retain()


class SerialRoiRuntimeTests(unittest.TestCase):
    def test_serial_rois_expand_twenty_percent_on_every_side(self):
        source = [{"x1": 100, "y1": 200, "x2": 400, "y2": 300}]
        self.assertEqual(
            fpga_serial.expand_serial_rois(source),
            [{"x1": 40, "y1": 180, "x2": 460, "y2": 320}],
        )
        self.assertEqual(
            source,
            [{"x1": 100, "y1": 200, "x2": 400, "y2": 300}],
        )

    def test_serial_roi_expansion_clips_to_full_frame(self):
        self.assertEqual(
            fpga_serial.expand_serial_rois([
                {"x1": 0, "y1": 1, "x2": 1920, "y2": 1080},
            ]),
            [{"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}],
        )

    def test_submit_expands_processing_rois_and_preserves_source(self):
        pipeline = fpga_serial.RoiPosePipeline.__new__(
            fpga_serial.RoiPosePipeline
        )
        pipeline._barrier_lock = threading.Lock()
        pipeline._clear_barrier = 0
        pipeline.packet_queue = queue.Queue(maxsize=1)
        pipeline.stats = {"packets": 0}
        event = {
            "rois": [{"x1": 100, "y1": 200, "x2": 400, "y2": 300}],
        }
        pipeline.submit(event)
        queued = pipeline.packet_queue.get_nowait()
        self.assertEqual(queued["source_rois"], event["rois"])
        self.assertEqual(
            queued["rois"],
            [{"x1": 40, "y1": 180, "x2": 460, "y2": 320}],
        )

    def test_native_roi_ctypes_abi_extends_detection_with_roi_index(self):
        self.assertEqual(
            [name for name, _ctype in NativeRoiRect._fields_],
            ["x1", "y1", "x2", "y2"],
        )
        self.assertEqual(
            [name for name, _ctype in NativeRoiSlotInfo._fields_],
            [
                "roi_index", "slot_left", "slot_top", "ratio",
                "pad_x", "pad_y", "requested", "aligned",
            ],
        )
        self.assertEqual(
            ctypes.sizeof(NativeRoiPlateDetection),
            ctypes.sizeof(NativePlateDetection) + ctypes.sizeof(ctypes.c_int32),
        )

    def test_serial_snapshot_uses_roi_schema_and_binds_current_frame(self):
        frame = FakeFrame()
        packets = []
        clears = []
        source = fpga_serial.SerialRoiSource(
            "/dev/null", 115205, FakeLatestFrame(frame),
            packets.append, lambda *args: clears.append(args),
            printer=lambda _message: None,
        )
        source._record(
            [{"x1": 100, "y1": 200, "x2": 400, "y2": 300}],
            bytes.fromhex("AA 01 64 00 C8 00 90 01 2C 01"),
        )
        snapshot = source.snapshot()
        self.assertEqual(snapshot["count"], 1)
        self.assertEqual(snapshot["rois"][0]["x1"], 100)
        self.assertNotIn("points", snapshot)
        self.assertEqual(packets[0]["frame"], frame)
        packets[0]["frame"].release()
        self.assertEqual(frame.refs, 1)
        self.assertEqual(clears, [])

    def test_count_zero_calls_clear_without_acquiring_frame(self):
        packets = []
        clears = []
        latest = mock.Mock()
        source = fpga_serial.SerialRoiSource(
            "/dev/null", 115205, latest, packets.append,
            lambda *args: clears.append(args), printer=lambda _message: None,
        )
        source._record([], b"\xaa\x00")
        latest.acquire.assert_not_called()
        self.assertEqual(packets, [])
        self.assertEqual(clears, [("count-zero", 1)])
        self.assertEqual(source.snapshot()["count"], 0)

    def test_clear_barrier_publishes_empty_result_and_drops_waiting_packet(self):
        pipeline = fpga_serial.RoiPosePipeline.__new__(
            fpga_serial.RoiPosePipeline
        )
        pipeline._barrier_lock = threading.Lock()
        pipeline._clear_barrier = 0
        pipeline.packet_queue = queue.Queue(maxsize=1)
        pipeline.result_slot = LatestValue()
        pipeline.stats = {"clears": 0}
        pipeline.text_stabilizer = PlateTextStabilizer(
            iou_threshold=0.35, group_key="roi_index"
        )
        pipeline._last_stable_signature = ()
        pipeline._last_result_metadata = {}
        frame = FakeFrame()
        pipeline.packet_queue.put({"frame": frame})
        pipeline.clear("count-zero", 9)
        self.assertEqual(frame.refs, 0)
        result = pipeline.result_slot.get()
        self.assertEqual(result["overlays"], [])
        self.assertEqual(result["clear_reason"], "count-zero")
        self.assertEqual(result["serial_sequence"], 9)

    def test_removed_cli_options_are_unknown(self):
        for option, value in (
            ("--det-model", "/tmp/model.rknn"),
            ("--layout-model", "/tmp/layout.rknn"),
            ("--fallback-policy", "auto"),
            ("--rknn-backend", "python-toolkit"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                fpga_serial.build_args([option, value])


if __name__ == "__main__":
    unittest.main()
