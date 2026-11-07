import ctypes
import inspect
import queue
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from core.native_capture import NativeMppCapture
from core.realtime_pipeline import native_display_worker
from plate_det import ai_runtime


class _FakeJpegLibrary:
    def __init__(self, payload):
        self.payload = payload

    def rkhdmi_capture_read_jpeg(
        self,
        _handle,
        _last_sequence,
        destination,
        _capacity,
        out_bytes,
        out_sequence,
        out_timestamp,
        _error,
        _error_size,
    ):
        ctypes.memmove(destination, self.payload, len(self.payload))
        ctypes.cast(out_bytes, ctypes.POINTER(ctypes.c_size_t))[0] = len(
            self.payload
        )
        ctypes.cast(out_sequence, ctypes.POINTER(ctypes.c_uint32))[0] = 7
        ctypes.cast(out_timestamp, ctypes.POINTER(ctypes.c_uint64))[0] = 99
        return 1


class _Frame:
    token = 17

    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


class _Model:
    last_ocr_timing = {"npu_ms": 1.25}

    def release(self):
        pass


class _PrepareModel:
    def __init__(self):
        self.calls = []

    def prepare_pose26_nv12(self, frame, slot):
        self.calls.append((frame, slot))
        return {
            "slot": slot,
            "source_width": 1920,
            "source_height": 1080,
            "ratio": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "preprocess_ms": 4.0,
        }


class _BlockingPrepareModel(_PrepareModel):
    def __init__(self):
        super().__init__()
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def prepare_pose26_nv12(self, frame, slot):
        prepared = super().prepare_pose26_nv12(frame, slot)
        if len(self.calls) == 2:
            self.second_started.set()
            if not self.release_second.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to finish stale RGA")
        return prepared


class _Counters:
    def inc(self, *_args):
        pass

    def add_time(self, *_args):
        pass


class _RepeatDisplay:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.calls = 0

    def show(self, frame):
        self.calls += 1
        if self.calls == 3:
            self.stop_event.set()
        return 0


class StreamPerformanceTests(unittest.TestCase):
    def test_pose26_double_buffer_waits_for_timed_permission(self):
        source = queue.Queue()
        stop = threading.Event()
        model = _PrepareModel()
        first_frame = _Frame()
        second_frame = _Frame()
        source.put({"frame": first_frame, "frame_id": 1})
        source.put({"frame": second_frame, "frame_id": 2})
        worker = ai_runtime.Pose26DoubleBufferPreprocessor(
            model, source, stop
        )
        worker.start()
        first = worker.get(timeout=1.0)
        self.assertEqual(first["slot"], 0)
        time.sleep(0.05)
        self.assertEqual(len(model.calls), 1)

        worker.allow_next(0.0)
        second = worker.get(timeout=1.0)
        self.assertEqual(second["slot"], 1)
        self.assertEqual([slot for _frame, slot in model.calls], [0, 1])
        worker.release_slot(first)
        worker.release_slot(second)
        source.put(None)
        worker.allow_next(0.0)
        self.assertIsNone(worker.get(timeout=1.0))
        worker.close()
        self.assertEqual(first_frame.releases, 0)
        self.assertEqual(second_frame.releases, 0)

    def test_pose26_double_buffer_close_releases_ready_frame(self):
        source = queue.Queue()
        frame = _Frame()
        source.put({"frame": frame, "frame_id": 1})
        worker = ai_runtime.Pose26DoubleBufferPreprocessor(
            _PrepareModel(), source, threading.Event()
        )
        worker.start()
        deadline = time.monotonic() + 1.0
        while worker.prepared_q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.close()
        self.assertEqual(frame.releases, 1)

    def test_pose26_preload_delay_uses_previous_native_timing(self):
        self.assertAlmostEqual(
            ai_runtime.pose26_preload_delay_ms({
                "npu_ms": 18.8,
                "preprocess_ms": 4.1,
            }),
            15.7,
        )

    def test_pose26_double_buffer_discards_prepared_stale_frame(self):
        source = queue.Queue()
        stop = threading.Event()
        model = _PrepareModel()
        first_frame = _Frame()
        stale_frame = _Frame()
        latest_frame = _Frame()
        source.put({"frame": first_frame, "frame_id": 1})
        source.put({"frame": stale_frame, "frame_id": 2})
        source.put({"frame": latest_frame, "frame_id": 3})
        worker = ai_runtime.Pose26DoubleBufferPreprocessor(
            model, source, stop
        )
        worker.start()
        first = worker.get(timeout=1.0)
        worker.allow_next(0.0)
        deadline = time.monotonic() + 1.0
        while worker.prepared_q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        _generation, discarded = worker.discard_pending_and_allow(0.0)
        latest = worker.get(timeout=1.0)
        self.assertEqual(discarded, 1)
        self.assertEqual(latest["item"]["frame_id"], 3)
        self.assertEqual(stale_frame.releases, 1)
        self.assertEqual(first_frame.releases, 0)
        self.assertEqual(latest_frame.releases, 0)

        worker.release_slot(first)
        worker.release_slot(latest)
        worker.close()
        self.assertEqual(worker.free_slots.qsize(), 2)
        self.assertEqual(stale_frame.releases, 1)

    def test_pose26_double_buffer_discards_inflight_stale_frame(self):
        source = queue.Queue()
        model = _BlockingPrepareModel()
        first_frame = _Frame()
        stale_frame = _Frame()
        latest_frame = _Frame()
        source.put({"frame": first_frame, "frame_id": 1})
        worker = ai_runtime.Pose26DoubleBufferPreprocessor(
            model, source, threading.Event()
        )
        worker.start()
        first = worker.get(timeout=1.0)

        source.put({"frame": stale_frame, "frame_id": 2})
        worker.allow_next(0.0)
        self.assertTrue(model.second_started.wait(timeout=1.0))
        source.put({"frame": latest_frame, "frame_id": 3})
        _generation, discarded = worker.discard_pending_and_allow(0.0)
        self.assertEqual(discarded, 0)
        model.release_second.set()

        latest = worker.get(timeout=1.0)
        self.assertEqual(latest["item"]["frame_id"], 3)
        self.assertEqual(stale_frame.releases, 1)
        worker.release_slot(first)
        worker.release_slot(latest)
        worker.close()
        self.assertEqual(worker.free_slots.qsize(), 2)
        self.assertEqual(stale_frame.releases, 1)

    def test_pose26_recognition_delay_scales_with_candidate_count(self):
        timing = {"preprocess_ms": 4.1}
        self.assertAlmostEqual(
            ai_runtime.pose26_recognition_preload_delay_ms(1, timing),
            3.4,
        )
        self.assertAlmostEqual(
            ai_runtime.pose26_recognition_preload_delay_ms(2, timing),
            9.9,
        )

    def test_notifying_queue_wakes_event_on_arrival(self):
        wake = threading.Event()
        items = ai_runtime.NotifyingQueue(maxsize=1, wake_event=wake)
        items.put_nowait("frame")
        self.assertTrue(wake.is_set())
        self.assertEqual(items.get_nowait(), "frame")

    def test_native_loop_has_no_two_millisecond_queue_poll(self):
        source = inspect.getsource(ai_runtime.run_native_zero_copy)
        self.assertNotIn("timeout=0.002", source)
        self.assertIn("wake_event.wait", source)

    def test_native_wait_timeout_tracks_result_ttl(self):
        state = {
            "latest_detection_result": {"timestamp": 10.0},
            "latest_recognition_result": None,
        }
        self.assertAlmostEqual(
            ai_runtime._native_wait_timeout(
                state, ttl_ms=500, now=10.45, health_interval=0.2
            ),
            0.05,
        )
        self.assertEqual(
            ai_runtime._native_wait_timeout(
                state, ttl_ms=500, now=10.6, health_interval=0.2
            ),
            0.0,
        )

    def test_jpeg_read_uses_direct_native_bytes_copy(self):
        payload = b"\xff\xd8" + b"x" * 4096 + b"\xff\xd9"
        capture = NativeMppCapture.__new__(NativeMppCapture)
        capture.handle = object()
        capture.lib = _FakeJpegLibrary(payload)
        capture._jpeg_buffer = (ctypes.c_uint8 * (len(payload) + 16))()

        with mock.patch(
            "core.native_capture.ctypes.string_at",
            wraps=ctypes.string_at,
        ) as direct_copy:
            result = capture.read_jpeg(0)

        self.assertEqual(result, (payload, 7, 99))
        direct_copy.assert_called_once()

    def test_shared_display_worker_repeats_without_new_input(self):
        stop = threading.Event()
        frame = _Frame()
        frames = queue.Queue()
        frames.put(frame)
        display = _RepeatDisplay(stop)

        native_display_worker(display, frames, stop)

        self.assertEqual(display.calls, 3)
        self.assertEqual(frame.releases, 1)

    def test_ai_recognition_releases_full_frame_before_npu(self):
        frame = _Frame()
        now_ns = 1_000_000_000
        task = {
            "frame_id": 1,
            "timestamp": 1.0,
            "frame": frame,
            "det_results": [
                {
                    "bbox": [1, 2, 30, 12],
                    "landmarks": [[1, 2], [30, 2], [30, 12], [1, 12]],
                    "class": 0,
                    "conf": 0.9,
                }
            ],
            "detect_ms": 4.0,
            "source_timestamp_ns": now_ns - 20_000_000,
            "e2e_source_timestamp_ns": now_ns - 18_000_000,
            "detect_finished_ns": now_ns,
            "detection_latency_ms": 6.0,
            "decoded_timestamp_ns": now_ns - 10_000_000,
            "decode_latency_ms": 8.0,
            "detect_queue_ms": 2.0,
        }
        rec_queue = queue.Queue()
        rec_queue.put(task)
        rec_queue.put(None)
        result_queue = queue.Queue()
        status_queue = queue.Queue()
        args = SimpleNamespace(rknn_backend="native-c")

        def recognize_prepared(
            prepared_plate,
            candidate,
            _model,
        ):
            self.assertEqual(frame.releases, 1)
            self.assertEqual(prepared_plate, b"roi")
            return {
                "plate_no": "TEST123",
                "plate_color": "blue",
                "rec_conf": 0.9,
                "color_conf": 0.9,
                "bbox": candidate["bbox"],
                "landmarks": candidate["landmarks"],
                "detect_conf": candidate["conf"],
            }

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                ai_runtime, "load_rknn_model", return_value=_Model()
            ))
            stack.enter_context(mock.patch.object(
                ai_runtime, "prepare_plate_image", return_value=b"roi"
            ))
            stack.enter_context(mock.patch.object(
                ai_runtime,
                "process_prepared_plate",
                side_effect=recognize_prepared,
            ))
            ai_runtime.rec_worker(
                1,
                args,
                rec_queue,
                result_queue,
                status_queue,
                threading.Event(),
                _Counters(),
                threading.Lock(),
            )

        self.assertEqual(frame.releases, 1)
        result = result_queue.get_nowait()
        self.assertEqual(result["plate_results"][0]["plate_no"], "TEST123")
        self.assertEqual(result["recognition_npu_ms"], 1.25)


if __name__ == "__main__":
    unittest.main()
