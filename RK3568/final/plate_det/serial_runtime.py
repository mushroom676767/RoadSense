"""FPGA serial ROI + native YOLO26 Pose plate recognition runtime."""

import argparse
import hashlib
import math
import os
import queue
import threading
import time

import cv2

from core.camera_fps import configure_camera_fps
from core.display_drm_atomic import AtomicNv12Display
from core.hdmi_io import HDMI_DEVICE
from core.native_capture import NativeMppHdmiReceiver
from core.publisher import FramePublisher, add_publish_args, plate_overlays, validate_publish_args
from core.realtime_pipeline import (
    LatestValue,
    OwnedLatestFrame,
    native_display_worker,
    put_latest,
    release_frame,
)
from core.rknn_native import NativeRknnModel
from core.serial_io import SerialRoiProtocolError, open_serial, read_plate_rois
from plate_det.plate_pipeline import prepare_plate_image
from plate_det.plate_recognizer import recognize_rknn
from plate_det.result_stabilizer import PlateTextStabilizer


DEBUG_INTERVAL = 60
SERIAL_OVERLAY_RGBA = (255, 0, 0, 255)
DEFAULT_REC_MODEL = "/home/cat/RK3568/final/rknn_models/plate_ocr_168x48_int8.rknn"
ROI_POSE_MODEL = (
    "/home/cat/RK3568/final/rknn_models/"
    "plate_serial_roi_pose26_320x160_w8a8_fp16.rknn"
)
ROI_POSE_SHA256 = "FB90CA333E4E5DEC61E8231B887D1F11E90927DE340258E19A504BECF560B10C"
RECOGNITION_CONFIDENCE = 0.45
SERIAL_ROI_EXPAND_FRACTION = 0.20


def expand_serial_rois(rois, fraction=SERIAL_ROI_EXPAND_FRACTION,
                       frame_width=1920, frame_height=1080):
    """Expand each FPGA XYXY edge and clip only the processing ROI."""
    expanded = []
    for roi in rois:
        x1, y1, x2, y2 = (int(roi[key]) for key in ("x1", "y1", "x2", "y2"))
        margin_x = max(1, int(math.ceil((x2 - x1) * float(fraction))))
        margin_y = max(1, int(math.ceil((y2 - y1) * float(fraction))))
        expanded.append({
            "x1": max(0, x1 - margin_x),
            "y1": max(0, y1 - margin_y),
            "x2": min(int(frame_width), x2 + margin_x),
            "y2": min(int(frame_height), y2 + margin_y),
        })
    return expanded


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description="车牌识别：FPGA串口ROI + Native YOLO26 Pose"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115205)
    parser.add_argument("--rec_model", "--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--det-threshold", type=float, default=0.25)
    parser.add_argument("--draw-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--hdmi-device", default=HDMI_DEVICE)
    parser.add_argument("--hdmi-width", type=int, default=1920)
    parser.add_argument("--hdmi-height", type=int, default=1080)
    parser.add_argument("--hdmi-fps", type=int, default=60)
    parser.add_argument("--hdmi-fourcc", default="MJPG")
    parser.add_argument(
        "--camera-fps-mode",
        choices=["stable30", "auto-exposure", "off"],
        default="auto-exposure",
    )
    parser.add_argument("--camera-exposure", type=int, default=None)
    parser.add_argument("--camera-gain", type=int, default=None)
    parser.add_argument("--camera-config-strict", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save_dir", "--save-dir", default="plate_crops")
    parser.add_argument(
        "--result-ttl-ms", "--async-result-ttl",
        dest="result_ttl_ms", type=int, default=2000,
    )
    parser.add_argument("--result-log-interval", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    add_publish_args(parser)
    return parser.parse_args(argv)


def validate_args(args):
    validate_publish_args(args)
    if (args.hdmi_width, args.hdmi_height) != (1920, 1080):
        raise ValueError("串口ROI坐标要求1920x1080输入")
    if args.hdmi_fourcc.upper() not in {"MJPG", "MJPEG"}:
        raise ValueError("串口模式要求MJPEG输入")
    if not 0.0 <= args.det_threshold <= 1.0:
        raise ValueError("det-threshold必须在0到1之间")
    if args.result_ttl_ms < 0:
        raise ValueError("result TTL不能为负数")
    if args.result_log_interval < 1:
        raise ValueError("result-log-interval必须大于0")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_roi_pose_model(path=ROI_POSE_MODEL):
    if not os.path.isfile(path):
        raise RuntimeError(f"串口ROI检测模型不存在: {path}")
    actual = _sha256(path)
    if actual != ROI_POSE_SHA256:
        raise RuntimeError(
            f"串口ROI检测模型SHA-256不匹配: {actual}; 需要{ROI_POSE_SHA256}"
        )
    return actual


def _load_detector():
    validate_roi_pose_model()
    model = NativeRknnModel(ROI_POSE_MODEL)
    expected_input = (1, 160, 320, 3)
    expected_outputs = {
        (1, 18, 40, 80), (1, 18, 20, 40),
        (1, 18, 10, 20), (1, 18, 5, 10),
    }
    if tuple(model.input_info["shape"]) != expected_input or {
        tuple(item["shape"]) for item in model.output_info
    } != expected_outputs:
        model.release()
        raise RuntimeError("串口ROI检测模型张量合同不匹配")
    return model


def _load_recognizer(path):
    model = NativeRknnModel(path)
    if int(model.input_info["elements"]) != 168 * 48 * 3:
        model.release()
        raise RuntimeError(f"识别模型输入异常: {model.input_info['shape']}")
    return model


def start_receiver(args):
    if args.camera_fps_mode != "off":
        status = configure_camera_fps(
            device=args.hdmi_device,
            width=args.hdmi_width,
            height=args.hdmi_height,
            fps=args.hdmi_fps,
            fourcc=args.hdmi_fourcc,
            mode=args.camera_fps_mode,
            exposure=args.camera_exposure,
            gain=args.camera_gain,
            strict=args.camera_config_strict,
            printer=print if args.debug else (lambda _message: None),
        )
        if status.warning and not args.debug:
            print(f"摄像头帧率警告: {status.message}")
    frame_queue = queue.Queue(maxsize=1)
    receiver = NativeMppHdmiReceiver(
        frame_queue,
        device=args.hdmi_device,
        width=args.hdmi_width,
        height=args.hdmi_height,
        fps=args.hdmi_fps,
        fourcc=args.hdmi_fourcc,
        profile=args.debug,
        profile_interval=DEBUG_INTERVAL,
        jpeg_tap_fps=args.publish_fps if args.publish else 0,
    )
    receiver.start()
    return frame_queue, receiver


def _release_event(event):
    if event is not None:
        release_frame(event)


def _replace_latest(target, value, release_stale=None):
    try:
        target.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        stale = target.get_nowait()
    except queue.Empty:
        stale = None
    if stale is not None and release_stale is not None:
        release_stale(stale)
    target.put_nowait(value)


class SerialRoiSource:
    """Reconnectable serial reader with an independent publisher snapshot."""

    def __init__(self, port, baud, latest_frame, on_packet, on_clear,
                 reconnect_seconds=1.0, printer=print):
        self.port = port
        self.baud = int(baud)
        self.latest_frame = latest_frame
        self.on_packet = on_packet
        self.on_clear = on_clear
        self.reconnect_seconds = float(reconnect_seconds)
        self.printer = printer
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="serial-roi-source")
        self._lock = threading.Lock()
        self._serial = None
        self._online = False
        self._last_error = ""
        self._sequence = 0
        self._packet = None
        self._last_error_log = 0.0

    @property
    def online(self):
        with self._lock:
            return self._online

    def start(self):
        self.thread.start()
        return self

    def _set_state(self, online, serial_handle=None, error=""):
        with self._lock:
            self._online = bool(online)
            self._serial = serial_handle
            self._last_error = str(error or "")

    def _record(self, rois, raw_packet):
        received_ns = time.monotonic_ns()
        received_at_ms = int(time.time() * 1000)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._packet = {
                "sequence": sequence,
                "received_at_ms": received_at_ms,
                "_received_ns": received_ns,
                "raw_hex": " ".join(f"{value:02X}" for value in raw_packet),
                "count": len(rois),
                "rois": [dict(roi) for roi in rois],
            }
        event = {
            "sequence": sequence,
            "received_ns": received_ns,
            "received_at_ms": received_at_ms,
            "raw_packet": bytes(raw_packet),
            "rois": [dict(roi) for roi in rois],
        }
        if rois:
            frame = self.latest_frame.acquire()
            if frame is not None:
                event["frame"] = frame
            self.on_packet(event)
        else:
            self.on_clear("count-zero", sequence)

    def snapshot(self, now_ns=None):
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            online = self._online
            packet = None if self._packet is None else dict(self._packet)
        if packet is None:
            return {
                "online": bool(online), "sequence": 0,
                "received_at_ms": None, "age_ms": None,
                "raw_hex": "", "count": 0, "rois": [],
            }
        received_ns = int(packet.pop("_received_ns"))
        packet["online"] = bool(online)
        packet["age_ms"] = max(0.0, current_ns - received_ns) / 1_000_000.0
        return packet

    def _log_error(self, text):
        now = time.monotonic()
        if now - self._last_error_log >= 5.0:
            self.printer(text)
            self._last_error_log = now

    def _run(self):
        while not self.stop_event.is_set():
            handle = None
            try:
                handle = open_serial(self.port, self.baud, timeout=0.2)
                self._set_state(True, handle)
                self.printer("串口在线，开始接收FPGA ROI")
                while not self.stop_event.is_set():
                    try:
                        rois, raw = read_plate_rois(handle, return_raw=True)
                    except SerialRoiProtocolError as exc:
                        self.on_clear("invalid-roi", None)
                        self._log_error(f"串口ROI协议错误，结果已清空: {exc}")
                        continue
                    self._record(rois, raw)
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self._set_state(False, None, exc)
                self._log_error(f"串口不可用，视频继续显示并后台重连: {exc}")
            finally:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
            self.stop_event.wait(self.reconnect_seconds)

    def stop(self):
        self.stop_event.set()
        with self._lock:
            handle = self._serial
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        self.thread.join(timeout=2.0)


class RoiPosePipeline:
    """Latest-wins RGA producer and one NPU detector/OCR consumer."""

    def __init__(self, args, latest_frame, result_slot):
        self.args = args
        self.latest_frame = latest_frame
        self.result_slot = result_slot
        self.stop_event = threading.Event()
        self.packet_queue = queue.Queue(maxsize=1)
        self.prepared_queue = queue.Queue(maxsize=1)
        self.free_slots = queue.Queue(maxsize=2)
        self.free_slots.put(0)
        self.free_slots.put(1)
        self.error_queue = queue.Queue(maxsize=1)
        self._barrier_lock = threading.Lock()
        self._clear_barrier = 0
        self.text_stabilizer = PlateTextStabilizer(
            iou_threshold=0.35,
            first_confirmations=2,
            switch_confirmations=3,
            track_ttl_ms=1200,
            group_key="roi_index",
        )
        self._last_stable_signature = ()
        self._last_result_metadata = {}
        self._model_gate = threading.Lock()
        self.detector = _load_detector()
        self.recognizer = _load_recognizer(args.rec_model)
        self.stats = {
            "packets": 0, "prepared": 0, "detections": 0,
            "recognitions": 0, "clears": 0, "errors": 0,
            "rga_ms": 0.0, "npu_ms": 0.0,
            "output_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0,
        }
        self.pack_thread = threading.Thread(
            target=self._pack_loop, daemon=True, name="serial-roi-rga"
        )
        self.infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="serial-roi-npu"
        )

    def start(self):
        self.pack_thread.start()
        self.infer_thread.start()
        return self

    def _barrier(self):
        with self._barrier_lock:
            return self._clear_barrier

    def submit(self, event):
        event = dict(event)
        event["source_rois"] = [dict(roi) for roi in event["rois"]]
        event["rois"] = expand_serial_rois(event["rois"])
        event["barrier"] = self._barrier()
        self.stats["packets"] += 1
        _replace_latest(self.packet_queue, event, _release_event)

    def clear(self, reason, sequence=None):
        with self._barrier_lock:
            self._clear_barrier += 1
            barrier = self._clear_barrier
            self.text_stabilizer.clear()
            self._last_stable_signature = ()
            self._last_result_metadata = {}
            self.result_slot.set({
                "_ts": time.monotonic(),
                "source": "rk-roi",
                "overlays": [],
                "source_timestamp_ns": time.monotonic_ns(),
                "ready_timestamp_ns": time.monotonic_ns(),
                "clear_barrier": barrier,
                "clear_reason": str(reason),
                "serial_sequence": sequence,
            })
        self.stats["clears"] += 1
        while True:
            try:
                _release_event(self.packet_queue.get_nowait())
            except queue.Empty:
                break

    @staticmethod
    def _stable_signature(results):
        return tuple((
            int(item.get("roi_index", -1)),
            tuple(round(float(value), 1) for value in item.get("bbox", ())),
            str(item.get("plate_no", "")),
            round(float(item.get("rec_conf", 0.0)), 3),
        ) for item in (results or ()))

    def refresh_stability(self):
        """Publish target expiry even when no new serial inference completes."""
        with self._barrier_lock:
            results = self.text_stabilizer.current()
            signature = self._stable_signature(results)
            if signature == self._last_stable_signature:
                return False
            self._last_stable_signature = signature
            now_ns = time.monotonic_ns()
            metadata = dict(self._last_result_metadata)
            metadata.update({
                "_ts": time.monotonic(),
                "source": "rk-roi",
                "overlays": results,
                "ready_timestamp_ns": now_ns,
            })
            metadata.setdefault("source_timestamp_ns", now_ns)
            self.result_slot.set(metadata)
            return True

    def _report_error(self, exc):
        self.stats["errors"] += 1
        try:
            self.error_queue.put_nowait(exc)
        except queue.Full:
            pass

    def check_error(self):
        try:
            error = self.error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"串口ROI流水线失败: {error}") from error

    def _release_prepared(self, job):
        if job is None:
            return
        release_frame(job)
        if not job.get("slot_released"):
            self.free_slots.put(int(job["slot"]))
            job["slot_released"] = True

    def _latest_after_slot(self, event):
        while True:
            try:
                newer = self.packet_queue.get_nowait()
            except queue.Empty:
                return event
            _release_event(event)
            event = newer

    def _pack_loop(self):
        active = None
        slot = None
        try:
            while not self.stop_event.is_set():
                try:
                    active = self.packet_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    slot = self.free_slots.get(timeout=0.1)
                except queue.Empty:
                    _replace_latest(self.packet_queue, active, _release_event)
                    active = None
                    continue
                active = self._latest_after_slot(active)
                if int(active.get("barrier", -1)) != self._barrier():
                    _release_event(active)
                    active = None
                    self.free_slots.put(slot)
                    slot = None
                    continue
                frame = active.get("frame")
                if frame is None:
                    frame = self.latest_frame.acquire()
                    if frame is None:
                        self.free_slots.put(slot)
                        slot = None
                        active = None
                        continue
                    active["frame"] = frame
                with self._model_gate:
                    prepared = self.detector.prepare_roi_pose26_nv12(
                        frame, active["rois"], slot
                    )
                job = dict(active)
                job.update({"slot": slot, "prepared": prepared})
                active = None
                slot = None
                self.stats["prepared"] += 1
                self.stats["rga_ms"] = float(prepared["preprocess_ms"])
                _replace_latest(
                    self.prepared_queue, job, self._release_prepared
                )
        except Exception as exc:
            if active is not None:
                _release_event(active)
            if slot is not None:
                self.free_slots.put(slot)
            if not self.stop_event.is_set():
                self._report_error(exc)

    def _rebuild_detector_and_prepare(self, job):
        with self._model_gate:
            self.detector.release()
            self.detector = _load_detector()
            while True:
                try:
                    stale = self.prepared_queue.get_nowait()
                except queue.Empty:
                    break
                self._release_prepared(stale)
            prepared = self.detector.prepare_roi_pose26_nv12(
                job["frame"], job["rois"], job["slot"]
            )
        job["prepared"] = prepared
        return self.detector.detect_roi_pose26_prepared(
            prepared, self.args.det_threshold
        )

    def _save_crop(self, crop, sequence, index, plate_text):
        if not self.args.save_crops:
            return
        os.makedirs(self.args.save_dir, exist_ok=True)
        name = f"{sequence:08d}_{index}_{plate_text or 'unknown'}.jpg"
        cv2.imwrite(os.path.join(self.args.save_dir, name), crop)

    def _publish(self, job, detections, crops):
        if int(job["barrier"]) != self._barrier():
            return
        results = []
        recognition_npu_ms = 0.0
        recognition_runs = 0
        for index, (detection, crop) in enumerate(zip(detections, crops)):
            if int(job["barrier"]) != self._barrier():
                return
            rebuilt = False
            while True:
                try:
                    plate_text, plate_color, rec_conf, color_conf = (
                        recognize_rknn(self.recognizer, crop)
                    )
                    ocr_timing = (
                        getattr(self.recognizer, "last_ocr_timing", None) or {}
                    )
                    recognition_npu_ms += max(
                        0.0, float(ocr_timing.get("npu_ms", 0.0) or 0.0)
                    )
                    recognition_runs += 1
                    break
                except Exception:
                    if rebuilt:
                        raise
                    rebuilt = True
                    self.recognizer.release()
                    self.recognizer = _load_recognizer(self.args.rec_model)
            self.stats["recognitions"] += 1
            if (
                not plate_text or len(str(plate_text)) < 5 or
                float(rec_conf) < RECOGNITION_CONFIDENCE or
                float(color_conf) < RECOGNITION_CONFIDENCE
            ):
                continue
            self._save_crop(crop, int(job["sequence"]), index, plate_text)
            landmarks = detection["landmarks"]
            bbox = detection["bbox"]
            result = {
                "plate_no": plate_text,
                "rec_conf": float(rec_conf),
                "_plate_color": plate_color,
                "_color_conf": float(color_conf),
                "plate_type": int(detection["class"]),
                "bbox": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
                "landmarks": (
                    landmarks.tolist() if hasattr(landmarks, "tolist")
                    else landmarks
                ),
                "landmark_visibility": detection[
                    "landmark_visibility"
                ].tolist(),
                "detect_conf": float(detection["conf"]),
                "roi_index": int(detection["roi_index"]),
                "source": "rk-roi",
                "geometry_source": "fpga-pose",
                "recognition_source": "rk",
                "rgba": SERIAL_OVERLAY_RGBA,
            }
            # 只有进入本循环的Pose命中才允许显示；框坐标取该命中
            # roi_index对应的原始串口ROI，而不是为检测扩张后的ROI。
            overlay = serial_pose_confirmed_roi_overlay(
                result, job.get("source_rois")
            )
            if overlay is not None:
                results.append(overlay)
        with self._barrier_lock:
            if int(job["barrier"]) != self._clear_barrier:
                return
            results = self.text_stabilizer.update(results)
            ready_ns = time.monotonic_ns()
            detection_npu_ms = float(self.stats["npu_ms"])
            published_recognition_npu_ms = (
                recognition_npu_ms if recognition_runs else None
            )
            total_latency_ms = (
                ready_ns - int(job["received_ns"])
            ) / 1_000_000.0
            metadata = {
                "_ts": time.monotonic(),
                "source": "rk-roi",
                "overlays": results,
                "source_timestamp_ns": int(job["received_ns"]),
                "ready_timestamp_ns": ready_ns,
                "serial_sequence": int(job["sequence"]),
                "detection_npu_ms": detection_npu_ms,
                "recognition_npu_ms": published_recognition_npu_ms,
                "detection_latency_ms": detection_npu_ms,
                "recognition_latency_ms": published_recognition_npu_ms,
                "total_latency_ms": total_latency_ms,
            }
            self._last_stable_signature = self._stable_signature(results)
            self._last_result_metadata = dict(metadata)
            self.result_slot.set(metadata)
        if results and int(job["sequence"]) % self.args.result_log_interval == 0:
            print("识别结果: " + ", ".join(
                f"ROI{item['roi_index']} {item['plate_no']} "
                f"({item['rec_conf']:.2f})" for item in results
            ))

    def _infer_loop(self):
        active = None
        try:
            while not self.stop_event.is_set():
                try:
                    active = self.prepared_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if int(active["barrier"]) != self._barrier():
                    self._release_prepared(active)
                    active = None
                    continue
                try:
                    detections = self.detector.detect_roi_pose26_prepared(
                        active["prepared"], self.args.det_threshold
                    )
                except Exception:
                    detections = self._rebuild_detector_and_prepare(active)
                timing = self.detector.last_detection_timing or {}
                for key in ("npu_ms", "output_ms", "decode_ms", "total_ms"):
                    self.stats[key] = float(timing.get(key, 0.0))
                self.stats["detections"] += len(detections)
                crops = []
                kept = []
                if int(active["barrier"]) == self._barrier():
                    for detection in detections:
                        try:
                            crops.append(prepare_plate_image(
                                active["frame"], detection
                            ))
                            kept.append(detection)
                        except (ValueError, RuntimeError):
                            continue
                release_frame(active)
                active["frame"] = None
                self.free_slots.put(int(active["slot"]))
                active["slot_released"] = True
                if int(active["barrier"]) == self._barrier():
                    self._publish(active, kept, crops)
                active = None
        except Exception as exc:
            if active is not None:
                self._release_prepared(active)
            if not self.stop_event.is_set():
                self._report_error(exc)

    def stop(self):
        self.clear("runtime-stop")
        self.stop_event.set()
        self.pack_thread.join(timeout=2.0)
        self.infer_thread.join(timeout=5.0)
        while True:
            try:
                _release_event(self.packet_queue.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                self._release_prepared(self.prepared_queue.get_nowait())
            except queue.Empty:
                break
        self.detector.release()
        self.recognizer.release()


def _overlay_signature(results, source_ns):
    return (
        int(source_ns or 0),
        tuple((
            tuple(round(float(value), 1) for value in item.get("bbox", ())),
            item.get("plate_no", ""),
            round(float(item.get("rec_conf", 0.0)), 3),
        ) for item in results or ()),
    )


def serial_pose_confirmed_roi_overlay(result, source_rois):
    """Use an original serial ROI only after its Pose slot detects a plate."""
    output = dict(result)
    output.pop("polygon", None)
    output.pop("polygon_landmarks", None)
    try:
        roi_index = int(output["roi_index"])
        roi = source_rois[roi_index]
        bbox = [int(roi[key]) for key in ("x1", "y1", "x2", "y2")]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (0 <= roi_index < len(source_rois)):
        return None
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return None
    output["bbox"] = bbox
    output["geometry_source"] = "fpga-pose"
    return output


def run_native(args):
    display = AtomicNv12Display(font_path=args.font_path)
    try:
        frame_queue, receiver = start_receiver(args)
    except Exception:
        display.close()
        raise
    display_queue = queue.Queue(maxsize=1)
    display_errors = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=native_display_worker,
        args=(display, display_queue, stop_event, display_errors, {}),
        daemon=True,
        name="atomic-display",
    )
    display_thread.start()
    latest_frame = OwnedLatestFrame()
    result_slot = LatestValue()
    pipeline = None
    serial_source = None
    publisher = None
    try:
        pipeline = RoiPosePipeline(args, latest_frame, result_slot).start()
        serial_source = SerialRoiSource(
            args.port, args.baud, latest_frame,
            pipeline.submit, pipeline.clear,
            printer=print if args.debug else (lambda _message: None),
        ).start()
        publisher = FramePublisher.from_args(
            args, mode="plate_serial", jpeg_provider=receiver.read_latest_jpeg
        )
        if publisher is not None:
            publisher.start()
        frame_count = 0
        started = time.monotonic()
        latest = None
        last_signature = None
        while True:
            pipeline.check_error()
            try:
                error = display_errors.get_nowait()
                raise RuntimeError(f"Atomic显示线程失败: {error}")
            except queue.Empty:
                pass
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                if not receiver.is_alive():
                    raise RuntimeError(f"采集线程退出: {receiver.error}")
                continue
            latest_frame.publish(frame.retain())
            put_latest(display_queue, frame, release_dropped=True)
            frame_count += 1
            pipeline.refresh_stability()
            candidate = result_slot.get_fresh(args.result_ttl_ms)
            if candidate is not None:
                latest = candidate
            elif latest is not None and (
                time.monotonic() - latest["_ts"]
            ) * 1000.0 > args.result_ttl_ms:
                latest = None
            overlays = (latest or {}).get("overlays", [])
            source_ns = (latest or {}).get("source_timestamp_ns", 0)
            ready_ns = (latest or {}).get("ready_timestamp_ns", 0)
            signature = _overlay_signature(overlays, source_ns)
            if signature != last_signature:
                display.submit_overlay(
                    overlays,
                    ttl_ms=args.result_ttl_ms,
                    draw_mode=args.draw_mode,
                    source_timestamp_ns=int(source_ns or 0),
                    ready_timestamp_ns=int(ready_ns or 0),
                )
                last_signature = signature
            native_stats = display.get_stats()
            elapsed = max(time.monotonic() - started, 1e-6)
            if publisher is not None:
                publisher.update(
                    overlays=plate_overlays(overlays, "rk-roi"),
                    results=[{
                        "plate_no": item.get("plate_no", ""),
                        "confidence": item.get("rec_conf", 0.0),
                        "source": "rk-roi",
                    } for item in overlays if item.get("plate_no")],
                    serial_data=serial_source.snapshot(),
                    metrics={
                        "capture_fps": frame_count / elapsed,
                        "page_flip_fps": native_stats.display_fps,
                        "detect_packets": pipeline.stats["packets"],
                        "detections": pipeline.stats["detections"],
                        "recognitions": pipeline.stats["recognitions"],
                        "detect_rga_ms": pipeline.stats["rga_ms"],
                        "detect_npu_ms": pipeline.stats["npu_ms"],
                        "detect_output_ms": pipeline.stats["output_ms"],
                        "detect_decode_ms": pipeline.stats["decode_ms"],
                        "detect_total_ms": pipeline.stats["total_ms"],
                        "detection_latency_ms": (latest or {}).get(
                            "detection_latency_ms"
                        ),
                        "recognition_latency_ms": (latest or {}).get(
                            "recognition_latency_ms"
                        ),
                        "total_latency_ms": (latest or {}).get(
                            "total_latency_ms"
                        ),
                        "dropped_frames": receiver.native_drop_count(),
                    },
                )
            if args.debug and frame_count % DEBUG_INTERVAL == 0:
                print(
                    f"[调试 {frame_count}] "
                    f"PageFlip:{native_stats.display_fps:.1f}FPS | "
                    f"串口:{'在线' if serial_source.online else '离线'} | "
                    f"包:{pipeline.stats['packets']} | "
                    f"RGA:{pipeline.stats['rga_ms']:.2f}ms | "
                    f"NPU:{pipeline.stats['npu_ms']:.2f}ms | "
                    f"输出:{pipeline.stats['output_ms']:.2f}ms | "
                    f"解码:{pipeline.stats['decode_ms']:.2f}ms | "
                    f"车牌:{len(overlays)} | "
                    f"丢帧:{receiver.native_drop_count()}"
                )
    finally:
        if publisher is not None:
            publisher.stop()
        if serial_source is not None:
            serial_source.stop()
        if pipeline is not None:
            pipeline.stop()
        stop_event.set()
        receiver.terminate()
        receiver.join(timeout=3.0)
        latest_frame.clear()
        while True:
            try:
                release_frame(frame_queue.get_nowait())
            except queue.Empty:
                break
        display_thread.join(timeout=3.0)
        display.close()


def main(argv=None):
    args = build_args(argv)
    validate_args(args)
    print("=" * 60)
    print("FPGA串口ROI + Native YOLO26 Pose车牌识别")
    print(
        f"检测模型:{ROI_POSE_MODEL} 阈值:{args.det_threshold:.2f} "
        f"OCR:{args.rec_model}"
    )
    print("=" * 60)
    try:
        run_native(args)
    except KeyboardInterrupt:
        print("\n程序被终止，正在清理...")
    finally:
        print("资源清理完毕。")


# 历史入口名称保留，但行为已经统一为ROI运行时。
run_native_hybrid = run_native


if __name__ == "__main__":
    main()
