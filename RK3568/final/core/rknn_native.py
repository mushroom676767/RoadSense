"""Small compatibility wrapper around the RKNPU2 C runtime API."""

from __future__ import annotations

import ctypes
from pathlib import Path
import time

import cv2
import numpy as np


MAX_DIMS = 16


class NativeTensorInfo(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * MAX_DIMS),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("type", ctypes.c_int32),
        ("format", ctypes.c_int32),
        ("qnt_type", ctypes.c_int32),
        ("zero_point", ctypes.c_int32),
        ("scale", ctypes.c_float),
    ]


class NativePlateDetection(ctypes.Structure):
    """与C端最终车牌候选结构严格一致的ctypes ABI。"""

    _fields_ = [
        ("bbox", ctypes.c_float * 4),
        ("landmarks", ctypes.c_float * 8),
        ("landmark_visibility", ctypes.c_float * 4),
        ("confidence", ctypes.c_float),
        ("class_id", ctypes.c_int32),
    ]


class NativeRoiRect(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("x2", ctypes.c_int32),
        ("y2", ctypes.c_int32),
    ]


class NativeRoiSlotInfo(ctypes.Structure):
    _fields_ = [
        ("roi_index", ctypes.c_int32),
        ("slot_left", ctypes.c_int32),
        ("slot_top", ctypes.c_int32),
        ("ratio", ctypes.c_float),
        ("pad_x", ctypes.c_float),
        ("pad_y", ctypes.c_float),
        ("requested", NativeRoiRect),
        ("aligned", NativeRoiRect),
    ]


class NativeRoiPlateDetection(ctypes.Structure):
    _fields_ = NativePlateDetection._fields_ + [
        ("roi_index", ctypes.c_int32),
    ]


class NativeDetectionTiming(ctypes.Structure):
    """记录Native检测快路径各阶段耗时及候选数量。"""

    _fields_ = [
        ("preprocess_ms", ctypes.c_float),
        ("npu_ms", ctypes.c_float),
        ("output_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("nms_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
        ("anchors_scanned", ctypes.c_uint32),
        ("candidates_decoded", ctypes.c_uint32),
    ]


class NativeOcrResult(ctypes.Structure):
    """与C端融合OCR结果及阶段耗时严格一致的ctypes ABI。"""

    _fields_ = [
        ("char_indices", ctypes.c_int32 * 21),
        ("char_count", ctypes.c_uint32),
        ("color_index", ctypes.c_int32),
        ("text_confidence", ctypes.c_float),
        ("color_confidence", ctypes.c_float),
        ("resize_ms", ctypes.c_float),
        ("npu_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
    ]


class NativeRknnModel:
    """Expose the subset of rknn.api.RKNN used by this project."""

    def __init__(self, model_path, library_path=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        native_dir = Path(__file__).resolve().parent / "native"
        path = Path(library_path) if library_path else native_dir / "librknn_runtime.so"
        if not path.exists():
            raise RuntimeError(f"native RKNN runtime helper not found: {path}")
        self._lib = ctypes.CDLL(str(path))
        self._bind()
        err = ctypes.create_string_buffer(512)
        self._handle = self._lib.rknn_native_open(
            str(model_path).encode("utf-8"), err, len(err)
        )
        if not self._handle:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        self.last_detection_timing = None
        self.last_ocr_timing = None
        self.input_info = self._get_input_info(err)
        count = self._lib.rknn_native_output_count(self._handle)
        if count <= 0:
            self.release()
            raise RuntimeError("native RKNN model has no outputs")
        self.output_info = []
        self._outputs = []
        for index in range(count):
            info = NativeTensorInfo()
            if self._lib.rknn_native_get_output_info(
                self._handle, index, ctypes.byref(info), err, len(err)
            ) != 0:
                self.release()
                raise RuntimeError(err.value.decode("utf-8", "replace"))
            shape = tuple(int(info.dims[i]) for i in range(info.n_dims))
            self.output_info.append({
                "shape": shape,
                "elements": int(info.n_elems),
                "size": int(info.size),
                "type": int(info.type),
                "format": int(info.format),
                "quantization": int(info.qnt_type),
                "zero_point": int(info.zero_point),
                "scale": float(info.scale),
            })
            # C端INT8检测快路径不需要float输出；兼容路径首次调用时再分配。
            self._outputs.append(None)

    def _bind(self):
        """实现模块内部的“bind”辅助逻辑，不作为新的外部接口。"""
        lib = self._lib
        lib.rknn_native_open.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t
        ]
        lib.rknn_native_open.restype = ctypes.c_void_p
        lib.rknn_native_output_count.argtypes = [ctypes.c_void_p]
        lib.rknn_native_output_count.restype = ctypes.c_int
        lib.rknn_native_get_input_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NativeTensorInfo),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_get_input_info.restype = ctypes.c_int
        lib.rknn_native_get_output_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(NativeTensorInfo),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_get_output_info.restype = ctypes.c_int
        lib.rknn_native_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_run.restype = ctypes.c_int
        lib.rknn_native_prepare_nv12_slot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_prepare_nv12_slot.restype = ctypes.c_int
        lib.rknn_native_detect_pose26_prepared.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(NativePlateDetection),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(NativeDetectionTiming),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_detect_pose26_prepared.restype = ctypes.c_int
        lib.rknn_native_prepare_roi_pose26_slot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(NativeRoiRect),
            ctypes.c_int,
            ctypes.POINTER(NativeRoiSlotInfo),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_prepare_roi_pose26_slot.restype = ctypes.c_int
        lib.rknn_native_detect_roi_pose26_prepared.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(NativeRoiSlotInfo),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(NativeRoiPlateDetection),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(NativeDetectionTiming),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_detect_roi_pose26_prepared.restype = ctypes.c_int
        lib.rknn_native_recognize_bgr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(NativeOcrResult),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_recognize_bgr.restype = ctypes.c_int
        lib.rknn_native_prepare_nv12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_prepare_nv12.restype = ctypes.c_int
        lib.rknn_native_run_prepared.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_run_prepared.restype = ctypes.c_int
        lib.rknn_native_get_last_run_us.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rknn_native_get_last_run_us.restype = ctypes.c_int
        lib.rknn_native_close.argtypes = [ctypes.c_void_p]
        lib.rknn_native_close.restype = None

    def _get_input_info(self, err):
        """查询并缓存Native RKNN输入尺寸、布局、类型和量化信息。"""
        info = NativeTensorInfo()
        if self._lib.rknn_native_get_input_info(
            self._handle, ctypes.byref(info), err, len(err)
        ) != 0:
            self.release()
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return {
            "shape": tuple(int(info.dims[i]) for i in range(info.n_dims)),
            "elements": int(info.n_elems),
            "size": int(info.size),
            "type": int(info.type),
            "format": int(info.format),
            "quantization": int(info.qnt_type),
            "zero_point": int(info.zero_point),
            "scale": float(info.scale),
        }

    def _output_buffers(self):
        """实现模块内部的“output buffers”辅助逻辑，不作为新的外部接口。"""
        for index, output in enumerate(self._outputs):
            if output is None:
                self._outputs[index] = np.empty(
                    self.output_info[index]["shape"],
                    dtype=np.float32,
                )
        pointers = (ctypes.c_void_p * len(self._outputs))(
            *(ctypes.c_void_p(output.ctypes.data) for output in self._outputs)
        )
        sizes = (ctypes.c_size_t * len(self._outputs))(
            *(output.nbytes for output in self._outputs)
        )
        return pointers, sizes

    def tensor_summary(self):
        """返回模型输入、输出尺寸及量化参数的单行诊断信息。"""
        outputs = [item["shape"] for item in self.output_info]
        return (
            f"input={self.input_info['shape']} "
            f"type={self.input_info['type']} "
            f"qnt={self.input_info['quantization']} "
            f"scale={self.input_info['scale']:.9g} "
            f"zp={self.input_info['zero_point']} "
            f"outputs={outputs}"
        )

    def inference(self, inputs, data_format="nhwc"):
        """执行一次模型推理并返回与旧后端一致的输出数组。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        if len(inputs) != 1:
            raise ValueError("native RKNN wrapper expects exactly one input")
        input_array = np.ascontiguousarray(inputs[0])
        if input_array.dtype == np.uint8:
            input_kind = 0
        elif input_array.dtype == np.float32:
            input_kind = 1
        else:
            raise TypeError("RKNN input must be uint8 or float32")
        if data_format not in {"nhwc", "nchw"}:
            raise ValueError("data_format must be nhwc or nchw")
        if data_format == "nchw":
            if input_array.ndim != 4:
                raise ValueError("NCHW input must be four-dimensional")
            input_array = np.ascontiguousarray(
                input_array.transpose(0, 2, 3, 1)
            )
        pointers, sizes = self._output_buffers()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_run(
            self._handle,
            ctypes.c_void_p(input_array.ctypes.data),
            input_array.nbytes,
            input_kind,
            0,
            pointers,
            sizes,
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return self._outputs

    def recognize_plate_bgr(self, plate_roi):
        """用OpenCV缩放BGR ROI，再融合执行RKNN与Native解码。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        if (
            not isinstance(plate_roi, np.ndarray)
            or plate_roi.dtype != np.uint8
            or plate_roi.ndim != 3
            or plate_roi.shape[2] != 3
        ):
            raise ValueError("plate_roi must be HxWx3 uint8 BGR")
        resize_started = time.perf_counter_ns()
        roi = cv2.resize(
            plate_roi, (168, 48), interpolation=cv2.INTER_LINEAR
        )
        roi = np.ascontiguousarray(roi, dtype=np.uint8)
        resize_ms = (time.perf_counter_ns() - resize_started) / 1_000_000.0
        height, width = roi.shape[:2]
        source_span = (height - 1) * roi.strides[0] + width * 3
        native_result = NativeOcrResult()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_recognize_bgr(
            self._handle,
            roi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(width),
            int(height),
            int(roi.strides[0]),
            int(source_span),
            ctypes.byref(native_result),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        count = int(native_result.char_count)
        if count < 0 or count > len(native_result.char_indices):
            raise RuntimeError(f"native OCR returned invalid character count: {count}")
        self.last_ocr_timing = {
            "resize_ms": float(resize_ms),
            "npu_ms": float(native_result.npu_ms),
            "decode_ms": float(native_result.decode_ms),
            "total_ms": float(resize_ms + native_result.total_ms),
        }
        return (
            [int(native_result.char_indices[index]) for index in range(count)],
            int(native_result.color_index),
            float(native_result.text_confidence),
            float(native_result.color_confidence),
        )

    def prepare_nv12(self, frame):
        """把NV12 DMA-BUF同步写入持久RKNN输入内存。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        ratio = ctypes.c_float()
        pad_x = ctypes.c_int()
        pad_y = ctypes.c_int()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_prepare_nv12(
            self._handle,
            int(frame.dma_fd),
            int(frame.width),
            int(frame.height),
            int(frame.hor_stride),
            int(frame.ver_stride),
            ctypes.byref(ratio),
            ctypes.byref(pad_x),
            ctypes.byref(pad_y),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return float(ratio.value), (pad_x.value, pad_y.value)

    def inference_prepared(self):
        """对prepare_nv12已准备的输入执行推理，此时不再引用源帧。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        pointers, sizes = self._output_buffers()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_run_prepared(
            self._handle, pointers, sizes, err, len(err)
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return self._outputs

    def prepare_pose26_nv12(self, frame, slot):
        """把一帧NV12同步写入指定Pose26输入槽，不运行NPU。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        slot = int(slot)
        if slot not in (0, 1):
            raise ValueError("Pose26 input slot must be 0 or 1")
        ratio = ctypes.c_float()
        pad_x = ctypes.c_int()
        pad_y = ctypes.c_int()
        preprocess_ms = ctypes.c_float()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_prepare_nv12_slot(
            self._handle,
            slot,
            int(frame.dma_fd),
            int(frame.width),
            int(frame.height),
            int(frame.hor_stride),
            int(frame.ver_stride),
            ctypes.byref(ratio),
            ctypes.byref(pad_x),
            ctypes.byref(pad_y),
            ctypes.byref(preprocess_ms),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return {
            "slot": slot,
            "source_width": int(frame.width),
            "source_height": int(frame.height),
            "ratio": float(ratio.value),
            "pad_x": int(pad_x.value),
            "pad_y": int(pad_y.value),
            "preprocess_ms": float(preprocess_ms.value),
        }

    def detect_pose26_prepared(
        self, prepared, conf_threshold, max_results=32
    ):
        """在指定双缓冲槽上运行Pose26 NPU与Native解码。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        max_results = int(max_results)
        if max_results <= 0 or max_results > 300:
            raise ValueError("max_results must be in [1, 300]")
        slot = int(prepared["slot"])
        native_results = (NativePlateDetection * max_results)()
        result_count = ctypes.c_int()
        timing = NativeDetectionTiming()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_detect_pose26_prepared(
            self._handle,
            slot,
            int(prepared["source_width"]),
            int(prepared["source_height"]),
            float(prepared["ratio"]),
            int(prepared["pad_x"]),
            int(prepared["pad_y"]),
            float(prepared["preprocess_ms"]),
            float(conf_threshold),
            native_results,
            max_results,
            ctypes.byref(result_count),
            ctypes.byref(timing),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        self.last_detection_timing = {
            "preprocess_ms": float(timing.preprocess_ms),
            "npu_ms": float(timing.npu_ms),
            "output_ms": float(timing.output_ms),
            "decode_ms": float(timing.decode_ms),
            "nms_ms": float(timing.nms_ms),
            "total_ms": float(timing.total_ms),
            "anchors_scanned": int(timing.anchors_scanned),
            "candidates_decoded": int(timing.candidates_decoded),
        }
        detections = []
        for index in range(result_count.value):
            detection = native_results[index]
            detections.append({
                "bbox": np.asarray(
                    detection.bbox, dtype=np.float32
                ).copy(),
                "landmarks": np.asarray(
                    detection.landmarks, dtype=np.float32
                ).reshape(4, 2).copy(),
                "landmark_visibility": np.asarray(
                    detection.landmark_visibility, dtype=np.float32
                ).copy(),
                "conf": float(detection.confidence),
                "class": int(detection.class_id),
            })
        return detections

    def prepare_roi_pose26_nv12(self, frame, rois, slot):
        """RGA-pack 1..4 full-frame XYXY ROIs into one prepared input slot."""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        slot = int(slot)
        if slot not in (0, 1):
            raise ValueError("ROI Pose26 input slot must be 0 or 1")
        rois = list(rois or ())
        if not 1 <= len(rois) <= 4:
            raise ValueError("ROI Pose26 requires 1 to 4 ROIs")
        native_rois = (NativeRoiRect * len(rois))()
        slot_infos = (NativeRoiSlotInfo * len(rois))()
        for index, roi in enumerate(rois):
            values = (
                roi.get("x1"), roi.get("y1"),
                roi.get("x2"), roi.get("y2"),
            ) if isinstance(roi, dict) else tuple(roi)
            if len(values) != 4:
                raise ValueError(f"ROI{index} must contain four XYXY values")
            native_rois[index] = NativeRoiRect(*(int(value) for value in values))
        preprocess_ms = ctypes.c_float()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_prepare_roi_pose26_slot(
            self._handle,
            slot,
            int(frame.dma_fd),
            int(frame.width),
            int(frame.height),
            int(frame.hor_stride),
            int(frame.ver_stride),
            native_rois,
            len(rois),
            slot_infos,
            len(rois),
            ctypes.byref(preprocess_ms),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return {
            "slot": slot,
            "roi_count": len(rois),
            "preprocess_ms": float(preprocess_ms.value),
            "slot_infos": slot_infos,
            "rois": [dict(roi) if isinstance(roi, dict) else {
                key: int(value) for key, value in zip(
                    ("x1", "y1", "x2", "y2"), roi
                )
            } for roi in rois],
        }

    def detect_roi_pose26_prepared(self, prepared, conf_threshold=0.25):
        """Run the fixed FP16 ROI Pose26 model and return at most one per ROI."""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        roi_count = int(prepared["roi_count"])
        native_results = (NativeRoiPlateDetection * 4)()
        result_count = ctypes.c_int()
        timing = NativeDetectionTiming()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_detect_roi_pose26_prepared(
            self._handle,
            int(prepared["slot"]),
            prepared["slot_infos"],
            roi_count,
            float(prepared["preprocess_ms"]),
            float(conf_threshold),
            native_results,
            4,
            ctypes.byref(result_count),
            ctypes.byref(timing),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        self.last_detection_timing = {
            "preprocess_ms": float(timing.preprocess_ms),
            "npu_ms": float(timing.npu_ms),
            "output_ms": float(timing.output_ms),
            "decode_ms": float(timing.decode_ms),
            "nms_ms": float(timing.nms_ms),
            "total_ms": float(timing.total_ms),
            "anchors_scanned": int(timing.anchors_scanned),
            "candidates_decoded": int(timing.candidates_decoded),
        }
        detections = []
        for index in range(result_count.value):
            detection = native_results[index]
            detections.append({
                "bbox": np.asarray(detection.bbox, dtype=np.float32).copy(),
                "landmarks": np.asarray(
                    detection.landmarks, dtype=np.float32
                ).reshape(4, 2).copy(),
                "landmark_visibility": np.asarray(
                    detection.landmark_visibility, dtype=np.float32
                ).copy(),
                "conf": float(detection.confidence),
                "class": int(detection.class_id),
                "roi_index": int(detection.roi_index),
            })
        return detections

    def last_run_duration_ms(self):
        """返回RKNN驱动报告的最近一次纯NPU执行时间。"""
        if not self._handle:
            raise RuntimeError("native RKNN model is released")
        duration_us = ctypes.c_int64()
        err = ctypes.create_string_buffer(512)
        result = self._lib.rknn_native_get_last_run_us(
            self._handle,
            ctypes.byref(duration_us),
            err,
            len(err),
        )
        if result != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return max(0.0, float(duration_us.value) / 1000.0)

    def release(self):
        """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
        if getattr(self, "_handle", None):
            self._lib.rknn_native_close(self._handle)
            self._handle = None
