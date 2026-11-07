"""ctypes wrapper for NV12 + ARGB Atomic KMS composition."""

from __future__ import annotations

import ctypes
from pathlib import Path


MAX_OVERLAY_ITEMS = 32
ITEM_LABEL = 0x01
ITEM_CORNERS = 0x02
ITEM_FULL = 0x04
ITEM_GENERIC = 0x08
ITEM_POLYGON = 0x10


class AtomicOverlayItem(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("x2", ctypes.c_int32),
        ("y2", ctypes.c_int32),
        ("landmarks", ctypes.c_float * 8),
        ("detect_conf", ctypes.c_float),
        ("rec_conf", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("plate_no", ctypes.c_char * 32),
        ("plate_color", ctypes.c_char * 24),
        ("label", ctypes.c_char * 64),
        ("score", ctypes.c_float),
        ("red", ctypes.c_uint8),
        ("green", ctypes.c_uint8),
        ("blue", ctypes.c_uint8),
        ("alpha", ctypes.c_uint8),
    ]


class AtomicDisplayStats(ctypes.Structure):
    _fields_ = [
        ("submitted", ctypes.c_uint64),
        ("displayed", ctypes.c_uint64),
        ("commit_errors", ctypes.c_uint64),
        ("overlay_updates", ctypes.c_uint64),
        ("overlay_expirations", ctypes.c_uint64),
        ("commit_total_ms", ctypes.c_double),
        ("commit_max_ms", ctypes.c_double),
        ("overlay_total_ms", ctypes.c_double),
        ("overlay_max_ms", ctypes.c_double),
        ("display_fps", ctypes.c_double),
        ("last_flip_ns", ctypes.c_uint64),
        ("last_overlay_source_timestamp_ns", ctypes.c_uint64),
        ("last_overlay_ready_timestamp_ns", ctypes.c_uint64),
        ("last_overlay_e2e_ms", ctypes.c_double),
        ("last_overlay_display_ms", ctypes.c_double),
    ]


def _utf8_field(value, capacity):
    """实现模块内部的“utf8 field”辅助逻辑，不作为新的外部接口。"""
    encoded = str(value or "").encode("utf-8")
    return encoded[: capacity - 1]


class AtomicNv12Display:
    def __init__(self, device="/dev/dri/card0", font_path=None, library_path=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        native_dir = Path(__file__).resolve().parent / "native"
        path = Path(library_path) if library_path else native_dir / "librkdrm_atomic.so"
        if not path.exists():
            raise RuntimeError(f"Atomic DRM helper not found: {path}")
        self._lib = ctypes.CDLL(str(path))
        self._bind()
        self._handle = None
        err = ctypes.create_string_buffer(512)
        width = ctypes.c_int()
        height = ctypes.c_int()
        video_plane = ctypes.c_uint32()
        overlay_plane = ctypes.c_uint32()
        encoded_font = str(font_path).encode("utf-8") if font_path else None
        self._handle = self._lib.rkdrm_atomic_open(
            str(device).encode("utf-8"),
            encoded_font,
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(video_plane),
            ctypes.byref(overlay_plane),
            err,
            len(err),
        )
        if not self._handle:
            raise RuntimeError(
                err.value.decode("utf-8", "replace") or "Atomic DRM初始化失败"
            )
        self.width = width.value
        self.height = height.value
        self.video_plane = video_plane.value
        self.overlay_plane = overlay_plane.value
        self.font_path = self._lib.rkdrm_atomic_font_path(self._handle).decode(
            "utf-8", "replace"
        )
        print(
            f"DRM Atomic NV12: {self.width}x{self.height}, "
            f"video-plane={self.video_plane}, overlay-plane={self.overlay_plane}, "
            f"font={self.font_path}",
            flush=True,
        )

    def _bind(self):
        """实现模块内部的“bind”辅助逻辑，不作为新的外部接口。"""
        lib = self._lib
        lib.rkdrm_atomic_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkdrm_atomic_open.restype = ctypes.c_void_p
        lib.rkdrm_atomic_submit_overlay.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AtomicOverlayItem),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkdrm_atomic_submit_overlay.restype = ctypes.c_int
        lib.rkdrm_atomic_show_nv12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkdrm_atomic_show_nv12.restype = ctypes.c_int
        lib.rkdrm_atomic_get_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AtomicDisplayStats),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkdrm_atomic_get_stats.restype = ctypes.c_int
        lib.rkdrm_atomic_overlay_percentile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
        ]
        lib.rkdrm_atomic_overlay_percentile.restype = ctypes.c_double
        lib.rkdrm_atomic_font_path.argtypes = [ctypes.c_void_p]
        lib.rkdrm_atomic_font_path.restype = ctypes.c_char_p
        lib.rkdrm_atomic_close.argtypes = [ctypes.c_void_p]
        lib.rkdrm_atomic_close.restype = None

    def submit_overlay(self, results, ttl_ms=500, draw_mode="fast",
                       source_timestamp_ns=0, ready_timestamp_ns=0):
        """把最新标注状态提交给C Overlay，绘制本身不阻塞推理线程。"""
        selected = list(results or [])[:MAX_OVERLAY_ITEMS]
        array = (AtomicOverlayItem * max(1, len(selected)))()
        for index, result in enumerate(selected):
            item = array[index]
            bbox = result.get("bbox", (0, 0, 0, 0))
            item.x1, item.y1, item.x2, item.y2 = (int(value) for value in bbox)
            flat_landmarks = []
            polygon_landmarks = result.get(
                "polygon_landmarks", result.get("landmarks", ())
            )
            for point in polygon_landmarks or ():
                flat_landmarks.extend((float(point[0]), float(point[1])))
            for point_index, value in enumerate(flat_landmarks[:8]):
                item.landmarks[point_index] = value
            plate_no = result.get("plate_no", "")
            item.detect_conf = float(result.get("detect_conf", 0.0))
            item.rec_conf = float(result.get("rec_conf", 0.0))
            item.flags = 0
            if plate_no:
                item.flags |= ITEM_LABEL
            if draw_mode == "full":
                item.flags |= ITEM_CORNERS | ITEM_FULL
            if result.get("polygon"):
                item.flags |= ITEM_POLYGON
            item.plate_no = _utf8_field(plate_no, 32)
            # 保留C结构字段以维持ABI，但颜色不再提交给显示层。
            item.plate_color = _utf8_field("", 24)
            rgba = result.get("rgba", (255, 0, 0, 255))
            if len(rgba) != 4:
                raise ValueError("overlay RGBA color must contain four values")
            item.red, item.green, item.blue, item.alpha = (
                max(0, min(255, int(value))) for value in rgba
            )
        if source_timestamp_ns or ready_timestamp_ns:
            self._submit_items(
                array, len(selected), ttl_ms,
                source_timestamp_ns, ready_timestamp_ns,
            )
        else:
            self._submit_items(array, len(selected), ttl_ms)

    def submit_detections(self, detections, class_names=None, ttl_ms=500,
                          draw_mode="fast", colors=None,
                          source_timestamp_ns=0, ready_timestamp_ns=0):
        """提交通用彩色检测框，同时保持原车牌Overlay API不变。"""
        selected = list(detections or [])[:MAX_OVERLAY_ITEMS]
        array = (AtomicOverlayItem * max(1, len(selected)))()
        names = list(class_names or [])
        palette = colors or {}
        for index, detection in enumerate(selected):
            item = array[index]
            bbox = detection.get("bbox", (0, 0, 0, 0))
            item.x1, item.y1, item.x2, item.y2 = (
                int(value) for value in bbox
            )
            class_id = int(detection.get("class_id", -1))
            label = detection.get("label")
            if label is None and 0 <= class_id < len(names):
                label = names[class_id]
            item.label = _utf8_field(label or "", 64)
            item.score = float(detection.get("score", 0.0))
            item.flags = ITEM_GENERIC
            if draw_mode == "full" and label:
                item.flags |= ITEM_LABEL | ITEM_FULL
            rgba = detection.get("rgba", palette.get(class_id, (255, 0, 0, 255)))
            if len(rgba) != 4:
                raise ValueError("overlay RGBA color must contain four values")
            item.red, item.green, item.blue, item.alpha = (
                max(0, min(255, int(value))) for value in rgba
            )
        if source_timestamp_ns or ready_timestamp_ns:
            self._submit_items(
                array, len(selected), ttl_ms,
                source_timestamp_ns, ready_timestamp_ns,
            )
        else:
            self._submit_items(array, len(selected), ttl_ms)

    def _submit_items(self, array, count, ttl_ms, source_timestamp_ns=0,
                      ready_timestamp_ns=0):
        """提交items任务，不阻塞独立显示流程。"""
        err = ctypes.create_string_buffer(512)
        pointer = array if count else None
        rc = self._lib.rkdrm_atomic_submit_overlay(
            self._handle, pointer, int(count), int(ttl_ms),
            int(source_timestamp_ns), int(ready_timestamp_ns), err, len(err)
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))

    def show(self, frame):
        """提交一帧到显示后端，并保持原有Page Flip和缓冲复用语义。"""
        retired = ctypes.c_uint64()
        err = ctypes.create_string_buffer(512)
        rc = self._lib.rkdrm_atomic_show_nv12(
            self._handle,
            int(frame.dma_fd),
            int(frame.token),
            int(frame.width),
            int(frame.height),
            int(frame.hor_stride),
            int(frame.ver_stride),
            ctypes.byref(retired),
            err,
            len(err),
        )
        if rc != 0:
            raise RuntimeError(
                err.value.decode("utf-8", "replace") or "Atomic NV12显示失败"
            )
        return int(retired.value)

    def get_stats(self):
        """返回当前累计统计快照，不重置正在采集的计数器。"""
        stats = AtomicDisplayStats()
        err = ctypes.create_string_buffer(256)
        rc = self._lib.rkdrm_atomic_get_stats(
            self._handle, ctypes.byref(stats), err, len(err)
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return stats

    def overlay_percentile(self, quantile):
        """执行AtomicNv12Display的“overlay percentile”职责，保持原有输入输出约定。"""
        value = self._lib.rkdrm_atomic_overlay_percentile(
            self._handle, float(quantile)
        )
        if value < 0:
            raise RuntimeError("overlay percentile is not available")
        return float(value)

    def close(self):
        """按原有依赖顺序停止工作并释放本对象持有的硬件或系统资源。"""
        if self._handle:
            self._lib.rkdrm_atomic_close(self._handle)
            self._handle = None
