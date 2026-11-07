import ctypes
import time
from pathlib import Path

import cv2
import numpy as np


class RgaDrmDisplay:
    """DRM/KMS page-flip display accelerated by Rockchip RGA."""

    def __init__(self, device="/dev/dri/card0", show_fps=True, library_path=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.device = device
        self.show_fps = show_fps
        self._fps_count = 0
        self._fps_last_t = time.perf_counter()
        self._fps_value = 0.0
        self._closed = False

        native_dir = Path(__file__).resolve().parent / "native"
        lib_path = Path(library_path) if library_path else native_dir / "librkdrm_display.so"
        if not lib_path.exists():
            raise RuntimeError(
                f"DRM/RGA native helper not found: {lib_path}. "
                f"Build it with: make -C {native_dir}"
            )

        self._lib = ctypes.CDLL(str(lib_path))
        self._bind_api()

        err = ctypes.create_string_buffer(512)
        out_w = ctypes.c_int()
        out_h = ctypes.c_int()
        handle = self._lib.rkdrm_display_open(
            device.encode("utf-8"),
            ctypes.byref(out_w),
            ctypes.byref(out_h),
            err,
            ctypes.sizeof(err),
        )
        if not handle:
            raise RuntimeError(f"DRM/RGA 初始化失败: {err.value.decode('utf-8', errors='replace')}")

        self._handle = handle
        self.fb_w = int(out_w.value)
        self.fb_h = int(out_h.value)
        self.bpp = 32
        print(f"DRM/RGA Display: {self.fb_w}x{self.fb_h}, device={device}")

    def _bind_api(self):
        """实现模块内部的“bind api”辅助逻辑，不作为新的外部接口。"""
        self._lib.rkdrm_display_open.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.rkdrm_display_open.restype = ctypes.c_void_p

        self._lib.rkdrm_display_show.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.rkdrm_display_show.restype = ctypes.c_int

        self._lib.rkdrm_display_close.argtypes = [ctypes.c_void_p]
        self._lib.rkdrm_display_close.restype = None

    def _draw_fps(self, cv2_img):
        """绘制fps内容，保持现有颜色与坐标规则。"""
        if not self.show_fps:
            return
        now = time.perf_counter()
        self._fps_count += 1
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            # 真实显示帧率：统计窗口内实际 show() 次数 / 耗时
            self._fps_value = round(self._fps_count / elapsed, 1)
            self._fps_count = 0
            self._fps_last_t = now
        cv2.putText(
            cv2_img,
            f"FPS: {self._fps_value:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 255),  # 红色 (BGR)
            4,
            cv2.LINE_AA,
        )

    def show(self, cv2_img):
        """提交一帧到显示后端，并保持原有Page Flip和缓冲复用语义。"""
        if self._closed:
            raise RuntimeError("DRM/RGA display is already closed")
        if cv2_img is None or cv2_img.ndim != 3 or cv2_img.shape[2] != 3:
            raise ValueError("RgaDrmDisplay.show expects a BGR image with shape HxWx3")
        if cv2_img.dtype != np.uint8:
            raise ValueError("RgaDrmDisplay.show expects uint8 image data")

        t_start = time.perf_counter()
        self._draw_fps(cv2_img)
        img = np.ascontiguousarray(cv2_img)
        height, width = img.shape[:2]
        stride = int(img.strides[0])

        err = ctypes.create_string_buffer(512)
        rc = self._lib.rkdrm_display_show(
            self._handle,
            ctypes.c_void_p(img.ctypes.data),
            int(width),
            int(height),
            stride,
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(f"DRM/RGA 显示失败: {err.value.decode('utf-8', errors='replace')}")
        return (time.perf_counter() - t_start) * 1000

    def close(self):
        """按原有依赖顺序停止工作并释放本对象持有的硬件或系统资源。"""
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_handle", None):
            self._lib.rkdrm_display_close(self._handle)
            self._handle = None
