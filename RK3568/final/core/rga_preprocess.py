"""RGA 硬件前处理的 ctypes 封装 (librga_preprocess.so)。

替代车牌工程里 OpenCV 的 CPU 前处理:
- letterbox_rgb: 检测前处理 (等比 letterbox + BGR->RGB, packed NHWC)

RGA 不可用时 available() 返回 False, 调用方回退到 cv2 实现。
"""

import ctypes
from pathlib import Path

import numpy as np

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_RGA_LIB = _NATIVE_DIR / "librga_preprocess.so"

_lib = None
_load_error = None


def letterbox_geometry(src_w, src_h, dst_w, dst_h):
    """计算BGR与NV12路径共用的缩放比例和居中填充参数。"""
    src_w, src_h = int(src_w), int(src_h)
    dst_w, dst_h = int(dst_w), int(dst_h)
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        raise ValueError("letterbox dimensions must be positive")
    ratio = min(dst_w / src_w, dst_h / src_h)
    resized_w = min(dst_w, max(1, int(src_w * ratio + 0.5)))
    resized_h = min(dst_h, max(1, int(src_h * ratio + 0.5)))
    return ratio, ((dst_w - resized_w) // 2, (dst_h - resized_h) // 2)


def _load():
    """实现模块内部的“load”辅助逻辑，不作为新的外部接口。"""
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    if not _RGA_LIB.exists():
        _load_error = f"librga_preprocess.so 未找到: {_RGA_LIB} (make -C core/native)"
        return None
    try:
        lib = ctypes.CDLL(str(_RGA_LIB))
    except OSError as exc:
        _load_error = f"加载 librga_preprocess.so 失败: {exc}"
        return None

    lib.rga_letterbox_rgb.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # 源BGR缓冲
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # 源宽、高与stride
        ctypes.POINTER(ctypes.c_uint8),  # 目标模型输入缓冲
        ctypes.c_int,                    # 目标模型输入缓冲_size
        ctypes.c_int,                    # 目标正方形边长
        ctypes.POINTER(ctypes.c_float),  # 输出缩放比例
        ctypes.POINTER(ctypes.c_int),    # 输出水平填充
        ctypes.POINTER(ctypes.c_int),    # 输出垂直填充
        ctypes.c_char_p, ctypes.c_int,   # 错误缓冲及其长度
    ]
    lib.rga_letterbox_rgb.restype = ctypes.c_int
    lib.rga_letterbox_rgb_size.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.rga_letterbox_rgb_size.restype = ctypes.c_int

    lib.rga_letterbox_nv12_fd.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.rga_letterbox_nv12_fd.restype = ctypes.c_int
    lib.rga_letterbox_nv12_fd_size.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.rga_letterbox_nv12_fd_size.restype = ctypes.c_int

    lib.rga_nv12_roi_bgr_fd.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.rga_nv12_roi_bgr_fd.restype = ctypes.c_int

    _lib = lib
    return _lib


def available():
    """RGA 前处理库是否可用。"""
    return _load() is not None


def load_error():
    """返回最近一次Native RGA加载错误，供旧诊断脚本使用。"""
    return _load_error


def letterbox_rgb(bgr, target):
    """检测前处理: BGR 全帧 -> 等比 letterbox(target×target) + BGR->RGB, packed。

    返回 (rgb_nhwc, ratio, (pad_w, pad_h))；rgb_nhwc 形状 (1, target, target, 3) uint8,
    可直接 rknn.inference(inputs=[rgb], data_format='nhwc')。
    失败抛 RuntimeError, 调用方负责回退 cv2。
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error or "RGA 前处理不可用")
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr 必须是 HxWx3")
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = bgr.shape[:2]
    stride = int(bgr.strides[0])

    dst = np.empty((1, target, target, 3), dtype=np.uint8)
    ratio = ctypes.c_float(0.0)
    pad_w = ctypes.c_int(0)
    pad_h = ctypes.c_int(0)
    err = ctypes.create_string_buffer(256)

    rc = lib.rga_letterbox_rgb(
        bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h, stride,
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.nbytes, int(target),
        ctypes.byref(ratio), ctypes.byref(pad_w), ctypes.byref(pad_h),
        err, len(err),
    )
    if rc != 1:
        raise RuntimeError(err.value.decode("utf-8", "replace") or "rga_letterbox_rgb 失败")
    return dst, float(ratio.value), (int(pad_w.value), int(pad_h.value))


def letterbox_rgb_size(bgr, dst_w, dst_h, out=None):
    """把BGR帧转换为任意尺寸的紧凑RGB NHWC letterbox输入。"""
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error or "RGA前处理不可用")
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr必须是HxWx3")
    dst_w, dst_h = int(dst_w), int(dst_h)
    if out is None:
        out = np.empty((1, dst_h, dst_w, 3), dtype=np.uint8)
    if out.shape != (1, dst_h, dst_w, 3) or out.dtype != np.uint8:
        raise ValueError("out尺寸或类型不匹配")
    h, w = bgr.shape[:2]
    ratio = ctypes.c_float()
    pad_w = ctypes.c_int()
    pad_h = ctypes.c_int()
    err = ctypes.create_string_buffer(256)
    rc = lib.rga_letterbox_rgb_size(
        bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h, int(bgr.strides[0]),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(out.nbytes), dst_w, dst_h,
        ctypes.byref(ratio), ctypes.byref(pad_w), ctypes.byref(pad_h),
        err, len(err),
    )
    if rc != 1:
        raise RuntimeError(
            err.value.decode("utf-8", "replace") or "RGA letterbox失败"
        )
    return out, float(ratio.value), (int(pad_w.value), int(pad_h.value))


def letterbox_nv12_fd(frame, target):
    """把NV12 DMA-BUF直接转换为紧凑RGB NHWC模型输入。"""
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error or "RGA前处理不可用")
    dst = np.empty((1, target, target, 3), dtype=np.uint8)
    ratio = ctypes.c_float()
    pad_w = ctypes.c_int()
    pad_h = ctypes.c_int()
    err = ctypes.create_string_buffer(256)
    rc = lib.rga_letterbox_nv12_fd(
        int(frame.dma_fd),
        int(frame.width), int(frame.height),
        int(frame.hor_stride), int(frame.ver_stride),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(dst.nbytes), int(target),
        ctypes.byref(ratio), ctypes.byref(pad_w), ctypes.byref(pad_h),
        err, len(err),
    )
    if rc != 1:
        raise RuntimeError(
            err.value.decode("utf-8", "replace") or "NV12 RGA letterbox失败"
        )
    return dst, float(ratio.value), (int(pad_w.value), int(pad_h.value))


def letterbox_nv12_fd_size(frame, dst_w, dst_h, out=None):
    """把NV12 DMA-BUF转换为任意尺寸的紧凑RGB NHWC letterbox输入。"""
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error or "RGA前处理不可用")
    dst_w, dst_h = int(dst_w), int(dst_h)
    if out is None:
        out = np.empty((1, dst_h, dst_w, 3), dtype=np.uint8)
    if out.shape != (1, dst_h, dst_w, 3) or out.dtype != np.uint8:
        raise ValueError("out尺寸或类型不匹配")
    ratio = ctypes.c_float()
    pad_w = ctypes.c_int()
    pad_h = ctypes.c_int()
    err = ctypes.create_string_buffer(256)
    rc = lib.rga_letterbox_nv12_fd_size(
        int(frame.dma_fd), int(frame.width), int(frame.height),
        int(frame.hor_stride), int(frame.ver_stride),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(out.nbytes), dst_w, dst_h,
        ctypes.byref(ratio), ctypes.byref(pad_w), ctypes.byref(pad_h),
        err, len(err),
    )
    if rc != 1:
        raise RuntimeError(
            err.value.decode("utf-8", "replace") or "NV12 RGA letterbox失败"
        )
    return out, float(ratio.value), (int(pad_w.value), int(pad_h.value))


def aligned_nv12_roi_geometry(frame_width, frame_height, x, y, width, height):
    """返回裁剪后的NV12 ROI及满足RGA约束的目标stride。"""
    frame_width, frame_height = int(frame_width), int(frame_height)
    x, y, width, height = int(x), int(y), int(width), int(height)
    if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0:
        raise ValueError("NV12 ROI dimensions must be positive")

    right = x + width
    bottom = y + height
    x1 = min(frame_width, max(0, x)) & ~1
    y1 = min(frame_height, max(0, y)) & ~1
    x2 = min(frame_width, max(0, right))
    y2 = min(frame_height, max(0, bottom))
    x2 = min(frame_width & ~1, (x2 + 1) & ~1)
    y2 = min(frame_height & ~1, (y2 + 1) & ~1)
    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("NV12 ROI is outside the source frame")

    # RGB888要求水平stride按字节对齐；这里采用保守的
    # 16像素和16行对齐，防止RGA写入NumPy堆元数据。
    dst_stride = (roi_width + 15) & ~15
    dst_vstride = (roi_height + 15) & ~15
    return x1, y1, roi_width, roi_height, dst_stride, dst_vstride


def nv12_roi_bgr(frame, x, y, width, height):
    """从NV12 DMA-BUF提取对齐且紧凑的BGR ROI。"""
    geometry = aligned_nv12_roi_geometry(
        frame.width, frame.height, x, y, width, height
    )
    roi_x, roi_y, roi_width, roi_height, dst_stride, dst_vstride = geometry
    storage = np.zeros((dst_vstride, dst_stride, 3), dtype=np.uint8)
    out_x = ctypes.c_int()
    out_y = ctypes.c_int()
    out_w = ctypes.c_int()
    out_h = ctypes.c_int()
    err = ctypes.create_string_buffer(256)
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error or "RGA前处理不可用")
    rc = lib.rga_nv12_roi_bgr_fd(
        int(frame.dma_fd),
        int(frame.width), int(frame.height),
        int(frame.hor_stride), int(frame.ver_stride),
        roi_x, roi_y, roi_width, roi_height,
        storage.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(storage.nbytes), dst_stride, dst_vstride,
        ctypes.byref(out_x), ctypes.byref(out_y),
        ctypes.byref(out_w), ctypes.byref(out_h),
        err, len(err),
    )
    if rc != 1:
        raise RuntimeError(
            err.value.decode("utf-8", "replace") or "NV12 RGA ROI失败"
        )
    if (
        out_x.value != roi_x or out_y.value != roi_y
        or out_w.value != roi_width or out_h.value != roi_height
    ):
        raise RuntimeError("RGA returned unexpected NV12 ROI geometry")
    # RGA分配包含填充行列；返回独立所有权的
    # 紧凑数组，避免OpenCV或RKNN继续持有DMA目标缓冲。
    roi = np.ascontiguousarray(storage[:roi_height, :roi_width])
    return roi, (out_x.value, out_y.value)
