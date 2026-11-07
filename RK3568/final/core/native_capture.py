"""V4L2 → MPP → NV12/BGR采集运行时。

模块分为四层，顺序与数据流一致：

1. ctypes结构和动态库绑定：必须与 ``rkhdmi_capture.c`` 的公开ABI逐字段一致；
2. ``NativeNv12Frame``：用不透明token管理MPP帧槽的retain/release；
3. ``NativeMppCapture``：同步封装一次读取、统计和关闭操作；
4. ``NativeMppHdmiReceiver``：后台线程持续采集，并以latest-wins或阻塞策略发布。

NV12模式不会把整帧复制到NumPy。队列、检测和DRM每增加一个持有者都必须
retain，完成后必须release；关闭顺序始终是停止接收、清空队列引用、最后关闭
Native capture，避免DMA-BUF仍被显示或RGA使用时提前回收。
"""

import ctypes
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .hdmi_io import HDMI_DEVICE, HDMI_WIDTH, HDMI_HEIGHT, HDMI_FPS, HDMI_FOURCC


NATIVE_DIR = Path(__file__).resolve().parent / "native"
NATIVE_CAPTURE_LIB = NATIVE_DIR / "librkhdmi_capture.so"
_NATIVE_CAPTURE_LIB_CACHE = {}

# native-mpp 硬解依赖官方新版 MPP (系统自带的是 2021 旧版, ABI 可能不符)。
# 默认把这份新库作为 HDMI 采集的运行时库: 在加载 librkhdmi_capture.so 之前
# 用绝对路径 + RTLD_GLOBAL 预加载它, 其 MPP 符号即由新版满足, 三个入口无需
# 再手动设置 LD_LIBRARY_PATH。可用环境变量 RKHDMI_MPP_LIB 覆盖该路径。
import os

_DEFAULT_MPP_LIB = "/home/cat/mpp-official/build/linux/aarch64/mpp/librockchip_mpp.so"
_mpp_preloaded = False


def _preload_official_mpp():
    """预加载板卡官方MPP动态库及其依赖，确保后续ctypes加载使用同一套运行时。"""
    global _mpp_preloaded
    if _mpp_preloaded:
        return
    lib = os.environ.get("RKHDMI_MPP_LIB", _DEFAULT_MPP_LIB)
    if lib and Path(lib).exists():
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            _mpp_preloaded = True
        except OSError:
            # 预加载失败则回退到系统默认解析 (LD_LIBRARY_PATH / ldconfig)
            pass


class NativeCaptureStats(ctypes.Structure):
    _fields_ = [
        ("frames", ctypes.c_uint64),
        ("captured", ctypes.c_uint64),
        ("drops", ctypes.c_uint64),
        ("errors", ctypes.c_uint64),
        ("timeouts", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("v4l2_total_ms", ctypes.c_double),
        ("v4l2_max_ms", ctypes.c_double),
        ("mpp_total_ms", ctypes.c_double),
        ("mpp_max_ms", ctypes.c_double),
        ("rga_total_ms", ctypes.c_double),
        ("rga_max_ms", ctypes.c_double),
        ("decoded", ctypes.c_uint64),
        ("output_frames", ctypes.c_uint64),
        ("compressed_drops", ctypes.c_uint64),
        ("result_drops", ctypes.c_uint64),
        ("decode_errors", ctypes.c_uint64),
        ("result_total_ms", ctypes.c_double),
        ("result_max_ms", ctypes.c_double),
        ("output_total_ms", ctypes.c_double),
        ("output_max_ms", ctypes.c_double),
    ]


class NativeNv12Info(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_uint64),
        ("dma_fd", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("hor_stride", ctypes.c_int32),
        ("ver_stride", ctypes.c_int32),
        ("format", ctypes.c_uint32),
        ("sequence", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
        ("decoded_timestamp_ns", ctypes.c_uint64),
    ]


class NativeNv12Frame:
    """一个由token标识的MPP NV12 DMA-BUF所有权引用。

    ``retain``返回新的Python所有权对象；每个对象只能``release``一次。对象析构
    不负责隐式释放，以便资源错误能在调用链中暴露，而不是被GC时机掩盖。
    """

    __slots__ = (
        "_capture", "_released", "token", "dma_fd", "width", "height",
        "hor_stride", "ver_stride", "format", "sequence", "timestamp_ns",
        "processing_timestamp_ns", "decoded_timestamp_ns",
    )

    def __init__(self, capture, info, processing_timestamp_ns=None):
        """初始化帧时间戳与所有权；处理起点缺失时兼容回退到采集时间戳。"""
        self._capture = capture
        self._released = False
        for name, _ctype in NativeNv12Info._fields_:
            setattr(self, name, int(getattr(info, name)))
        self.processing_timestamp_ns = int(
            processing_timestamp_ns or self.timestamp_ns
        )

    def retain(self):
        """增加当前帧的Native引用计数，供另一个异步使用者安全持有。"""
        if self._released:
            raise RuntimeError("cannot retain a released NV12 frame")
        self._capture._retain_native_frame(self.token)
        info = NativeNv12Info()
        for name, _ctype in NativeNv12Info._fields_:
            setattr(info, name, getattr(self, name))
        return NativeNv12Frame(
            self._capture, info, self.processing_timestamp_ns
        )

    def release(self):
        """释放当前对象持有的Native帧引用，确保每份所有权只释放一次。"""
        if self._released:
            return
        self._capture._release_native_frame(self.token)
        self._released = True

    def __enter__(self):
        """进入上下文并返回当前资源对象，资源释放由对应的退出方法负责。"""
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        """退出上下文并按既定顺序释放资源，异常继续交给调用方处理。"""
        self.release()


def _clone_native_capture_stats(stats: NativeCaptureStats) -> NativeCaptureStats:
    """复制Native采集统计结构，避免后台线程更新调用方正在读取的内存。"""
    snapshot = NativeCaptureStats()
    for name, _ctype in NativeCaptureStats._fields_:
        setattr(snapshot, name, getattr(stats, name))
    return snapshot


def _native_drop_count_from_stats(stats) -> int:
    """汇总Native采集链路中的丢帧计数，供应用层统一展示。"""
    if stats is None:
        return 0
    drops = int(getattr(stats, "drops", 0))
    if drops > 0:
        return drops
    return int(getattr(stats, "compressed_drops", 0)) + int(getattr(stats, "result_drops", 0))


def _diff_native_capture_stats(current, previous):
    """计算两次Native统计快照的差值，用于一次运行周期的增量报告。"""
    if previous is None:
        return _clone_native_capture_stats(current)

    delta = NativeCaptureStats()
    for name, _ctype in NativeCaptureStats._fields_:
        current_value = getattr(current, name)
        previous_value = getattr(previous, name, 0)
        if name.endswith("_max_ms"):
            setattr(delta, name, current_value)
        else:
            setattr(delta, name, current_value - previous_value)
    return delta


def load_native_capture(path=None):
    """加载并缓存HDMI Native采集动态库，同时绑定稳定的ctypes ABI。"""
    lib_path = Path(path) if path is not None else NATIVE_CAPTURE_LIB
    cache_key = str(lib_path)
    cached = _NATIVE_CAPTURE_LIB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not lib_path.exists():
        raise RuntimeError(
            f"native HDMI helper not found: {lib_path}. "
            "Build it with: make -C /home/cat/RK3568/final/core/native "
            "(expected librkhdmi_capture.so)"
        )

    _preload_official_mpp()
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise RuntimeError(
            f"failed to load native helper dependencies from {lib_path}: {exc}. "
            "Rebuild with: make -C /home/cat/RK3568/final/core/native."
        ) from exc
    lib.rkhdmi_capture_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.rkhdmi_capture_open.restype = ctypes.c_void_p
    if hasattr(lib, "rkhdmi_capture_open_decoder"):
        lib.rkhdmi_capture_open_decoder.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_open_decoder.restype = ctypes.c_void_p

    if hasattr(lib, "rkhdmi_capture_read_nv12"):
        lib.rkhdmi_capture_read_nv12.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NativeNv12Info),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_read_nv12.restype = ctypes.c_int
        lib.rkhdmi_capture_retain_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_retain_frame.restype = ctypes.c_int
        lib.rkhdmi_capture_release_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_release_frame.restype = ctypes.c_int
        if hasattr(lib, "rkhdmi_capture_get_frame_dequeued_timestamp_ns"):
            lib.rkhdmi_capture_get_frame_dequeued_timestamp_ns.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            lib.rkhdmi_capture_get_frame_dequeued_timestamp_ns.restype = (
                ctypes.c_int
            )
    if hasattr(lib, "rkhdmi_capture_enable_jpeg_tap"):
        lib.rkhdmi_capture_enable_jpeg_tap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_enable_jpeg_tap.restype = ctypes.c_int
        lib.rkhdmi_capture_read_jpeg.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_read_jpeg.restype = ctypes.c_int
    lib.rkhdmi_capture_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NativeCaptureStats),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.rkhdmi_capture_get_stats.restype = ctypes.c_int

    if hasattr(lib, "rkhdmi_capture_stop_and_get_stats"):
        lib.rkhdmi_capture_stop_and_get_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NativeCaptureStats),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.rkhdmi_capture_stop_and_get_stats.restype = ctypes.c_int

    lib.rkhdmi_capture_close.argtypes = [ctypes.c_void_p]
    lib.rkhdmi_capture_close.restype = None
    _NATIVE_CAPTURE_LIB_CACHE[cache_key] = lib
    return lib


class NativeMppCapture:
    """同步Native采集句柄，拥有C侧V4L2、MPP、RGA及帧池上下文。"""
    def __init__(self, device=HDMI_DEVICE, width=HDMI_WIDTH, height=HDMI_HEIGHT,
                 fps=HDMI_FPS, fourcc=HDMI_FOURCC, lib_path=None, decoder="mpp",
                 jpeg_tap_fps=0):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.lib = load_native_capture(lib_path)
        self.handle = None
        self._err = ctypes.create_string_buffer(512)
        self.decoder = str(decoder)
        self._jpeg_buffer = None

        if hasattr(self.lib, "rkhdmi_capture_open_decoder"):
            self.handle = self.lib.rkhdmi_capture_open_decoder(
                str(device).encode("utf-8"),
                int(width),
                int(height),
                int(fps),
                str(fourcc).upper().encode("ascii"),
                self.decoder.encode("ascii"),
                self._err,
                ctypes.sizeof(self._err),
            )
        elif self.decoder == "mpp":
            self.handle = self.lib.rkhdmi_capture_open(
                str(device).encode("utf-8"),
                int(width),
                int(height),
                int(fps),
                str(fourcc).upper().encode("ascii"),
                self._err,
                ctypes.sizeof(self._err),
            )
        else:
            raise RuntimeError("librkhdmi_capture.so does not support decoder selection")
        if not self.handle:
            raise RuntimeError(
                self._error_text(f"{self._backend_label()} HDMI capture open failed")
            )
        if jpeg_tap_fps:
            if not hasattr(self.lib, "rkhdmi_capture_enable_jpeg_tap"):
                self.close()
                raise RuntimeError(
                    "librkhdmi_capture.so does not support the MJPEG publish tap"
                )
            tap_error = ctypes.create_string_buffer(256)
            ret = self.lib.rkhdmi_capture_enable_jpeg_tap(
                self.handle,
                int(jpeg_tap_fps),
                tap_error,
                ctypes.sizeof(tap_error),
            )
            if ret != 0:
                text = tap_error.value.decode("utf-8", errors="replace")
                self.close()
                raise RuntimeError(text or "enable MJPEG publish tap failed")
            capacity = max(int(width) * int(height) * 2, 1024 * 1024)
            self._jpeg_buffer = (ctypes.c_uint8 * capacity)()

    def _error_text(self, fallback, err=None):
        """优先解码Native错误缓冲；缓冲为空时返回调用方提供的兜底说明。"""
        err = err if err is not None else self._err
        text = err.value.decode("utf-8", errors="replace")
        return text or fallback

    def _backend_label(self):
        """返回Native MPP采集后端的稳定标识。"""
        return "native-mpp"

    def read_nv12(self):
        """获取最新NV12 DMA-BUF帧及其时间戳；返回对象持有一份Native引用。"""
        if not hasattr(self.lib, "rkhdmi_capture_read_nv12"):
            raise RuntimeError("native HDMI helper does not expose NV12 frames")
        info = NativeNv12Info()
        err = ctypes.create_string_buffer(512)
        ret = self.lib.rkhdmi_capture_read_nv12(
            self.handle,
            ctypes.byref(info),
            err,
            ctypes.sizeof(err),
        )
        if ret < 0:
            raise RuntimeError(self._error_text("native NV12 read failed", err))
        if ret == 0:
            return None
        processing_timestamp_ns = self._frame_dequeued_timestamp_ns(
            info.token, info.timestamp_ns
        )
        return NativeNv12Frame(self, info, processing_timestamp_ns)

    def _frame_dequeued_timestamp_ns(self, token, fallback_timestamp_ns):
        """查询帧进入板卡内存的DQBUF完成时刻；旧动态库保持原时间戳语义。"""
        getter = getattr(
            self.lib, "rkhdmi_capture_get_frame_dequeued_timestamp_ns", None
        )
        if getter is None:
            return int(fallback_timestamp_ns or 0)
        timestamp_ns = ctypes.c_uint64()
        err = ctypes.create_string_buffer(256)
        ret = getter(
            self.handle,
            int(token),
            ctypes.byref(timestamp_ns),
            err,
            ctypes.sizeof(err),
        )
        if ret != 0:
            raise RuntimeError(
                self._error_text(
                    "native frame dequeue timestamp query failed", err
                )
            )
        return int(timestamp_ns.value or fallback_timestamp_ns or 0)

    def read_jpeg(self, last_sequence=0):
        """读取JPEG旁路中比指定序号更新的帧，不改变主NV12采集链路。"""
        if self._jpeg_buffer is None:
            raise RuntimeError("MJPEG publish tap is not enabled")
        out_bytes = ctypes.c_size_t()
        out_sequence = ctypes.c_uint32()
        out_timestamp_ns = ctypes.c_uint64()
        err = ctypes.create_string_buffer(256)
        ret = self.lib.rkhdmi_capture_read_jpeg(
            self.handle,
            int(last_sequence),
            self._jpeg_buffer,
            ctypes.sizeof(self._jpeg_buffer),
            ctypes.byref(out_bytes),
            ctypes.byref(out_sequence),
            ctypes.byref(out_timestamp_ns),
            err,
            ctypes.sizeof(err),
        )
        if ret < 0:
            raise RuntimeError(self._error_text("native JPEG tap read failed", err))
        if ret == 0:
            return None
        # ctypes array slicing first materializes a Python ``list[int]`` and
        # then converts that list to bytes.  A 1080p MJPEG frame therefore
        # held the GIL for roughly 19 ms on RK3568 and starved the capture,
        # inference and DRM threads.  string_at performs one native memcpy
        # directly into a bytes object.
        jpeg = ctypes.string_at(
            ctypes.addressof(self._jpeg_buffer), out_bytes.value
        )
        return jpeg, int(out_sequence.value), int(out_timestamp_ns.value)

    def _retain_native_frame(self, token):
        """通过帧token增加Native引用计数，防止DMA-BUF被提前复用。"""
        err = ctypes.create_string_buffer(256)
        ret = self.lib.rkhdmi_capture_retain_frame(
            self.handle, int(token), err, ctypes.sizeof(err)
        )
        if ret != 0:
            raise RuntimeError(self._error_text("native frame retain failed", err))

    def _release_native_frame(self, token):
        """通过帧token减少Native引用计数，使空闲槽可被解码器复用。"""
        if not self.handle:
            return
        err = ctypes.create_string_buffer(256)
        ret = self.lib.rkhdmi_capture_release_frame(
            self.handle, int(token), err, ctypes.sizeof(err)
        )
        if ret != 0:
            raise RuntimeError(self._error_text("native frame release failed", err))

    def get_stats(self, reset=False):
        """返回当前累计统计快照，不重置正在采集的计数器。"""
        stats = NativeCaptureStats()
        err = ctypes.create_string_buffer(512)
        ret = self.lib.rkhdmi_capture_get_stats(
            self.handle,
            ctypes.byref(stats),
            1 if reset else 0,
            err,
            ctypes.sizeof(err),
        )
        if ret < 0:
            raise RuntimeError(self._error_text("native HDMI capture stats failed", err))
        return stats

    def stop_and_get_stats(self):
        """停止Native采集线程并取得最终统计快照，随后仍由close释放底层句柄。"""
        if not hasattr(self.lib, "rkhdmi_capture_stop_and_get_stats"):
            raise RuntimeError("librkhdmi_capture.so does not support stop_and_get_stats")
        stats = NativeCaptureStats()
        err = ctypes.create_string_buffer(512)
        ret = self.lib.rkhdmi_capture_stop_and_get_stats(
            self.handle,
            ctypes.byref(stats),
            err,
            ctypes.sizeof(err),
        )
        if ret < 0:
            raise RuntimeError(self._error_text("native HDMI capture stop stats failed", err))
        return stats

    def close(self):
        """按原有依赖顺序停止工作并释放本对象持有的硬件或系统资源。"""
        if self.handle:
            self.lib.rkhdmi_capture_close(self.handle)
            self.handle = None

    def __enter__(self):
        """进入上下文并返回当前资源对象，资源释放由对应的退出方法负责。"""
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        """退出上下文并按既定顺序释放资源，异常继续交给调用方处理。"""
        self.close()


class NativeMppHdmiReceiver:
    """后台采集线程，并兼容旧Process风格的start/join/terminate接口。

    ``latest``策略用于实时显示：队列满时释放旧帧并放入新帧，绝不反压V4L2。
    ``blocking``策略用于必须逐帧消费的场景。线程只把帧所有权交给队列，不负责
    DRM扫描引用；消费者取走后必须在显示退休或处理完成时释放。
    """

    def __init__(self, q, device=HDMI_DEVICE, width=HDMI_WIDTH, height=HDMI_HEIGHT,
                 fps=HDMI_FPS, fourcc=HDMI_FOURCC, profile=False,
                 profile_interval=60, lib_path=None, decoder="mpp",
                 queue_policy="latest", jpeg_tap_fps=0):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        if queue_policy not in {"latest", "blocking"}:
            raise ValueError("queue_policy must be 'latest' or 'blocking'")
        self.q = q
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.profile = profile
        self.profile_interval = max(1, int(profile_interval))
        self.lib_path = lib_path
        self.decoder = "mpp"
        self.queue_policy = queue_policy
        self.jpeg_tap_fps = max(0, int(jpeg_tap_fps))
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._latest_native_stats = None
        self._capture_ref_lock = threading.Lock()
        self._capture_ref = None
        self._stats_refresh_lock = threading.Lock()
        self._stats_refresh_request = threading.Event()
        self._stats_refresh_done = threading.Event()
        self._stats_refresh_snapshot = None
        self._stats_refresh_error = None
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"hdmi-{self.backend_label}-receiver",
        )
        self.error = None

    @property
    def backend_label(self):
        """返回接收器使用的稳定采集后端标识。"""
        return "native-mpp"

    @property
    def backend_runtime_label(self):
        """返回固定NV12零复制采集链路的运行时后端标识。"""
        return f"{self.backend_label}-nv12"

    def start(self):
        """启动后台资源或工作线程；重复启动遵循原有幂等规则。"""
        self.thread.start()
        if not self.ready_event.wait(timeout=5.0):
            self.stop_event.set()
            raise RuntimeError(f"HDMI {self.backend_label} 采集启动超时")
        if self.error is not None:
            self.thread.join(timeout=1.0)
            raise RuntimeError(f"HDMI {self.backend_label} 采集启动失败: {self.error}")

    def is_alive(self):
        """返回后台采集线程当前是否仍在运行。"""
        return self.thread.is_alive()

    def get_latest_stats_snapshot(self):
        """在线程锁保护下复制最近一次Native统计快照。"""
        with self._stats_lock:
            if self._latest_native_stats is None:
                return None
            return _clone_native_capture_stats(self._latest_native_stats)

    def native_drop_count(self):
        """返回最近Native统计中的累计丢帧总数。"""
        return _native_drop_count_from_stats(self.get_latest_stats_snapshot())

    def read_latest_jpeg(self, last_sequence=0):
        """从当前采集句柄读取JPEG旁路帧，重连窗口内返回空结果。"""
        with self._capture_ref_lock:
            capture = self._capture_ref
            if capture is None:
                return None
            return capture.read_jpeg(last_sequence)

    def _store_latest_native_stats(self, stats):
        """复制并保存最新Native统计，隔离采集线程和监控线程的并发访问。"""
        snapshot = _clone_native_capture_stats(stats)
        with self._stats_lock:
            self._latest_native_stats = snapshot
        return snapshot

    def refresh_native_stats(self, timeout=1.0):
        """请求接收线程通过其采集句柄刷新Native统计。"""
        if not self.is_alive():
            return self.get_latest_stats_snapshot()

        with self._stats_refresh_lock:
            if not self.is_alive():
                return self.get_latest_stats_snapshot()

            self._stats_refresh_snapshot = None
            self._stats_refresh_error = None
            self._stats_refresh_done.clear()
            self._stats_refresh_request.set()

            if not self._stats_refresh_done.wait(timeout=max(0.0, float(timeout))):
                self._stats_refresh_request.clear()
                raise TimeoutError(f"native stats refresh timed out after {timeout}s")

            if self._stats_refresh_error is not None:
                raise RuntimeError(f"native stats refresh failed: {self._stats_refresh_error}") from self._stats_refresh_error

            if self._stats_refresh_snapshot is not None:
                return _clone_native_capture_stats(self._stats_refresh_snapshot)
            latest = self.get_latest_stats_snapshot()
            if latest is None:
                raise RuntimeError("native stats refresh returned no snapshot")
            return latest

    def stop_and_get_native_stats(self, timeout=3.0):
        """停止接收器并返回最终Native统计快照。"""
        return self.stop_and_snapshot_native_stats(timeout=timeout)

    def stop_and_snapshot_native_stats(self, timeout=3.0):
        """在测量边界冻结Native采集并返回最终统计。"""
        self.stop_event.set()
        snapshot = None
        with self._capture_ref_lock:
            capture = self._capture_ref
            if capture is not None:
                snapshot = self._store_latest_native_stats(capture.stop_and_get_stats())
            else:
                snapshot = self.get_latest_stats_snapshot()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError(f"native receiver did not stop within {timeout}s")
        if snapshot is None:
            raise RuntimeError("native receiver stopped without stats snapshot")
        return _clone_native_capture_stats(snapshot)

    @staticmethod
    def _is_expected_stop_error(exc):
        """判断异常是否由主动停止采集引起，此类异常无需触发重连。"""
        text = str(exc).lower()
        return "capture stopped" in text or "capture thread stopped" in text

    @staticmethod
    def _is_recoverable_capture_error(exc):
        """判断异常是否属于USB断开或设备暂时消失等可重连故障。"""
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "no such device",
                "no such file or directory",
                "device or resource busy",
                "input/output error",
                "failed to open",
                "open /dev/video",
                "poll v4l2 failed",
                "poll v4l2 device event",
                "v4l2 frame timeout",
                "vidioc_dqbuf failed",
                "vidioc_qbuf",
                "vidioc_querycap",
                "vidioc_s_fmt",
                "vidioc_s_parm",
                "vidioc_reqbufs",
                "vidioc_streamon",
                "native hdmi capture thread stopped",
            )
        )

    def _discard_queued_frames(self):
        """释放接收队列仍持有的全部帧，确保关闭采集句柄前引用计数归零。"""
        while True:
            try:
                frame = self.q.get_nowait()
            except queue.Empty:
                return
            if isinstance(frame, NativeNv12Frame):
                frame.release()

    def _service_native_stats_refresh(self, capture):
        """在采集循环中处理异步统计刷新请求，避免监控线程直接竞争Native句柄。"""
        if not self._stats_refresh_request.is_set():
            return

        snapshot = None
        error = None
        try:
            stats = capture.get_stats(reset=False)
            snapshot = self._store_latest_native_stats(stats)
        except Exception as exc:
            error = exc

        self._stats_refresh_snapshot = snapshot
        self._stats_refresh_error = error
        self._stats_refresh_request.clear()
        self._stats_refresh_done.set()

    def _finish_pending_native_stats_refresh(self):
        """结束尚未完成的统计刷新等待，防止停止阶段有线程永久阻塞。"""
        if not self._stats_refresh_request.is_set():
            return
        self._stats_refresh_snapshot = self.get_latest_stats_snapshot()
        self._stats_refresh_request.clear()
        self._stats_refresh_done.set()

    def terminate(self):
        """设置终止状态并唤醒可能阻塞的采集或消费线程。"""
        self.stop_event.set()

    def join(self, timeout=None):
        """等待后台线程退出，不改变线程内部的停止条件。"""
        self.thread.join(timeout=timeout)

    def _run(self):
        """执行对象的后台工作循环，直到收到停止请求或不可恢复错误。"""
        dropped_total = 0
        frame_count = 0
        profile_stats = None
        capture = None
        read_total_ms = 0.0
        read_max_ms = 0.0
        copy_total_ms = 0.0
        copy_max_ms = 0.0
        put_total_ms = 0.0
        put_max_ms = 0.0
        profile_last_t = time.perf_counter()
        started_once = False
        reconnect_attempt = 0

        try:
            while not self.stop_event.is_set():
                capture = None
                try:
                    capture = NativeMppCapture(
                        self.device,
                        self.width,
                        self.height,
                        self.fps,
                        self.fourcc,
                        self.lib_path,
                        decoder=self.decoder,
                        jpeg_tap_fps=self.jpeg_tap_fps,
                    )
                    with self._capture_ref_lock:
                        self._capture_ref = capture
                    if started_once:
                        print(
                            f"HDMI {self.backend_label} 采集已恢复: "
                            f"{self.device}",
                            flush=True,
                        )
                    else:
                        print(
                            f"HDMI {self.backend_label} 采集启动: {self.device} "
                            f"({self.width}x{self.height}) "
                            f"fourcc={str(self.fourcc).upper()} fps={self.fps}",
                            flush=True,
                        )
                        started_once = True
                        self.ready_event.set()
                    profile_stats = None
                    profile_last_t = time.perf_counter()

                    while not self.stop_event.is_set():
                        self._service_native_stats_refresh(capture)
                        t_read = time.perf_counter()
                        try:
                            frame = capture.read_nv12()
                            copy_ms = 0.0
                        except RuntimeError as exc:
                            if (
                                self.stop_event.is_set()
                                and self._is_expected_stop_error(exc)
                            ):
                                break
                            raise
                        read_ms = (time.perf_counter() - t_read) * 1000
                        if frame is None:
                            continue
                        reconnect_attempt = 0
                        if self.stop_event.is_set():
                            if isinstance(frame, NativeNv12Frame):
                                frame.release()
                            break

                        read_total_ms += read_ms
                        read_max_ms = max(read_max_ms, read_ms)
                        copy_total_ms += copy_ms
                        copy_max_ms = max(copy_max_ms, copy_ms)

                        t_put = time.perf_counter()
                        if self.queue_policy == "blocking":
                            dropped = 0
                            while not self.stop_event.is_set():
                                try:
                                    self.q.put(frame, timeout=0.05)
                                    break
                                except queue.Full:
                                    continue
                            else:
                                if isinstance(frame, NativeNv12Frame):
                                    frame.release()
                        else:
                            dropped = 0
                            try:
                                self.q.put_nowait(frame)
                            except queue.Full:
                                try:
                                    stale = self.q.get_nowait()
                                except queue.Empty:
                                    stale = None
                                if stale is not None:
                                    stale.release()
                                    dropped = 1
                                self.q.put_nowait(frame)
                        dropped_total += dropped
                        put_ms = (time.perf_counter() - t_put) * 1000
                        put_total_ms += put_ms
                        put_max_ms = max(put_max_ms, put_ms)
                        frame_count += 1

                        if frame_count % self.profile_interval == 0:
                            now = time.perf_counter()
                            elapsed = max(now - profile_last_t, 1e-6)
                            stats = capture.get_stats(reset=False)
                            self._store_latest_native_stats(stats)
                            if self.profile:
                                interval_stats = _diff_native_capture_stats(
                                    stats, profile_stats
                                )
                                profile_stats = _clone_native_capture_stats(stats)
                                frames = max(int(interval_stats.frames), 1)
                                captured = max(int(interval_stats.captured), 1)
                                mpp_drops = max(
                                    0,
                                    int(interval_stats.compressed_drops)
                                    - int(interval_stats.drops),
                                )
                                print(
                                    f"[采集 {frame_count}] "
                                    f"采集帧率:{self.profile_interval / elapsed:.1f} FPS | "
                                    f"V4L2:{interval_stats.v4l2_total_ms / captured:.1f}ms | "
                                    f"mpp:{interval_stats.mpp_total_ms / frames:.1f}ms | "
                                    f"V4L2丢帧:{int(interval_stats.drops)} | "
                                    f"MPP丢帧:{mpp_drops} | "
                                    f"输出丢帧:{int(interval_stats.result_drops)} | "
                                    f"队列丢帧:{dropped_total}",
                                    flush=True,
                                )
                                profile_last_t = now
                                read_total_ms = 0.0
                                read_max_ms = 0.0
                                copy_total_ms = 0.0
                                copy_max_ms = 0.0
                                put_total_ms = 0.0
                                put_max_ms = 0.0
                                dropped_total = 0
                    try:
                        final_stats = capture.get_stats(reset=False)
                    except Exception:
                        final_stats = None
                    if final_stats is not None:
                        self._store_latest_native_stats(final_stats)
                except Exception as exc:
                    if (
                        self.stop_event.is_set()
                        and self._is_expected_stop_error(exc)
                    ):
                        break
                    if not self._is_recoverable_capture_error(exc):
                        raise
                    reconnect_attempt += 1
                    if reconnect_attempt == 1 or reconnect_attempt % 10 == 0:
                        print(
                            f"HDMI {self.backend_label} 采集卡断开，"
                            f"正在自动重连(第{reconnect_attempt}次): {exc}",
                            flush=True,
                        )
                    self._discard_queued_frames()
                    self.stop_event.wait(
                        min(0.25 * reconnect_attempt, 1.0)
                    )
                finally:
                    with self._capture_ref_lock:
                        if self._capture_ref is capture:
                            self._capture_ref = None
                    if capture is not None:
                        capture.close()
        except Exception as exc:
            self.error = exc
            self.ready_event.set()
            print(f"HDMI {self.backend_label} 采集失败: {exc}", flush=True)
        finally:
            with self._capture_ref_lock:
                self._capture_ref = None
            self._finish_pending_native_stats_refresh()
            self.ready_event.set()
            print(f"HDMI {self.backend_label} 采集已退出", flush=True)
