"""Canonical RK-side latency calculations for manager publishing."""

from collections import OrderedDict
import math
import time


EMPTY_LATENCIES = {
    "detection_latency_ms": None,
    "recognition_latency_ms": None,
    "total_latency_ms": None,
}


def frame_e2e_source_timestamp_ns(frame):
    """返回板卡处理起点；优先使用DQBUF完成时刻，并兼容旧Native帧对象。"""
    processing_ns = int(
        getattr(frame, "processing_timestamp_ns", 0) or 0
    )
    if processing_ns > 0:
        return processing_ns
    return int(getattr(frame, "timestamp_ns", 0) or 0)


def _nonnegative(value):
    """实现模块内部的“nonnegative”辅助逻辑，不作为新的外部接口。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _required_sum(*values):
    """实现模块内部的“required sum”辅助逻辑，不作为新的外部接口。"""
    numbers = [_nonnegative(value) for value in values]
    if any(value is None for value in numbers):
        return None
    return sum(numbers)


def ai_latency_values(result):
    """返回车牌AI检测与识别模型的纯NPU耗时。"""
    if not result:
        return None, None
    detection = _nonnegative(
        (result.get("native_detection_timing") or {}).get("npu_ms")
    )
    recognition = _nonnegative(result.get("recognition_npu_ms"))
    return detection, recognition


def serial_latency_values(result):
    """返回串口ROI检测与识别模型的纯NPU耗时。"""
    if not result:
        return None, None
    detection = _nonnegative(result.get("detection_npu_ms"))
    recognition = _nonnegative(result.get("recognition_npu_ms"))
    return detection, recognition


def people_latency_values(result):
    """返回行人推理排队、RGA准备、NPU后处理与端到端时延。"""
    if not result or not result.get("source_timestamp_ns"):
        return None
    return _required_sum(
        result.get("inference_queue_ms"),
        result.get("prepare_ms"),
        result.get("inference_compute_ms"),
    )


class DisplayedLatencyTracker:
    """Correlate stage timings with the overlay source actually page-flipped."""

    def __init__(self, capacity=64):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self._capacity = max(2, int(capacity))
        self._samples = OrderedDict()

    def record(self, source_timestamp_ns, ready_timestamp_ns,
               detection_latency_ms, recognition_latency_ms=None):
        """记录指定阶段的单调时钟时间戳，供后续一致性计算使用。"""
        source_ns = int(source_timestamp_ns or 0)
        ready_ns = int(ready_timestamp_ns or 0)
        if source_ns <= 0 or ready_ns <= 0:
            return False
        sample = {
            "ready_timestamp_ns": ready_ns,
            "detection_latency_ms": _nonnegative(detection_latency_ms),
            "recognition_latency_ms": _nonnegative(recognition_latency_ms),
            "algorithm_e2e_ms": max(
                0.0, (ready_ns - source_ns) / 1_000_000.0
            ),
        }
        self._samples.pop(source_ns, None)
        self._samples[source_ns] = sample
        while len(self._samples) > self._capacity:
            self._samples.popitem(last=False)
        return True

    def resolve(self, display_stats, ttl_ms, now_ns=None):
        """根据同一来源帧的时间戳计算各阶段与端到端时延。"""
        if display_stats is None:
            return dict(EMPTY_LATENCIES)
        source_ns = int(
            getattr(display_stats, "last_overlay_source_timestamp_ns", 0)
            or 0
        )
        sample = self._samples.get(source_ns)
        if sample is None:
            return dict(EMPTY_LATENCIES)
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        ttl_ns = max(0, int(ttl_ms)) * 1_000_000
        if (
            ttl_ns > 0
            and current_ns - sample["ready_timestamp_ns"] > ttl_ns
        ):
            return dict(EMPTY_LATENCIES)
        return {
            "detection_latency_ms": sample["detection_latency_ms"],
            "recognition_latency_ms": sample["recognition_latency_ms"],
            # total_latency_ms统一表示算法结果就绪时间，不再包含Overlay、
            # Atomic Commit、VBlank或Page Flip等待。
            "total_latency_ms": sample["algorithm_e2e_ms"],
        }


class DisplayedStageTracker:
    """保存逐帧阶段耗时，并只返回已经完成Page Flip的同源样本。"""

    def __init__(self, capacity=64):
        """创建固定容量的来源时间戳索引，避免调试日志混用不同帧。"""
        self._capacity = max(2, int(capacity))
        self._samples = OrderedDict()

    def record(self, source_timestamp_ns, ready_timestamp_ns, values):
        """记录一帧的阶段耗时快照；时间戳无效时忽略该样本。"""
        source_ns = int(source_timestamp_ns or 0)
        ready_ns = int(ready_timestamp_ns or 0)
        if source_ns <= 0 or ready_ns <= 0:
            return False
        sample = dict(values or {})
        sample["source_timestamp_ns"] = source_ns
        sample["ready_timestamp_ns"] = ready_ns
        sample["algorithm_e2e_ms"] = max(
            0.0, (ready_ns - source_ns) / 1_000_000.0
        )
        self._samples.pop(source_ns, None)
        self._samples[source_ns] = sample
        while len(self._samples) > self._capacity:
            self._samples.popitem(last=False)
        return True

    def resolve(self, display_stats, ttl_ms, now_ns=None):
        """返回当前已显示Overlay对应的阶段快照，未匹配或过期时返回None。"""
        if display_stats is None:
            return None
        source_ns = int(
            getattr(display_stats, "last_overlay_source_timestamp_ns", 0)
            or 0
        )
        sample = self._samples.get(source_ns)
        if sample is None:
            return None
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        ttl_ns = max(0, int(ttl_ms)) * 1_000_000
        if (
            ttl_ns > 0
            and current_ns - sample["ready_timestamp_ns"] > ttl_ns
        ):
            return None
        return dict(sample)
