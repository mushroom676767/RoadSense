"""解析并汇总 ``fpga_ai`` schema v2 性能日志。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Optional


_NUMBER = r"[\d.]+"
_CAPTURE_RE = re.compile(
    rf"\[采集\s+(?P<frame>\d+)\]\s+"
    rf"采集帧率:(?P<fps>{_NUMBER})\s+FPS\s+\|\s+"
    rf"V4L2:(?P<v4l2>{_NUMBER})ms\s+\|\s+"
    rf"mpp:(?P<mpp>{_NUMBER})ms\s+\|\s+"
    rf"V4L2丢帧:(?P<v4l2_drops>\d+)\s+\|\s+"
    rf"MPP丢帧:(?P<mpp_drops>\d+)\s+\|\s+"
    rf"输出丢帧:(?P<output_drops>\d+)\s+\|\s+"
    rf"队列丢帧:(?P<queue_drops>\d+)"
)

_DISPLAY_FIELDS = (
    ("decode_ms", "DQBUF→解码", "ms"),
    ("detect_queue_ms", "检测排队", "ms"),
    ("detect_npu_wait_ms", "检测NPU等待", "ms"),
    ("detect_compute_ms", "检测计算", "ms"),
    ("detect_rga_ms", "检测RGA", "ms"),
    ("detect_npu_ms", "检测NPU", "ms"),
    ("detect_output_ms", "检测输出", "ms"),
    ("detect_decode_ms", "检测解码", "ms"),
    ("detect_nms_ms", "检测NMS", "ms"),
    ("recognition_queue_ms", "识别排队", "ms"),
    ("recognition_npu_wait_ms", "识别NPU等待", "ms"),
    ("recognition_compute_ms", "识别计算", "ms"),
    ("recognition_candidates", "识别候选", ""),
    ("result_dispatch_ms", "结果调度", "ms"),
    ("overlay_ms", "Overlay", "ms"),
    ("ready_display_ms", "就绪→显示", "ms"),
    ("e2e_ms", "端到端", "ms"),
    ("drm_errors", "DRM错误", ""),
)
_DISPLAY_PATTERN = [
    rf"\[调试\s+(?P<frame>\d+)\]\s+显示:(?P<fps>{_NUMBER})FPS"
]
for field, label, suffix in _DISPLAY_FIELDS:
    value = r"\d+" if field in {"recognition_candidates", "drm_errors"} else _NUMBER
    _DISPLAY_PATTERN.append(rf"\s+\|\s+{label}:(?P<{field}>{value}){suffix}")
_DISPLAY_RE = re.compile("".join(_DISPLAY_PATTERN))


@dataclass(frozen=True)
class CaptureSample:
    frame: int
    fps: float
    v4l2_ms: float
    mpp_ms: float
    v4l2_drops: int
    mpp_drops: int
    output_drops: int
    queue_drops: int


@dataclass(frozen=True)
class DisplaySample:
    frame: int
    fps: float
    decode_ms: float
    detect_queue_ms: float
    detect_npu_wait_ms: float
    detect_compute_ms: float
    detect_rga_ms: float
    detect_npu_ms: float
    detect_output_ms: float
    detect_decode_ms: float
    detect_nms_ms: float
    recognition_queue_ms: float
    recognition_npu_wait_ms: float
    recognition_compute_ms: float
    recognition_candidates: int
    result_dispatch_ms: float
    overlay_ms: float
    ready_display_ms: float
    e2e_ms: float
    drm_errors: int


def parse_capture_sample(line: str) -> Optional[CaptureSample]:
    """解析当前Native采集日志。"""
    match = _CAPTURE_RE.search(line)
    if not match:
        return None
    return CaptureSample(
        frame=int(match["frame"]),
        fps=float(match["fps"]),
        v4l2_ms=float(match["v4l2"]),
        mpp_ms=float(match["mpp"]),
        v4l2_drops=int(match["v4l2_drops"]),
        mpp_drops=int(match["mpp_drops"]),
        output_drops=int(match["output_drops"]),
        queue_drops=int(match["queue_drops"]),
    )


def parse_display_sample(line: str) -> Optional[DisplaySample]:
    """解析当前AI显示与推理阶段日志。"""
    match = _DISPLAY_RE.search(line)
    if not match:
        return None
    values = {
        field: float(match[field])
        for field, _label, _suffix in _DISPLAY_FIELDS
    }
    values["recognition_candidates"] = int(match["recognition_candidates"])
    values["drm_errors"] = int(match["drm_errors"])
    return DisplaySample(
        frame=int(match["frame"]), fps=float(match["fps"]), **values
    )


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """使用线性插值计算百分位。"""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if percentile_value <= 0:
        return ordered[0]
    if percentile_value >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]):
    """返回一组样本的平均值和验收所需百分位。"""
    samples = [float(value) for value in values]
    return {
        "avg": mean(samples) if samples else 0.0,
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
    }


def summarize(capture_samples, display_samples):
    """生成benchmark schema v2流水线摘要。"""
    captures = list(capture_samples)
    displays = list(display_samples)
    stage_sources = {
        "v4l2": (captures, "v4l2_ms"),
        "mpp_decode": (captures, "mpp_ms"),
        **{
            field[:-3]: (displays, field)
            for field, _label, suffix in _DISPLAY_FIELDS
            if suffix == "ms"
        },
    }
    return {
        "schema_version": 2,
        "sample_counts": {
            "capture": len(captures),
            "display": len(displays),
        },
        "fps": {
            "capture": {
                "avg": mean(item.fps for item in captures) if captures else 0.0,
                "min": min((item.fps for item in captures), default=0.0),
            },
            "display": {
                "avg": mean(item.fps for item in displays) if displays else 0.0,
                "min": min((item.fps for item in displays), default=0.0),
            },
        },
        "stages_ms": {
            name: distribution(getattr(item, attr) for item in samples)
            for name, (samples, attr) in stage_sources.items()
        },
        "drops": {
            "v4l2": sum(item.v4l2_drops for item in captures),
            "mpp": sum(item.mpp_drops for item in captures),
            "output": sum(item.output_drops for item in captures),
            "queue": sum(item.queue_drops for item in captures),
            "drm": max((item.drm_errors for item in displays), default=0),
        },
        "samples": {
            "capture": [asdict(item) for item in captures],
            "display": [asdict(item) for item in displays],
        },
    }
