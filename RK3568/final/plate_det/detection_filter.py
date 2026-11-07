from dataclasses import dataclass

import numpy as np

from .result_utils import bbox_iou


@dataclass
class DetectionFilterConfig:
    mode: str = "balanced"
    conf_low: float = 0.18
    conf_high: float = 0.30
    low_confirm_frames: int = 2
    max_candidates: int = 4
    confirm_iou: float = 0.5
    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 10.0


@dataclass
class DetectionFilterStats:
    high_conf: int = 0
    low_conf_candidates: int = 0
    low_conf_confirmed: int = 0
    geometry_dropped: int = 0


@dataclass
class DetectionFilterResult:
    draw_candidates: list
    recognition_candidates: list
    stats: DetectionFilterStats


class DetectionCandidateFilter:
    """Filter weak plate detections while keeping strong detections immediate."""

    def __init__(self, config=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.config = config or DetectionFilterConfig()
        self._low_tracks = []

    def filter(self, detections):
        """执行DetectionCandidateFilter的“filter”职责，保持原有输入输出约定。"""
        stats = DetectionFilterStats()
        high_candidates = []
        low_candidates = []

        for det in detections:
            conf = float(det.get("conf", det.get("detect_conf", 0.0)))
            if conf >= self.config.conf_high:
                marked = _marked_candidate(det, "high")
                high_candidates.append(marked)
                stats.high_conf += 1
                continue

            if self.config.mode == "off" or conf < self.config.conf_low:
                continue

            stats.low_conf_candidates += 1
            if not _valid_plate_geometry(det, self.config):
                stats.geometry_dropped += 1
                continue
            low_candidates.append(_marked_candidate(det, "low"))

        confirmed_low = self._confirm_low_candidates(low_candidates, stats)
        draw_candidates = _limit_candidates(high_candidates, confirmed_low, self.config.max_candidates)
        return DetectionFilterResult(
            draw_candidates=draw_candidates,
            recognition_candidates=list(draw_candidates),
            stats=stats,
        )

    def _confirm_low_candidates(self, low_candidates, stats):
        """实现模块内部的“confirm low candidates”辅助逻辑，不作为新的外部接口。"""
        if not low_candidates:
            self._low_tracks = []
            return []

        new_tracks = []
        confirmed = []
        for candidate in sorted(low_candidates, key=_candidate_conf, reverse=True):
            previous = self._best_previous_track(candidate)
            hits = 1 if previous is None else previous["hits"] + 1
            new_tracks.append({
                "bbox": np.asarray(candidate["bbox"], dtype=np.float32).copy(),
                "hits": hits,
            })
            if hits >= self.config.low_confirm_frames:
                confirmed.append(candidate)
                stats.low_conf_confirmed += 1

        self._low_tracks = new_tracks
        return confirmed

    def _best_previous_track(self, candidate):
        """实现模块内部的“best previous track”辅助逻辑，不作为新的外部接口。"""
        bbox = np.asarray(candidate["bbox"], dtype=np.float32)
        best = None
        best_iou = 0.0
        for track in self._low_tracks:
            iou = bbox_iou(bbox, track["bbox"])
            if iou > best_iou:
                best = track
                best_iou = iou
        if best is None or best_iou < self.config.confirm_iou:
            return None
        return best


def _candidate_conf(candidate):
    """实现模块内部的“candidate conf”辅助逻辑，不作为新的外部接口。"""
    return float(candidate.get("conf", candidate.get("detect_conf", 0.0)))


def _marked_candidate(det, level):
    """实现模块内部的“marked candidate”辅助逻辑，不作为新的外部接口。"""
    candidate = dict(det)
    candidate["det_conf_level"] = level
    return candidate


def _limit_candidates(high_candidates, low_candidates, max_candidates):
    """实现模块内部的“limit candidates”辅助逻辑，不作为新的外部接口。"""
    high_sorted = sorted(high_candidates, key=_candidate_conf, reverse=True)
    low_sorted = sorted(low_candidates, key=_candidate_conf, reverse=True)
    return (high_sorted + low_sorted)[:max_candidates]


def _valid_plate_geometry(det, config):
    """实现模块内部的“valid plate geometry”辅助逻辑，不作为新的外部接口。"""
    try:
        bbox = np.asarray(det["bbox"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return False
    if bbox.size != 4 or not np.all(np.isfinite(bbox)):
        return False

    x1, y1, x2, y2 = bbox
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width <= 1.0 or height <= 1.0:
        return False

    aspect_ratio = width / height
    if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
        return False

    landmarks = det.get("landmarks")
    if landmarks is None:
        return True
    landmarks_arr = np.asarray(landmarks, dtype=np.float32)
    return landmarks_arr.shape == (4, 2) and bool(np.all(np.isfinite(landmarks_arr)))
