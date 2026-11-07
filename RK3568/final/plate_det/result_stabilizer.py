"""Output-only plate text hysteresis for live plate tracks."""

from __future__ import annotations

import copy
import math
import threading
import time

from plate_det.result_utils import bbox_iou, is_complete_plate_result


_GEOMETRY_KEYS = (
    "bbox",
    "landmarks",
    "landmark_visibility",
    "detect_conf",
    "plate_type",
    "det_conf_level",
    "roi_index",
    "source",
    "geometry_source",
    "recognition_source",
    "rgba",
    "polygon",
)


class PlateTextStabilizer:
    """Stabilize published text without caching or suppressing OCR inference."""

    def __init__(
        self,
        *,
        iou_threshold,
        first_confirmations=2,
        switch_confirmations=3,
        track_ttl_ms=1200,
        max_missed_updates=1,
        group_key=None,
        clock=time.monotonic,
    ):
        self.iou_threshold = float(iou_threshold)
        self.first_confirmations = int(first_confirmations)
        self.switch_confirmations = int(switch_confirmations)
        self.track_ttl_s = float(track_ttl_ms) / 1000.0
        self.max_missed_updates = int(max_missed_updates)
        self.group_key = group_key
        self.clock = clock
        self._tracks = []
        self._next_id = 1
        self._lock = threading.Lock()

    def clear(self):
        """Immediately discard all pending and stable tracks."""
        with self._lock:
            self._tracks.clear()

    def update(self, results, now=None):
        """Consume one OCR batch and return the currently publishable results."""
        current = self.clock() if now is None else float(now)
        incoming = [
            copy.deepcopy(item)
            for item in (results or ())
            if is_complete_plate_result(item) and item.get("bbox") is not None
        ]
        with self._lock:
            self._expire_locked(current)
            matches = self._match_locked(incoming)
            existing_track_count = len(self._tracks)
            matched_tracks = set()
            matched_incoming = set()
            for track_index, incoming_index in matches:
                matched_tracks.add(track_index)
                matched_incoming.add(incoming_index)
                self._update_track_locked(
                    self._tracks[track_index], incoming[incoming_index], current
                )
            if incoming:
                for track_index in range(existing_track_count):
                    if track_index not in matched_tracks:
                        self._tracks[track_index]["missed_updates"] += 1
            for incoming_index, result in enumerate(incoming):
                if incoming_index not in matched_incoming:
                    self._tracks.append(self._new_track(result, current))
            return self._current_locked()

    def current(self, now=None):
        """Return stable tracks, expiring targets that have not been updated."""
        current = self.clock() if now is None else float(now)
        with self._lock:
            self._expire_locked(current)
            return self._current_locked()

    def _group(self, result):
        if self.group_key is None:
            return None
        return result.get(self.group_key)

    def _match_locked(self, incoming):
        candidates = []
        for track_index, track in enumerate(self._tracks):
            for incoming_index, result in enumerate(incoming):
                if track["group"] != self._group(result):
                    continue
                iou = bbox_iou(track["bbox"], result["bbox"])
                same_text = self._track_has_text(track, result["plate_no"])
                if iou >= self.iou_threshold:
                    candidates.append(
                        (3, int(same_text), iou, track_index, incoming_index)
                    )
                    continue
                distance = self._normalized_center_distance(
                    track["bbox"], result["bbox"]
                )
                if distance is None:
                    continue
                if distance <= 0.75:
                    candidates.append(
                        (
                            2,
                            int(same_text),
                            -distance,
                            track_index,
                            incoming_index,
                        )
                    )
                elif same_text and distance <= 2.0:
                    candidates.append(
                        (1, 1, -distance, track_index, incoming_index)
                    )
        candidates.sort(reverse=True)
        used_tracks = set()
        used_incoming = set()
        matches = []
        for _tier, _same_text, _score, track_index, incoming_index in candidates:
            if track_index in used_tracks or incoming_index in used_incoming:
                continue
            used_tracks.add(track_index)
            used_incoming.add(incoming_index)
            matches.append((track_index, incoming_index))
        return matches

    @staticmethod
    def _track_has_text(track, text):
        text = str(text)
        return text in {track.get("stable_text"), track.get("pending_text")}

    @staticmethod
    def _normalized_center_distance(first_bbox, second_bbox):
        """Return motion distance in plate-diagonal units, rejecting size jumps."""
        try:
            ax1, ay1, ax2, ay2 = (float(value) for value in first_bbox[:4])
            bx1, by1, bx2, by2 = (float(value) for value in second_bbox[:4])
        except (TypeError, ValueError):
            return None
        aw, ah = ax2 - ax1, ay2 - ay1
        bw, bh = bx2 - bx1, by2 - by1
        if min(aw, ah, bw, bh) <= 0:
            return None
        width_ratio = max(aw, bw) / min(aw, bw)
        height_ratio = max(ah, bh) / min(ah, bh)
        if width_ratio > 2.5 or height_ratio > 2.5:
            return None
        acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        scale = max(math.hypot(aw, ah), math.hypot(bw, bh))
        return math.hypot(acx - bcx, acy - bcy) / scale

    def _new_track(self, result, now):
        track = {
            "id": self._next_id,
            "group": self._group(result),
            "bbox": copy.deepcopy(result["bbox"]),
            "last_seen": now,
            "missed_updates": 0,
            "stable_text": None,
            "stable_result": None,
            "pending_text": str(result["plate_no"]),
            "pending_count": 1,
            "pending_result": copy.deepcopy(result),
        }
        self._next_id += 1
        if self.first_confirmations <= 1:
            self._promote_pending(track)
        return track

    def _update_track_locked(self, track, result, now):
        text = str(result["plate_no"])
        track["bbox"] = copy.deepcopy(result["bbox"])
        track["last_seen"] = now
        track["missed_updates"] = 0

        if track["stable_text"] == text:
            track["stable_result"] = copy.deepcopy(result)
            track["pending_text"] = None
            track["pending_count"] = 0
            track["pending_result"] = None
            return

        if track["stable_result"] is not None:
            for key in _GEOMETRY_KEYS:
                if key in result:
                    track["stable_result"][key] = copy.deepcopy(result[key])

        if track["pending_text"] == text:
            track["pending_count"] += 1
        else:
            track["pending_text"] = text
            track["pending_count"] = 1
        track["pending_result"] = copy.deepcopy(result)

        required = (
            self.first_confirmations
            if track["stable_text"] is None
            else self.switch_confirmations
        )
        if track["pending_count"] >= required:
            self._promote_pending(track)

    @staticmethod
    def _promote_pending(track):
        track["stable_text"] = track["pending_text"]
        track["stable_result"] = track["pending_result"]
        track["pending_text"] = None
        track["pending_count"] = 0
        track["pending_result"] = None

    def _expire_locked(self, now):
        self._tracks = [
            track
            for track in self._tracks
            if now - track["last_seen"] <= self.track_ttl_s
        ]

    def _current_locked(self):
        return [
            copy.deepcopy(track["stable_result"])
            for track in sorted(self._tracks, key=lambda item: item["id"])
            if track["stable_result"] is not None
            and track["missed_updates"] <= self.max_missed_updates
        ]
