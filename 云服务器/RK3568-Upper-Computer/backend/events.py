import csv
import io
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, request, send_file
from flask_jwt_extended import jwt_required

import config
from backend.auth import control_required
from backend.database import get_db


events_bp = Blueprint("events", __name__)
logger = logging.getLogger(__name__)


def _row_to_dict(row, include_metadata=False):
    item = {
        "id": row["id"],
        "mode": row["mode"],
        "event_type": row["event_type"],
        "occurred_at_ms": row["occurred_at_ms"],
        "label": row["label"],
        "color": row["color"],
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "snapshot_url": f"/api/events/{row['id']}/snapshot",
    }
    if include_metadata:
        try:
            item["metadata"] = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            item["metadata"] = {}
    return item


def _confidence(*values):
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            return number
    return 0.0


def save_event(
    mode,
    event_type,
    label,
    color,
    confidence,
    meta,
    jpeg_bytes,
    *,
    manager_received_at_ms=None,
    external_agent_id=None,
    external_event_id=None,
):
    occurred_at_ms = int(
        manager_received_at_ms
        if manager_received_at_ms is not None
        else time.time() * 1000
    )
    archive_meta = dict(meta)
    archive_meta["manager_received_at_ms"] = occurred_at_ms
    external_agent_id = str(external_agent_id or "")[:128] or None
    external_event_id = str(external_event_id or "")[:128] or None
    if config.DEPLOYMENT_ROLE == "agent" and external_event_id is None:
        external_agent_id = str(config.AGENT_ID or "rk3568-agent")[:128]
        external_event_id = uuid.uuid4().hex
    filename = (
        f"{occurred_at_ms}_{mode}_{event_type}_{uuid.uuid4().hex[:12]}.jpg"
    )
    snapshot_path = os.path.join(config.EVENT_SNAPSHOT_DIR, filename)
    temp_path = snapshot_path + ".tmp"
    snapshot_ready = False
    try:
        with open(temp_path, "wb") as handle:
            handle.write(jpeg_bytes)
        os.replace(temp_path, snapshot_path)
        snapshot_ready = True
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO events (
                        mode, event_type, occurred_at_ms, label, color,
                        confidence, metadata_json, snapshot_path,
                        external_agent_id, external_event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mode,
                        event_type,
                        occurred_at_ms,
                        label,
                        color,
                        _confidence(confidence),
                        json.dumps(
                            archive_meta,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        snapshot_path,
                        external_agent_id,
                        external_event_id,
                    ),
                )
                event_id = cursor.lastrowid
                if (
                    config.DEPLOYMENT_ROLE == "agent"
                    and external_event_id is not None
                ):
                    outbox_payload = {
                        "external_event_id": external_event_id,
                        "mode": mode,
                        "event_type": event_type,
                        "occurred_at_ms": occurred_at_ms,
                        "label": label,
                        "color": color,
                        "confidence": _confidence(confidence),
                        "metadata": archive_meta,
                    }
                    conn.execute(
                        """
                        INSERT INTO agent_event_outbox (
                            external_event_id, payload_json, snapshot_path,
                            created_at_ms
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            external_event_id,
                            json.dumps(
                                outbox_payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            snapshot_path,
                            occurred_at_ms,
                        ),
                    )
        except Exception:
            try:
                os.remove(snapshot_path)
            except FileNotFoundError:
                pass
            raise
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        if not snapshot_ready:
            try:
                os.remove(snapshot_path)
            except FileNotFoundError:
                pass
    try:
        cleanup_events()
    except Exception:
        logger.exception("event retention cleanup failed after event %s", event_id)
    return event_id


def cleanup_events():
    cutoff_ms = int(
        (time.time() - config.EVENT_RETENTION_DAYS * 86400) * 1000
    )
    with get_db() as conn:
        expired = conn.execute(
            """
            SELECT id, snapshot_path FROM events
            WHERE occurred_at_ms < ?
              AND snapshot_path NOT IN (
                  SELECT snapshot_path FROM agent_event_outbox
              )
            """,
            (cutoff_ms,),
        ).fetchall()
        overflow = conn.execute(
            """
            SELECT id, snapshot_path FROM events
            WHERE snapshot_path NOT IN (
                SELECT snapshot_path FROM agent_event_outbox
            )
            ORDER BY occurred_at_ms DESC
            LIMIT -1 OFFSET ?
            """,
            (config.EVENT_MAX_COUNT,),
        ).fetchall()
        remove = {row["id"]: row["snapshot_path"] for row in (*expired, *overflow)}
        if remove:
            placeholders = ",".join("?" for _ in remove)
            conn.execute(
                f"DELETE FROM events WHERE id IN ({placeholders})",
                tuple(remove),
            )
    for path in remove.values() if remove else ():
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def save_external_event(agent_id, payload, jpeg_bytes):
    """Persist one Agent event exactly once and return its cloud event id."""
    payload = payload if isinstance(payload, dict) else {}
    agent_id = str(agent_id or "")[:128]
    external_event_id = str(payload.get("external_event_id") or "")[:128]
    mode = str(payload.get("mode") or "")[:64]
    event_type = str(payload.get("event_type") or "")[:64]
    if not agent_id or not external_event_id:
        raise ValueError("agent_id and external_event_id are required")
    if mode not in config.MODES or not event_type:
        raise ValueError("invalid event mode or type")
    if (
        not isinstance(jpeg_bytes, (bytes, bytearray))
        or len(jpeg_bytes) < 4
        or not jpeg_bytes.startswith(b"\xff\xd8")
        or not jpeg_bytes.endswith(b"\xff\xd9")
    ):
        raise ValueError("invalid event JPEG")

    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id FROM events
            WHERE external_agent_id = ? AND external_event_id = ?
            """,
            (agent_id, external_event_id),
        ).fetchone()
    if existing:
        return int(existing["id"])

    metadata = payload.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["agent_id"] = agent_id
    metadata["external_event_id"] = external_event_id
    metadata["agent_occurred_at_ms"] = payload.get("occurred_at_ms")
    try:
        return save_event(
            mode,
            event_type,
            str(payload.get("label") or "")[:128],
            str(payload.get("color") or "")[:64],
            _confidence(payload.get("confidence")),
            metadata,
            bytes(jpeg_bytes),
            manager_received_at_ms=int(time.time() * 1000),
            external_agent_id=agent_id,
            external_event_id=external_event_id,
        )
    except sqlite3.IntegrityError:
        with get_db() as conn:
            existing = conn.execute(
                """
                SELECT id FROM events
                WHERE external_agent_id = ? AND external_event_id = ?
                """,
                (agent_id, external_event_id),
            ).fetchone()
        if existing:
            return int(existing["id"])
        raise


class EventDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self._plates = {}
        self._last_emitted = {}
        self._redlight_active = {}
        self._explicit_ids = {}
        self._last_save_error_log = 0.0

    def process(self, meta, jpeg_bytes, manager_received_at_ms=None):
        stream_id = str(meta.get("stream_id", ""))
        mode = str(meta.get("mode") or stream_id)
        now = time.monotonic()
        manager_received_at_ms = int(
            manager_received_at_ms
            if manager_received_at_ms is not None
            else time.time() * 1000
        )
        archive_meta = dict(meta)
        archive_meta["manager_received_at_ms"] = manager_received_at_ms
        emitted = []
        with self._lock:
            explicit, explicit_ids = self._process_explicit(
                archive_meta,
                jpeg_bytes,
                mode,
                stream_id,
                now,
                manager_received_at_ms,
            )
            emitted.extend(explicit_ids)
            if not explicit:
                if mode in {"plate_ai", "plate_serial"}:
                    emitted.extend(
                        self._process_plate(
                            archive_meta,
                            jpeg_bytes,
                            mode,
                            stream_id,
                            now,
                            manager_received_at_ms,
                        )
                    )
                elif mode == "people_redlight":
                    event_id = self._process_redlight(
                        archive_meta,
                        jpeg_bytes,
                        mode,
                        stream_id,
                        now,
                        manager_received_at_ms,
                    )
                    if event_id:
                        emitted.append(event_id)
            self._purge(now)
        return emitted

    def _save(self, *args, now, **kwargs):
        try:
            return save_event(*args, **kwargs)
        except Exception:
            if now - self._last_save_error_log >= 5:
                logger.exception("event archive write failed; it will be retried")
                self._last_save_error_log = now
            return None

    def clear_archive(self):
        """Atomically clear stored events and all in-memory dedupe state."""
        with self._lock:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT snapshot_path FROM events"
                ).fetchall()
                deleted = conn.execute("DELETE FROM events").rowcount
            self._plates.clear()
            self._last_emitted.clear()
            self._redlight_active.clear()
            self._explicit_ids.clear()
            self._last_save_error_log = 0.0
        return rows, deleted

    def _process_explicit(
        self,
        meta,
        jpeg_bytes,
        mode,
        stream_id,
        now,
        manager_received_at_ms,
    ):
        result = []
        has_valid_event = False
        for event in (meta.get("events") or [])[:16]:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip()[:64]
            if not event_type:
                continue
            has_valid_event = True
            dedupe_id = str(
                event.get("id")
                or f"{stream_id}:{event_type}:{event.get('label', '')}"
            )
            last = self._explicit_ids.get(dedupe_id, 0)
            if now - last < config.EVENT_COOLDOWN_SECONDS:
                continue
            event_id = self._save(
                mode,
                event_type,
                str(event.get("label") or "")[:128],
                str(event.get("color") or "")[:64],
                _confidence(event.get("confidence")),
                meta,
                jpeg_bytes,
                manager_received_at_ms=manager_received_at_ms,
                now=now,
            )
            if event_id:
                self._explicit_ids[dedupe_id] = now
                result.append(event_id)
        return has_valid_event, result

    def _process_plate(
        self,
        meta,
        jpeg_bytes,
        mode,
        stream_id,
        now,
        manager_received_at_ms,
    ):
        candidates = {}
        for overlay in (meta.get("overlays") or [])[:64]:
            if not isinstance(overlay, dict) or overlay.get("type") != "plate":
                continue
            label = str(overlay.get("label") or "").strip()
            if not label:
                continue
            previous = candidates.get(label)
            if previous is None or _confidence(
                overlay.get("score"),
                overlay.get("confidence"),
            ) > _confidence(
                previous.get("score"),
                previous.get("confidence"),
            ):
                candidates[label] = overlay
        active_keys = {(stream_id, label) for label in candidates}
        for key in list(self._plates):
            if key[0] == stream_id and key not in active_keys:
                # 已归档目标消失后立即结束本次事件，允许它再次出现时
                # 重新归档；尚在确认的弱目标则保留短暂命中计数，避免
                # 多车牌场景中一次OCR漏检就永远无法达到归档阈值。
                if self._plates[key].get("emitted"):
                    self._plates.pop(key, None)
        if not candidates:
            return []

        current_states = []
        for label, overlay in candidates.items():
            key = (stream_id, label)
            state = self._plates.get(key)
            if state and now - state["last"] < 2:
                state["count"] += 1
                state["last"] = now
            else:
                state = {
                    "count": 1,
                    "last": now,
                    "emitted": False,
                }
                self._plates[key] = state
            current_states.append((label, overlay, state))

        # 同一帧中的所有车牌属于同一张事件截图。只要当前场景中的
        # 任一文字达到归档稳定门槛，就把本帧全部有效文字合成一条档案；
        # 不再为每个号码重复保存同一张JPEG。
        if any(state["emitted"] for _label, _overlay, state in current_states):
            return []
        if not any(
            state["count"] >= config.PLATE_STABLE_FRAMES
            for _label, _overlay, state in current_states
        ):
            return []

        labels = [label for label, _overlay, _state in current_states]
        confidences = [
            _confidence(
                overlay.get("rec_score"),
                overlay.get("score"),
                overlay.get("confidence"),
            )
            for _label, overlay, _state in current_states
        ]
        archive_meta = dict(meta)
        archive_meta["plate_labels"] = labels
        archive_meta["plate_count"] = len(labels)
        event_id = self._save(
            mode,
            "plate",
            "、".join(labels)[:128],
            "",
            sum(confidences) / len(confidences),
            archive_meta,
            jpeg_bytes,
            manager_received_at_ms=manager_received_at_ms,
            now=now,
        )
        if not event_id:
            return []
        for _label, _overlay, state in current_states:
            state["emitted"] = True
        return [event_id]

    def _process_redlight(
        self,
        meta,
        jpeg_bytes,
        mode,
        stream_id,
        now,
        manager_received_at_ms,
    ):
        people = [
            overlay
            for overlay in (meta.get("overlays") or [])[:64]
            if isinstance(overlay, dict)
            and (
                overlay.get("type") in {"person", "redlight_violation"}
                or str(overlay.get("status", "")).lower()
                in {"normal", "violation", "redlight", "闯红灯"}
            )
        ]
        violations = [
            overlay
            for overlay in people
            if (
                overlay.get("type") == "redlight_violation"
                or str(overlay.get("status", "")).lower()
                in {"violation", "redlight", "闯红灯"}
            )
        ]
        active = bool(violations)
        was_active = self._redlight_active.get(stream_id, False)
        if not active:
            self._redlight_active[stream_id] = False
            return None
        key = (mode, "redlight_violation", stream_id)
        if (
            not was_active
            and now - self._last_emitted.get(key, 0)
            >= config.EVENT_COOLDOWN_SECONDS
        ):
            best = max(
                violations,
                key=lambda item: _confidence(
                    item.get("score"),
                    item.get("confidence"),
                ),
            )
            archive_meta = dict(meta)
            archive_meta["person_count"] = len(people)
            archive_meta["redlight_count"] = len(violations)
            event_id = self._save(
                mode,
                "redlight_violation",
                f"总行人数：{len(people)}，闯红灯人数：{len(violations)}",
                str(best.get("color") or "red")[:64],
                _confidence(best.get("score"), best.get("confidence")),
                archive_meta,
                jpeg_bytes,
                manager_received_at_ms=manager_received_at_ms,
                now=now,
            )
            if event_id:
                self._last_emitted[key] = now
                self._redlight_active[stream_id] = True
                return event_id
            self._redlight_active[stream_id] = False
            return None
        self._redlight_active[stream_id] = True
        return None

    def _purge(self, now):
        max_age = max(config.EVENT_COOLDOWN_SECONDS * 4, 60)
        self._plates = {
            key: value
            for key, value in self._plates.items()
            if now - value["last"] < 2
        }
        self._last_emitted = {
            key: value
            for key, value in self._last_emitted.items()
            if now - value < max_age
        }
        self._explicit_ids = {
            key: value
            for key, value in self._explicit_ids.items()
            if now - value < max_age
        }


event_detector = EventDetector()


@events_bp.get("/api/events")
@jwt_required()
def list_events():
    page = max(1, request.args.get("page", 1, type=int))
    page_size = min(100, max(1, request.args.get("page_size", 20, type=int)))
    clauses = []
    params = []
    for key, column in (("mode", "mode"), ("event_type", "event_type")):
        value = request.args.get(key, "").strip()
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    query = request.args.get("query", "").strip()
    if query:
        clauses.append("label LIKE ?")
        params.append(f"%{query}%")
    for key, op in (("start_ms", ">="), ("end_ms", "<=")):
        value = request.args.get(key, type=int)
        if value is not None:
            clauses.append(f"occurred_at_ms {op} ?")
            params.append(value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM events{where}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT * FROM events{where}
            ORDER BY occurred_at_ms DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, (page - 1) * page_size),
        ).fetchall()
    return jsonify(
        {
            "items": [_row_to_dict(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
        }
    )


@events_bp.post("/api/events/clear")
@events_bp.delete("/api/events")
@control_required()
def clear_events():
    rows, deleted = event_detector.clear_archive()

    deleted_snapshots = 0
    snapshot_delete_failures = 0
    for row in rows:
        try:
            os.remove(row["snapshot_path"])
            deleted_snapshots += 1
        except FileNotFoundError:
            pass
        except OSError:
            snapshot_delete_failures += 1

    return jsonify(
        {
            "message": "事件档案已清空",
            "deleted": deleted,
            "deleted_snapshots": deleted_snapshots,
            "snapshot_delete_failures": snapshot_delete_failures,
        }
    )


@events_bp.get("/api/events/<int:event_id>")
@jwt_required()
def get_event(event_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
    if row is None:
        return jsonify({"error": "事件不存在"}), 404
    return jsonify(_row_to_dict(row, include_metadata=True))


@events_bp.get("/api/events/<int:event_id>/snapshot")
@jwt_required()
def get_snapshot(event_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT snapshot_path FROM events WHERE id = ?", (event_id,)
        ).fetchone()
    if row is None or not os.path.isfile(row["snapshot_path"]):
        return jsonify({"error": "截图不存在"}), 404
    return send_file(row["snapshot_path"], mimetype="image/jpeg", conditional=True)


@events_bp.get("/api/events/export.csv")
@jwt_required()
def export_events():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, mode, event_type, occurred_at_ms, label, color, confidence
            FROM events ORDER BY occurred_at_ms DESC
            """
        ).fetchall()
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(
        ["ID", "模式", "事件类型", "发生时间(ms)", "标签", "颜色", "置信度"]
    )
    writer.writerows(tuple(row) for row in rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )
