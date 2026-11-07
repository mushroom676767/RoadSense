import hmac
import json
import re
import struct
import threading
import time

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import jwt_required

import config
from backend.events import event_detector
from backend.stream_hub import stream_hub


ingest_bp = Blueprint("ingest", __name__)
_STREAM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SESSION_ID_RE = re.compile(r"^[a-fA-F0-9]{32}$")
_latest = {}
_lock = threading.Lock()
_listeners = []
_listeners_lock = threading.Lock()


def get_latest(stream_id, max_age_seconds=None):
    with _lock:
        item = _latest.get(stream_id)
        if item is None:
            return None
        if (
            max_age_seconds is not None
            and time.monotonic() - item["received_monotonic"]
            > max(0.0, float(max_age_seconds))
        ):
            return None
        return item


def is_stream_online(stream_id):
    return get_latest(stream_id, config.STREAM_STALE_SECONDS) is not None


def register_ingest_listener(listener):
    with _listeners_lock:
        if listener not in _listeners:
            _listeners.append(listener)


def unregister_ingest_listener(listener):
    with _listeners_lock:
        if listener in _listeners:
            _listeners.remove(listener)


def _notify_ingest_listeners(stream_id, meta, jpeg_bytes, event_ids):
    with _listeners_lock:
        listeners = list(_listeners)
    for listener in listeners:
        try:
            listener(stream_id, meta, jpeg_bytes, list(event_ids))
        except Exception:
            current_app.logger.exception(
                "ingest listener failed for stream %s",
                stream_id,
            )


def list_streams():
    if config.DEPLOYMENT_ROLE == "cloud":
        from backend.agent_state import agent_registry

        return agent_registry.streams()
    now = time.monotonic()
    with _lock:
        return {
            stream_id: {
                "frame_id": item["meta"].get("frame_id"),
                "session_id": item["meta"].get("session_id"),
                "mode": item["meta"].get("mode"),
                "age_ms": round((now - item["received_monotonic"]) * 1000, 1),
                "online": now - item["received_monotonic"]
                <= config.STREAM_STALE_SECONDS,
                "metrics": item["meta"].get("metrics", {}),
                "subscribers": stream_hub.subscriber_count(stream_id),
            }
            for stream_id, item in _latest.items()
        }


def _error(message, status=400):
    return jsonify({"error": message}), status


def _validate_meta(meta):
    if not isinstance(meta, dict):
        return "JSON根节点必须是对象"
    if meta.get("version") != 1:
        return "不支持的协议版本"
    stream_id = str(meta.get("stream_id", ""))
    if not _STREAM_ID_RE.fullmatch(stream_id):
        return "stream_id格式错误"
    if stream_id not in config.MODES:
        return "未知stream_id"
    mode = str(meta.get("mode", stream_id))
    if mode != stream_id:
        return "mode必须与stream_id一致"
    session_id = meta.get("session_id")
    if session_id is not None and not _SESSION_ID_RE.fullmatch(
        str(session_id)
    ):
        return "session_id格式错误"
    overlays = meta.get("overlays", [])
    if not isinstance(overlays, list) or len(overlays) > 64:
        return "overlays必须是最多64项的数组"
    if not isinstance(meta.get("metrics", {}), dict):
        return "metrics必须是对象"
    if not isinstance(meta.get("events", []), list):
        return "events必须是数组"
    sent_at_ms = meta.get("sent_at_ms")
    if sent_at_ms is not None:
        if not isinstance(sent_at_ms, (int, float)):
            return "sent_at_ms必须是数字"
        if sent_at_ms > time.time() * 1000 + 300_000:
            return "sent_at_ms超出允许范围"
    return None


@ingest_bp.post("/api/ingest")
def ingest():
    configured = config.PUBLISH_TOKEN.encode("utf-8")
    supplied = request.headers.get("X-Publish-Token", "").encode("utf-8")
    if not configured or not hmac.compare_digest(supplied, configured):
        return _error("unauthorized", 401)

    content_length = request.content_length
    if content_length is not None and content_length > config.MAX_INGEST_BYTES:
        return _error("请求体过大", 413)
    data = request.get_data(cache=False)
    if len(data) < 8:
        return _error("请求体过短")
    if len(data) > config.MAX_INGEST_BYTES:
        return _error("请求体过大", 413)

    json_length = struct.unpack(">I", data[:4])[0]
    if json_length <= 1 or json_length > config.MAX_INGEST_JSON_BYTES:
        return _error("JSON长度无效")
    if len(data) <= 4 + json_length:
        return _error("请求体不完整")
    try:
        meta = json.loads(data[4 : 4 + json_length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _error("JSON编码或格式错误")
    error = _validate_meta(meta)
    if error:
        return _error(error)
    jpeg_bytes = data[4 + json_length :]
    if (
        len(jpeg_bytes) < 4
        or not jpeg_bytes.startswith(b"\xff\xd8")
        or not jpeg_bytes.endswith(b"\xff\xd9")
    ):
        return _error("无效的JPEG数据")

    stream_id = meta["stream_id"]
    received = time.monotonic()
    manager_received_at_ms = int(time.time() * 1000)
    with _lock:
        previous = _latest.get(stream_id)
        previous_frame_id = (
            previous["meta"].get("frame_id") if previous is not None else None
        )
        previous_session_id = (
            previous["meta"].get("session_id")
            if previous is not None
            else None
        )
        frame_id = meta.get("frame_id")
        session_id = meta.get("session_id")
        same_session = (
            session_id is None
            or previous_session_id is None
            or session_id == previous_session_id
        )
        if (
            same_session
            and isinstance(frame_id, int)
            and isinstance(previous_frame_id, int)
            and frame_id <= previous_frame_id
        ):
            return "", 204
        _latest[stream_id] = {
            "meta": meta,
            "jpeg": jpeg_bytes,
            "received_monotonic": received,
        }

    try:
        event_ids = event_detector.process(
            meta,
            jpeg_bytes,
            manager_received_at_ms=manager_received_at_ms,
        )
    except Exception:
        current_app.logger.exception(
            "unexpected event processing failure for stream %s",
            stream_id,
        )
        event_ids = []
    if event_ids:
        meta = dict(meta)
        meta["saved_event_ids"] = event_ids
    stream_hub.publish(stream_id, meta, jpeg_bytes)
    _notify_ingest_listeners(stream_id, meta, jpeg_bytes, event_ids)
    return "", 204


@ingest_bp.get("/api/streams")
@jwt_required()
def streams():
    return jsonify(list_streams())
