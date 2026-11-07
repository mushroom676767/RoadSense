"""Thread-safe desired-state and telemetry registry for the device agent."""

import hmac
import threading
import time

import config


class AgentRegistry:
    def __init__(self, *, token=None, stale_seconds=None, clock=None):
        self._token = config.AGENT_TOKEN if token is None else token
        self._stale_seconds = (
            config.AGENT_STALE_SECONDS
            if stale_seconds is None
            else float(stale_seconds)
        )
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._sid = None
        self._agent_id = ""
        self._connected_at = 0.0
        self._last_seen = 0.0
        self._capabilities = []
        self._heartbeat = {}
        self._streams = {}
        self._desired = {
            "generation": 0,
            "mode": None,
            "source": "hdmi",
            "running": False,
        }
        self._last_ack = {}
        self._logs = ""

    def authenticate(self, token):
        expected = str(self._token or "")
        candidate = str(token or "")
        return bool(expected) and hmac.compare_digest(expected, candidate)

    def connect(self, sid, hello):
        now = self._clock()
        hello = hello if isinstance(hello, dict) else {}
        with self._lock:
            self._sid = str(sid)
            self._agent_id = str(
                hello.get("agent_id") or config.AGENT_ID
            )[:128]
            self._capabilities = list(hello.get("capabilities") or [])[:64]
            self._connected_at = now
            self._last_seen = now
            return dict(self._desired)

    def disconnect(self, sid):
        with self._lock:
            if self._sid == str(sid):
                self._sid = None

    def touch(self, sid, heartbeat=None):
        heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
        with self._lock:
            if self._sid != str(sid):
                return False
            self._last_seen = self._clock()
            self._heartbeat = dict(heartbeat)
            command_ack = heartbeat.get("command_ack")
            if isinstance(command_ack, dict):
                try:
                    generation = int(command_ack.get("generation"))
                except (TypeError, ValueError):
                    generation = -1
                if generation >= int(
                    self._last_ack.get("generation") or -1
                ):
                    self._last_ack = dict(command_ack)
                if generation > int(self._desired["generation"]):
                    self._desired["generation"] = generation
            return True

    def update_stream(self, sid, stream_id, meta):
        if stream_id not in config.MODES or not isinstance(meta, dict):
            return False
        with self._lock:
            if self._sid != str(sid):
                return False
            now = self._clock()
            self._last_seen = now
            self._streams[stream_id] = {
                "meta": dict(meta),
                "received_monotonic": now,
            }
            return True

    def append_log(self, sid, text):
        clean = str(text or "")
        if not clean:
            return False
        with self._lock:
            if self._sid != str(sid):
                return False
            self._last_seen = self._clock()
            self._logs = (
                self._logs + clean
            )[-config.SHARED_LOG_MAX_CHARS :]
            return True

    def set_desired(
        self,
        *,
        mode=None,
        source="hdmi",
        running=False,
    ):
        if running and mode not in config.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if source not in config.SOURCES:
            source = "hdmi"
        with self._lock:
            self._desired = {
                "generation": self._desired["generation"] + 1,
                "mode": mode if running else None,
                "source": source,
                "running": bool(running),
            }
            return dict(self._desired)

    def acknowledge(self, sid, payload):
        payload = payload if isinstance(payload, dict) else {}
        with self._lock:
            if self._sid != str(sid):
                return False
            generation = int(payload.get("generation") or -1)
            if generation < int(self._last_ack.get("generation") or -1):
                return False
            self._last_seen = self._clock()
            self._last_ack = dict(payload)
            return True

    @property
    def active_sid(self):
        with self._lock:
            return self._sid if self._online_locked() else None

    def _online_locked(self):
        return bool(
            self._sid
            and self._last_seen
            and self._clock() - self._last_seen <= self._stale_seconds
        )

    def status(self):
        with self._lock:
            online = self._online_locked()
            heartbeat = dict(self._heartbeat)
            return {
                "agent_id": self._agent_id,
                "agent_online": online,
                "connected": online,
                "connected_seconds": (
                    round(self._clock() - self._connected_at, 1)
                    if online
                    else 0
                ),
                "current_mode": heartbeat.get("mode"),
                "running": bool(heartbeat.get("running", False)),
                "board_online": bool(
                    heartbeat.get("board_online", False)
                ),
                "cloud_publisher_online": bool(
                    heartbeat.get("cloud_publisher_online", False)
                ),
                "publisher_generation": int(
                    heartbeat.get("publisher_generation") or 0
                ),
                "publisher_reconnects": int(
                    heartbeat.get("publisher_reconnects") or 0
                ),
                "active_stream": str(
                    heartbeat.get("active_stream") or ""
                ),
                "encoded_frames": int(
                    heartbeat.get("encoded_frames") or 0
                ),
                "encode_ms": float(heartbeat.get("encode_ms") or 0),
                "upload_mbps": float(
                    heartbeat.get("upload_mbps") or 0
                ),
                "active_transport": str(
                    heartbeat.get("active_transport") or ""
                ),
                "last_media_error": str(
                    heartbeat.get("last_media_error") or ""
                ),
                "desired": dict(self._desired),
                "last_ack": dict(self._last_ack),
            }

    def streams(self):
        now = self._clock()
        status = self.status()
        with self._lock:
            result = {}
            for stream_id, item in self._streams.items():
                age = now - item["received_monotonic"]
                meta = item["meta"]
                stream_online = bool(
                    status["agent_online"]
                    and age <= config.STREAM_STALE_SECONDS
                )
                publisher_matches = (
                    status["active_stream"] == stream_id
                )
                result[stream_id] = {
                    "frame_id": meta.get("frame_id"),
                    "session_id": meta.get("session_id"),
                    "mode": meta.get("mode", stream_id),
                    "age_ms": round(age * 1000, 1),
                    "online": stream_online,
                    "agent_online": status["agent_online"],
                    "board_online": stream_online,
                    "cloud_publisher_online": bool(
                        publisher_matches
                        and status["cloud_publisher_online"]
                    ),
                    "publisher_generation": status[
                        "publisher_generation"
                    ],
                    "publisher_reconnects": status[
                        "publisher_reconnects"
                    ],
                    "active_transport": status["active_transport"],
                    "active_stream": status["active_stream"],
                    "last_media_error": status["last_media_error"],
                    "encoded_frames": status["encoded_frames"],
                    "encode_ms": status["encode_ms"],
                    "upload_mbps": status["upload_mbps"],
                    "metrics": meta.get("metrics", {}),
                }
            return result

    def log_snapshot(self):
        with self._lock:
            return self._logs


agent_registry = AgentRegistry()
