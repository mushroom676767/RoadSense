"""Windows device agent: local RK control plus outbound cloud telemetry."""

import queue
import threading
import time

import socketio

import config
from backend.cloud_media import NvencSrtPublisher
from backend.event_outbox import EventOutbox
from backend.ingest import (
    register_ingest_listener,
    unregister_ingest_listener,
)
from backend.ssh_manager import SSHManager


AGENT_NAMESPACE = "/agent"


class _AgentEmitter:
    def __init__(self, agent):
        self.agent = agent

    def emit(self, name, payload):
        self.agent.emit(name, payload)


class DeviceAgent:
    def __init__(self, client=None):
        self.client = client or socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=0.5,
            reconnection_delay_max=config.AGENT_RECONNECT_MAX_SECONDS,
            logger=False,
            engineio_logger=False,
        )
        self._stop = threading.Event()
        self._latest_meta = None
        self._meta_lock = threading.Lock()
        self._desired_queue = queue.Queue(maxsize=1)
        self._generation = -1
        self._last_command_ack = {}
        self._publisher = None
        self._publisher_stream = None
        self._publisher_lock = threading.RLock()
        self._media_wakeup = threading.Event()
        self._latest_jpeg_stream = None
        self._latest_jpeg_at = 0.0
        self._last_media_error = ""
        self.outbox = EventOutbox()
        self.control = SSHManager(_AgentEmitter(self))
        self._threads = []
        self._register_handlers()

    @property
    def connected(self):
        return bool(getattr(self.client, "connected", False))

    def _register_handlers(self):
        self.client.on(
            "desired_state",
            self._receive_desired,
            namespace=AGENT_NAMESPACE,
        )

    def emit(self, name, payload, callback=None):
        if not self.connected:
            return False
        try:
            self.client.emit(
                name,
                payload,
                namespace=AGENT_NAMESPACE,
                callback=callback,
            )
            return True
        except Exception:
            return False

    def _receive_desired(self, desired):
        desired = desired if isinstance(desired, dict) else {}
        try:
            generation = int(desired.get("generation"))
        except (TypeError, ValueError):
            return
        if generation <= self._generation:
            return
        while True:
            try:
                self._desired_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._desired_queue.put_nowait(dict(desired))
        except queue.Full:
            pass

    def _on_ingest(self, stream_id, meta, jpeg_bytes, event_ids):
        received = time.monotonic()
        with self._meta_lock:
            self._latest_meta = {
                "stream_id": stream_id,
                "meta": dict(meta),
            }
        with self._publisher_lock:
            self._latest_jpeg_stream = stream_id
            self._latest_jpeg_at = received
        self._media_wakeup.set()

    def _start_publisher(self, stream_id):
        with self._publisher_lock:
            if self._publisher_stream == stream_id and self._publisher:
                return False
            self._stop_publisher_locked()
            if config.MEDIA_MTX_INGEST_TRANSPORT != "srt":
                return False
            publisher = NvencSrtPublisher(stream_id)
            try:
                publisher.start()
            except Exception as exc:
                self._last_media_error = publisher._sanitize_error(exc)
                publisher.stop()
                raise
            self._publisher = publisher
            self._publisher_stream = stream_id
            self._last_media_error = ""
            return True

    def _stop_publisher(self):
        with self._publisher_lock:
            self._stop_publisher_locked()

    def _stop_publisher_locked(self):
        publisher = self._publisher
        self._publisher = None
        self._publisher_stream = None
        if publisher is not None:
            stats = publisher.stats
            self._last_media_error = str(stats.get("error") or "")
            publisher.stop()

    def _clear_media_source(self):
        with self._publisher_lock:
            self._latest_jpeg_stream = None
            self._latest_jpeg_at = 0.0

    def _reconcile_media(self, now=None):
        current = time.monotonic() if now is None else float(now)
        with self._publisher_lock:
            stream_id = self._latest_jpeg_stream
            age = current - self._latest_jpeg_at
            fresh = bool(
                stream_id
                and self._latest_jpeg_at
                and age <= config.WEBRTC_SOURCE_GRACE_SECONDS
            )
        if fresh:
            try:
                self._start_publisher(stream_id)
            except Exception:
                return False
            return True
        self._stop_publisher()
        return False

    def _media_loop(self):
        while not self._stop.is_set():
            self._media_wakeup.wait(timeout=0.25)
            self._media_wakeup.clear()
            self._reconcile_media()

    def _publisher_snapshot(self, now=None):
        current = time.monotonic() if now is None else float(now)
        with self._publisher_lock:
            stream_id = self._publisher_stream
            publisher = self._publisher
            latest_stream = self._latest_jpeg_stream
            latest_at = self._latest_jpeg_at
            fallback_error = self._last_media_error
        stats = publisher.stats if publisher is not None else {}
        board_online = bool(
            latest_stream
            and latest_at
            and current - latest_at <= config.STREAM_STALE_SECONDS
        )
        return stream_id, board_online, stats, fallback_error

    def _apply_desired(self, desired):
        generation = int(desired["generation"])
        if generation <= self._generation:
            return
        running = bool(desired.get("running"))
        if running:
            mode = str(desired.get("mode") or "")
            source = str(desired.get("source") or "hdmi")
            result = self.control.start_mode(mode, source=source)
        else:
            result = self.control.stop_mode()
            self._clear_media_source()
            self._stop_publisher()
        self._generation = generation
        acknowledgement = {
            "generation": generation,
            "success": "error" not in result,
            "error": result.get("error", ""),
            "status": self.control.get_status(),
        }
        self._last_command_ack = acknowledgement
        self.emit("command_ack", acknowledgement)

    def _command_loop(self):
        while not self._stop.is_set():
            try:
                desired = self._desired_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._apply_desired(desired)

    def _metadata_loop(self):
        while not self._stop.wait(1 / max(1, config.PUBLISH_FPS)):
            if not self.connected:
                continue
            with self._meta_lock:
                payload = self._latest_meta
                self._latest_meta = None
            if payload:
                self.emit("frame_meta", payload)

    def _heartbeat_loop(self):
        while not self._stop.wait(2):
            status = self.control.get_status()
            active_stream, board_online, publisher_stats, media_error = (
                self._publisher_snapshot()
            )
            mode = status.get("current_mode") or active_stream
            self.emit(
                "heartbeat",
                {
                    "mode": mode,
                    "running": bool(status.get("running")),
                    "board_online": board_online,
                    "cloud_publisher_online": bool(
                        publisher_stats.get("ready")
                    ),
                    "publisher_generation": publisher_stats.get(
                        "publisher_generation", 0
                    ),
                    "publisher_reconnects": publisher_stats.get(
                        "publisher_reconnects", 0
                    ),
                    "active_stream": active_stream,
                    "active_transport": (
                        config.MEDIA_MTX_INGEST_TRANSPORT
                    ),
                    "last_media_error": (
                        publisher_stats.get("error") or media_error
                    ),
                    "encoded_frames": publisher_stats.get(
                        "encoded_frames", 0
                    ),
                    "encode_ms": publisher_stats.get("encode_ms", 0),
                    "upload_mbps": publisher_stats.get("upload_mbps", 0),
                    "pending_events": self.outbox.count(),
                    "command_ack": dict(self._last_command_ack),
                },
                callback=self._receive_heartbeat_ack,
            )

    def _receive_heartbeat_ack(self, response=None):
        if not isinstance(response, dict) or not response.get("ok"):
            return
        desired = response.get("desired")
        if isinstance(desired, dict):
            self._receive_desired(desired)

    def _event_loop(self):
        backoff = 0.25
        while not self._stop.is_set():
            if not self.connected:
                self._stop.wait(min(backoff, 2))
                backoff = min(backoff * 2, 2)
                continue
            item = self.outbox.next_pending()
            if item is None:
                backoff = 0.25
                self._stop.wait(0.25)
                continue
            external_id = item["external_event_id"]
            try:
                with open(item["snapshot_path"], "rb") as handle:
                    jpeg = handle.read()
            except OSError as exc:
                self.outbox.fail(external_id, exc)
                self._stop.wait(min(backoff, 5))
                backoff = min(backoff * 2, 5)
                continue

            acknowledged = threading.Event()
            accepted = {"ok": False, "error": "acknowledgement timeout"}

            def on_ack(response=None):
                if isinstance(response, dict):
                    accepted.update(response)
                acknowledged.set()

            if not self.emit(
                "event",
                {"event": item["payload"], "jpeg": jpeg},
                callback=on_ack,
            ):
                self._stop.wait(min(backoff, 2))
                backoff = min(backoff * 2, 2)
                continue
            if not acknowledged.wait(5):
                self.outbox.fail(external_id, accepted["error"])
                continue
            if accepted.get("ok"):
                self.outbox.acknowledge(external_id)
                backoff = 0.25
            else:
                self.outbox.fail(
                    external_id,
                    accepted.get("error") or "cloud rejected event",
                )
                self._stop.wait(min(backoff, 5))
                backoff = min(backoff * 2, 5)

    def _connection_loop(self):
        while not self._stop.is_set():
            if self.connected:
                self._stop.wait(1)
                continue
            try:
                self.client.connect(
                    config.AGENT_CLOUD_URL,
                    auth={
                        "token": config.AGENT_TOKEN,
                        "agent_id": config.AGENT_ID,
                        "capabilities": ["srt", "events", "ssh-control"],
                    },
                    namespaces=[AGENT_NAMESPACE],
                    transports=["websocket"],
                    wait_timeout=8,
                )
            except Exception:
                self._stop.wait(1)

    def start(self):
        register_ingest_listener(self._on_ingest)
        for target, name in (
            (self._connection_loop, "agent-cloud"),
            (self._command_loop, "agent-command"),
            (self._metadata_loop, "agent-metadata"),
            (self._heartbeat_loop, "agent-heartbeat"),
            (self._media_loop, "agent-media-supervisor"),
            (self._event_loop, "agent-events"),
        ):
            thread = threading.Thread(
                target=target,
                name=name,
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self):
        self._stop.set()
        unregister_ingest_listener(self._on_ingest)
        self._media_wakeup.set()
        self._stop_publisher()
        self.control.close()
        try:
            self.client.disconnect()
        except Exception:
            pass
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)
