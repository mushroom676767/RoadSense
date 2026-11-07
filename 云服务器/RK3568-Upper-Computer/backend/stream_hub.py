import threading
import time


class StreamHub:
    """Per-client latest-wins delivery with Socket.IO acknowledgements."""

    def __init__(self):
        self._socketio = None
        self._lock = threading.RLock()
        self._clients = {}

    def init(self, socketio):
        self._socketio = socketio

    def connect(self, sid):
        with self._lock:
            self._clients[sid] = {
                "stream_id": None,
                "include_jpeg": True,
                "jpeg_interval": 0.2,
                "last_jpeg_at": 0.0,
                "inflight": False,
                "pending": None,
                "last_ack": time.monotonic(),
            }

    def disconnect(self, sid):
        with self._lock:
            self._clients.pop(sid, None)

    def subscribe(self, sid, stream_id, include_jpeg=True, jpeg_fps=5):
        try:
            jpeg_fps = max(1.0, min(5.0, float(jpeg_fps)))
        except (TypeError, ValueError):
            jpeg_fps = 5.0
        with self._lock:
            state = self._clients.setdefault(
                sid,
                {
                    "stream_id": None,
                    "include_jpeg": True,
                    "jpeg_interval": 0.2,
                    "last_jpeg_at": 0.0,
                    "inflight": False,
                    "pending": None,
                    "last_ack": time.monotonic(),
                },
            )
            state["stream_id"] = stream_id
            state["include_jpeg"] = bool(include_jpeg)
            state["jpeg_interval"] = 1.0 / jpeg_fps
            state["last_jpeg_at"] = 0.0
            state["pending"] = None
            state["inflight"] = False

    def unsubscribe(self, sid, stream_id=None):
        with self._lock:
            state = self._clients.get(sid)
            if state and (stream_id is None or state["stream_id"] == stream_id):
                state["stream_id"] = None
                state["pending"] = None
                state["inflight"] = False

    def publish(self, stream_id, meta, jpeg_bytes):
        deliver = []
        now = time.monotonic()
        with self._lock:
            for sid, state in self._clients.items():
                if state["stream_id"] != stream_id:
                    continue
                if state["include_jpeg"]:
                    if (
                        now - state["last_jpeg_at"]
                        < state["jpeg_interval"]
                    ):
                        continue
                    state["last_jpeg_at"] = now
                    state["pending"] = (
                        "frame",
                        {"meta": meta, "jpeg": jpeg_bytes},
                    )
                else:
                    state["pending"] = ("frame_meta", {"meta": meta})
                if not state["inflight"]:
                    state["inflight"] = True
                    deliver.append(sid)
        for sid in deliver:
            self._deliver(sid)

    def _deliver(self, sid):
        with self._lock:
            state = self._clients.get(sid)
            if not state:
                return
            pending = state["pending"]
            state["pending"] = None
            if pending is None:
                state["inflight"] = False
                return
            event_name, payload = pending

        def acknowledged(*_args):
            send_next = False
            with self._lock:
                current = self._clients.get(sid)
                if not current:
                    return
                current["last_ack"] = time.monotonic()
                if current["pending"] is None:
                    current["inflight"] = False
                else:
                    send_next = True
            if send_next:
                self._deliver(sid)

        self._socketio.emit(
            event_name,
            payload,
            to=sid,
            callback=acknowledged,
        )

    def subscriber_count(self, stream_id):
        with self._lock:
            return sum(
                state["stream_id"] == stream_id
                for state in self._clients.values()
            )


stream_hub = StreamHub()
