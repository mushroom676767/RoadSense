import asyncio
import io
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from flask import Blueprint, jsonify, request
from flask_jwt_extended import decode_token, jwt_required

import config
from backend.cloud_media import (
    CloudMediaCapacityError,
    CloudMediaError,
    cloud_media_service,
)
from backend.ingest import get_latest


try:
    import av
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
        RTCRtpSender,
        VideoStreamTrack,
    )
    from aiortc.codecs import h264 as aiortc_h264
    from aiortc.mediastreams import MediaStreamError

    WEBRTC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - used on minimal installations
    av = None
    aiortc_h264 = None
    RTCPeerConnection = None
    MediaStreamError = RuntimeError
    WEBRTC_IMPORT_ERROR = exc


webrtc_bp = Blueprint("webrtc", __name__, url_prefix="/api/webrtc")


class WebRtcCapacityError(RuntimeError):
    pass


def _configure_h264_encoder():
    """Apply project bitrate settings without modifying aiortc installation."""
    if aiortc_h264 is None:
        return
    aiortc_h264.DEFAULT_BITRATE = config.WEBRTC_VIDEO_BITRATE
    aiortc_h264.MAX_BITRATE = config.WEBRTC_VIDEO_MAX_BITRATE


_configure_h264_encoder()


def _urls(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def browser_ice_servers():
    servers = []
    stun_urls = _urls(config.WEBRTC_STUN_URLS)
    if stun_urls:
        servers.append({"urls": stun_urls})
    turn_urls = _urls(config.WEBRTC_TURN_URLS)
    if turn_urls:
        server = {"urls": turn_urls}
        if config.WEBRTC_TURN_USERNAME:
            server["username"] = config.WEBRTC_TURN_USERNAME
        if config.WEBRTC_TURN_CREDENTIAL:
            server["credential"] = config.WEBRTC_TURN_CREDENTIAL
        servers.append(server)
    return servers


def _rtc_configuration():
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=item["urls"],
                username=item.get("username"),
                credential=item.get("credential"),
            )
            for item in browser_ice_servers()
        ]
    )


if RTCPeerConnection is not None:

    class SharedEncodedVideoTrack(VideoStreamTrack):
        """A subscriber receiving packets from one shared H.264 encoder."""

        kind = "video"

        def __init__(self, source, queue):
            super().__init__()
            self._source = source
            self._queue = queue

        async def recv(self):
            item = await self._queue.get()
            if isinstance(item, BaseException):
                self.stop()
                raise MediaStreamError
            return item

        def stop(self):
            if self.readyState != "ended":
                self._source.unsubscribe(self._queue)
            super().stop()


    class SharedH264Broadcaster:
        """Decode and encode once, then fan out H.264 packets to all peers."""

        def __init__(self, stream_id, fps, width, height):
            self.stream_id = stream_id
            self.fps = fps
            self.width = width
            self.height = height
            self._subscribers = {}
            self._task = None
            self._codec = None
            self._force_keyframe = False
            self._last_good_jpeg = None
            self._last_good_received = None
            self._encoded_frames = 0
            self._encode_ms = 0.0
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"nvenc-{stream_id}",
            )

        @staticmethod
        def _decode(jpeg):
            with av.open(io.BytesIO(jpeg), format="mjpeg") as source:
                return next(source.decode(video=0))

        def _create_encoder(self):
            codec = av.CodecContext.create(config.WEBRTC_VIDEO_ENCODER, "w")
            codec.width = self.width
            codec.height = self.height
            codec.bit_rate = config.WEBRTC_VIDEO_BITRATE
            codec.pix_fmt = "yuv420p"
            codec.framerate = Fraction(self.fps, 1)
            codec.time_base = Fraction(1, self.fps)
            if config.WEBRTC_VIDEO_ENCODER == "h264_nvenc":
                codec.options = {
                    "preset": "p1",
                    "tune": "ull",
                    "rc": "cbr",
                    "zerolatency": "1",
                    "delay": "0",
                    "bf": "0",
                    "g": str(self.fps),
                    "forced-idr": "1",
                    "repeat_headers": "1",
                    "profile": "baseline",
                    "level": "4.1",
                }
            else:
                codec.options = {
                    "preset": "veryfast",
                    "tune": "zerolatency",
                    "profile": "baseline",
                    "level": "4.1",
                    "x264-params": (
                        f"keyint={self.fps}:min-keyint={self.fps}:"
                        "scenecut=0:bframes=0:repeat-headers=1"
                    ),
                }
            codec.open()
            return codec

        @property
        def subscriber_count(self):
            return len(self._subscribers)

        @property
        def stats(self):
            return {
                "stream_id": self.stream_id,
                "subscribers": self.subscriber_count,
                "encoded_frames": self._encoded_frames,
                "encode_ms": round(self._encode_ms, 3),
                "running": self._task is not None and not self._task.done(),
            }

        def subscribe(self):
            queue = asyncio.Queue(maxsize=config.WEBRTC_PEER_QUEUE_FRAMES)
            self._subscribers[queue] = True
            self._force_keyframe = True
            if self._task is None or self._task.done():
                self._codec = self._create_encoder()
                self._task = asyncio.create_task(self._run())
            return SharedEncodedVideoTrack(self, queue)

        def unsubscribe(self, queue):
            self._subscribers.pop(queue, None)
            if not self._subscribers and self._task and not self._task.done():
                self._task.cancel()

        @staticmethod
        def _clear_queue(queue):
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

        def _broadcast_packet(self, packet):
            is_keyframe = bool(packet.is_keyframe)
            for queue, waiting_for_keyframe in list(
                self._subscribers.items()
            ):
                if waiting_for_keyframe and not is_keyframe:
                    continue
                if queue.full():
                    self._clear_queue(queue)
                    if not is_keyframe:
                        self._subscribers[queue] = True
                        self._force_keyframe = True
                        continue
                if is_keyframe:
                    self._subscribers[queue] = False
                queue.put_nowait(packet)

        def _notify_failure(self):
            for queue in list(self._subscribers):
                self._clear_queue(queue)
                queue.put_nowait(MediaStreamError())

        def _encode(
            self,
            codec,
            jpeg,
            received_monotonic,
            frame_number,
            force,
        ):
            started = time.perf_counter()
            try:
                frame = self._decode(jpeg)
                self._last_good_jpeg = jpeg
                self._last_good_received = received_monotonic
            except Exception:
                now = time.monotonic()
                if (
                    self._last_good_jpeg is None
                    or self._last_good_received is None
                    or now - self._last_good_received
                    > config.WEBRTC_SOURCE_GRACE_SECONDS
                ):
                    raise
                frame = self._decode(self._last_good_jpeg)
            if (
                frame.width != self.width
                or frame.height != self.height
                or frame.format.name != "yuv420p"
            ):
                frame = frame.reformat(
                    width=self.width,
                    height=self.height,
                    format="yuv420p",
                )
            frame.pts = frame_number
            frame.time_base = Fraction(1, self.fps)
            frame.pict_type = (
                av.video.frame.PictureType.I
                if force
                else av.video.frame.PictureType.NONE
            )
            packets = list(codec.encode(frame))
            self._encode_ms = (time.perf_counter() - started) * 1000
            return packets

        async def _run(self):
            loop = asyncio.get_running_loop()
            frame_number = 0
            next_frame_at = time.monotonic()
            try:
                while self._subscribers:
                    delay = next_frame_at - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

                    latest = get_latest(
                        self.stream_id,
                        max_age_seconds=config.WEBRTC_SOURCE_GRACE_SECONDS,
                    )
                    if latest is None:
                        raise MediaStreamError
                    force = self._force_keyframe
                    self._force_keyframe = False
                    codec = self._codec
                    packets = await loop.run_in_executor(
                        self._executor,
                        self._encode,
                        codec,
                        latest["jpeg"],
                        latest["received_monotonic"],
                        frame_number,
                        force,
                    )
                    for packet in packets:
                        self._broadcast_packet(packet)
                    self._encoded_frames += 1
                    frame_number += 1
                    next_frame_at = max(
                        next_frame_at + 1 / self.fps,
                        time.monotonic(),
                    )
            except asyncio.CancelledError:
                pass
            except Exception:
                self._notify_failure()
            finally:
                self._codec = None
                self._task = None
class WebRtcService:
    def __init__(self):
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()
        self._peers = {}
        self._sources = {}

    @property
    def available(self):
        return RTCPeerConnection is not None

    @property
    def unavailable_reason(self):
        return str(WEBRTC_IMPORT_ERROR or "")

    @property
    def peer_count(self):
        return len(self._peers)

    @property
    def source_stats(self):
        return [source.stats for source in self._sources.values()]

    def _ensure_loop(self):
        with self._lock:
            if self._loop and self._loop.is_running():
                return
            ready = threading.Event()

            def run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(
                target=run,
                name="webrtc-event-loop",
                daemon=True,
            )
            self._thread.start()
            ready.wait(timeout=3)
        if not self._loop:
            raise RuntimeError("WebRTC event loop failed to start")

    def create_answer(self, stream_id, sdp, description_type):
        if not self.available:
            raise RuntimeError(
                f"WebRTC dependencies are unavailable: {self.unavailable_reason}"
            )
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._create_answer(stream_id, sdp, description_type),
            self._loop,
        )
        return future.result(timeout=config.WEBRTC_OFFER_TIMEOUT_SECONDS)

    async def _create_answer(self, stream_id, sdp, description_type):
        if self.peer_count >= config.WEBRTC_MAX_VIEWERS:
            raise WebRtcCapacityError(
                f"WebRTC viewer limit reached ({config.WEBRTC_MAX_VIEWERS})"
            )
        peer_id = uuid.uuid4().hex
        pc = RTCPeerConnection(configuration=_rtc_configuration())
        source = self._sources.get(stream_id)
        if source is None:
            source = SharedH264Broadcaster(
                stream_id,
                config.WEBRTC_VIDEO_FPS,
                config.WEBRTC_VIDEO_WIDTH,
                config.WEBRTC_VIDEO_HEIGHT,
            )
            self._sources[stream_id] = source
        track = source.subscribe()
        transceiver = pc.addTransceiver(track, direction="sendonly")
        h264_codecs = [
            codec
            for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/h264"
        ]
        if not h264_codecs:
            track.stop()
            await pc.close()
            raise RuntimeError("The installed WebRTC stack has no H.264 encoder")
        transceiver.setCodecPreferences(h264_codecs)
        self._peers[peer_id] = (pc, track)

        @pc.on("connectionstatechange")
        async def connection_state_changed():
            if pc.connectionState in {"failed", "closed"}:
                entry = self._peers.pop(peer_id, None)
                if entry is not None:
                    entry[1].stop()
                if pc.connectionState == "failed":
                    asyncio.create_task(pc.close())

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=description_type)
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return {
                "peer_id": peer_id,
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        except Exception:
            self._peers.pop(peer_id, None)
            track.stop()
            await pc.close()
            raise

    def close_peer(self, peer_id):
        if not self._loop or not self._loop.is_running():
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._close_peer(peer_id),
            self._loop,
        )
        return future.result(timeout=5)

    def touch_peer(self, peer_id):
        return peer_id in self._peers

    async def _close_peer(self, peer_id):
        entry = self._peers.pop(peer_id, None)
        if entry is None:
            return False
        pc, track = entry
        track.stop()
        await pc.close()
        return True


webrtc_service = WebRtcService()


def _active_service():
    if config.WEBRTC_PROVIDER == "mediamtx":
        return cloud_media_service
    return webrtc_service


@webrtc_bp.get("/config")
@jwt_required()
def webrtc_config():
    service = _active_service()
    return jsonify(
        {
            "transport": config.VIDEO_TRANSPORT,
            "provider": config.WEBRTC_PROVIDER,
            "available": service.available,
            "unavailable_reason": (
                "" if service.available else service.unavailable_reason
            ),
            "iceServers": browser_ice_servers(),
            "video_fps": config.WEBRTC_VIDEO_FPS,
            "video_width": config.WEBRTC_VIDEO_WIDTH,
            "video_height": config.WEBRTC_VIDEO_HEIGHT,
            "video_bitrate": config.WEBRTC_VIDEO_BITRATE,
            "video_max_bitrate": config.WEBRTC_VIDEO_MAX_BITRATE,
            "video_encoder": config.WEBRTC_VIDEO_ENCODER,
            "shared_encoding": True,
            "single_cloud_upload": (
                config.WEBRTC_PROVIDER == "mediamtx"
            ),
            "active_viewers": service.peer_count,
            "max_viewers": config.WEBRTC_MAX_VIEWERS,
            "encoder_sources": service.source_stats,
            "source_continuity": (
                config.WEBRTC_PROVIDER == "mediamtx"
                and not config.MEDIA_MTX_REQUIRE_PUBLISHER_READY
            ),
            "ingest_transport": config.MEDIA_MTX_INGEST_TRANSPORT,
            "hls_fallback_url": config.HLS_FALLBACK_BASE_URL,
            "available_transports": ["webrtc", "ll-hls"],
            "fallback": (
                "ll-hls"
                if config.DEPLOYMENT_ROLE == "cloud"
                else "jpeg"
            ),
        }
    )


@webrtc_bp.post("/offer")
@jwt_required()
def webrtc_offer():
    service = _active_service()
    if config.VIDEO_TRANSPORT != "webrtc":
        return jsonify({"error": "WebRTC video transport is disabled"}), 409
    if not service.available:
        return jsonify(
            {
                "error": "WebRTC is unavailable on the manager",
                "detail": service.unavailable_reason,
            }
        ), 503
    data = request.get_json(silent=True) or {}
    stream_id = str(data.get("stream_id") or "")
    sdp = str(data.get("sdp") or "")
    description_type = str(data.get("type") or "")
    if stream_id not in config.MODES:
        return jsonify({"error": "Unknown stream"}), 400
    if description_type != "offer" or not sdp or len(sdp) > 1024 * 1024:
        return jsonify({"error": "Invalid WebRTC offer"}), 400
    if (
        config.DEPLOYMENT_ROLE != "cloud"
        and get_latest(stream_id, config.STREAM_STALE_SECONDS) is None
    ):
        return jsonify(
            {
                "error": "板端视频源当前离线",
                "code": "stream_offline",
            }
        ), 409
    try:
        return jsonify(
            service.create_answer(stream_id, sdp, description_type)
        )
    except (WebRtcCapacityError, CloudMediaCapacityError) as exc:
        return jsonify({"error": str(exc), "code": "viewer_limit"}), 429
    except CloudMediaError as exc:
        return jsonify(
            {"error": "Cloud media service failed", "detail": str(exc)}
        ), 503
    except TimeoutError:
        return jsonify({"error": "WebRTC negotiation timed out"}), 504
    except Exception as exc:
        return jsonify(
            {"error": "WebRTC negotiation failed", "detail": str(exc)}
        ), 500


@webrtc_bp.post("/close")
@jwt_required()
def webrtc_close():
    peer_id = str((request.get_json(silent=True) or {}).get("peer_id") or "")
    if not peer_id:
        return jsonify({"error": "peer_id is required"}), 400
    service = _active_service()
    closed = service.close_peer(peer_id)
    if not closed:
        fallback = (
            webrtc_service
            if service is cloud_media_service
            else cloud_media_service
        )
        closed = fallback.close_peer(peer_id)
    return jsonify({"closed": closed})


@webrtc_bp.post("/heartbeat")
@jwt_required()
def webrtc_heartbeat():
    peer_id = str((request.get_json(silent=True) or {}).get("peer_id") or "")
    if not peer_id:
        return jsonify({"error": "peer_id is required"}), 400
    service = _active_service()
    alive = service.touch_peer(peer_id)
    if not alive:
        fallback = (
            webrtc_service
            if service is cloud_media_service
            else cloud_media_service
        )
        alive = fallback.touch_peer(peer_id)
    return jsonify({"alive": alive})


@webrtc_bp.get("/media-authorize")
def media_authorize():
    """Nginx auth_request endpoint for protected LL-HLS resources."""
    authorization = request.headers.get("Authorization", "")
    cookie_token = request.cookies.get("access_token_cookie", "")
    token = (
        authorization[7:].strip()
        if authorization.lower().startswith("bearer ")
        else cookie_token
    )
    if not token:
        return "", 401
    try:
        claims = decode_token(token)
    except Exception:
        return "", 401
    if not claims.get("sub"):
        return "", 401
    return "", 204
