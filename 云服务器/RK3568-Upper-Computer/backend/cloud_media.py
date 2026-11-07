"""Single-upload NVENC publisher and MediaMTX WHEP signaling proxy."""

import atexit
import base64
import io
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction

try:
    import av
except ImportError:  # Cloud role does not encode video.
    av = None

import config
from backend.ingest import get_latest


class CloudMediaError(RuntimeError):
    pass


class CloudMediaCapacityError(CloudMediaError):
    pass


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class NvencRtspPublisher:
    """Poll the latest JPEG, encode once with NVENC and publish over RTSP."""

    def __init__(
        self,
        stream_id,
        ensure_tunnel,
        invalidate_tunnel=None,
        tunnel_online=None,
    ):
        self.stream_id = stream_id
        self._ensure_tunnel = ensure_tunnel
        self._invalidate_tunnel = invalidate_tunnel or (lambda: None)
        self._tunnel_online = tunnel_online or (lambda: True)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._last_error = ""
        self._running = False
        self._encoded_frames = 0
        self._encode_ms = 0.0
        self._upload_mbps = 0.0
        self._last_good_jpeg = None
        self._last_good_received = None
        self._consecutive_publish_failures = 0

    @property
    def ready(self):
        return self._ready.is_set()

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    @property
    def stats(self):
        with self._lock:
            return {
                "stream_id": self.stream_id,
                "running": self._running,
                "ready": self.ready,
                "encoded_frames": self._encoded_frames,
                "encode_ms": round(self._encode_ms, 3),
                "upload_mbps": round(self._upload_mbps, 3),
                "error": self._last_error,
            }

    def start(self):
        if av is None:
            raise CloudMediaError("PyAV is required by the media publisher")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"mediamtx-publish-{self.stream_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout=3.0):
        self._stop.set()
        self._ready.clear()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is not None and not thread.is_alive():
            self._thread = None

    def wait_ready(self, timeout):
        return self._ready.wait(timeout)

    def _set_state(self, *, running=None, error=None):
        with self._lock:
            if running is not None:
                self._running = running
            if error is not None:
                self._last_error = self._sanitize_error(error)

    @staticmethod
    def _sanitize_error(error):
        message = str(error)
        for secret in (
            config.MEDIA_MTX_PUBLISH_PASSWORD,
            config.MEDIA_MTX_SRT_PASSPHRASE,
        ):
            if secret:
                message = message.replace(secret, "********")
                message = message.replace(
                    urllib.parse.quote(secret, safe=""),
                    "********",
                )
        return message

    def _rtsp_url(self):
        username = urllib.parse.quote(
            config.MEDIA_MTX_PUBLISH_USERNAME, safe=""
        )
        password = urllib.parse.quote(
            config.MEDIA_MTX_PUBLISH_PASSWORD, safe=""
        )
        return (
            f"rtsp://{username}:{password}@127.0.0.1:"
            f"{config.MEDIA_MTX_LOCAL_RTSP_PORT}/{self.stream_id}"
        )

    @staticmethod
    def _decode(jpeg):
        with av.open(io.BytesIO(jpeg), format="mjpeg") as source:
            return next(source.decode(video=0))

    def _prepare_frame(self, latest, frame_number):
        jpeg = latest["jpeg"]
        try:
            frame = self._decode(jpeg)
            self._last_good_jpeg = jpeg
            self._last_good_received = latest["received_monotonic"]
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
            frame.width != config.WEBRTC_VIDEO_WIDTH
            or frame.height != config.WEBRTC_VIDEO_HEIGHT
            or frame.format.name != "yuv420p"
        ):
            frame = frame.reformat(
                width=config.WEBRTC_VIDEO_WIDTH,
                height=config.WEBRTC_VIDEO_HEIGHT,
                format="yuv420p",
            )
        frame.pts = frame_number
        frame.time_base = Fraction(1, config.WEBRTC_VIDEO_FPS)
        return frame

    @staticmethod
    def _configure_stream(output):
        stream = output.add_stream(
            config.WEBRTC_VIDEO_ENCODER,
            rate=config.WEBRTC_VIDEO_FPS,
        )
        codec = stream.codec_context
        codec.width = config.WEBRTC_VIDEO_WIDTH
        codec.height = config.WEBRTC_VIDEO_HEIGHT
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = config.WEBRTC_VIDEO_BITRATE
        codec.time_base = Fraction(1, config.WEBRTC_VIDEO_FPS)
        codec.options = {
            "preset": "p1",
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "delay": "0",
            "bf": "0",
            "g": str(config.WEBRTC_VIDEO_FPS),
            "forced-idr": "1",
            "repeat_headers": "1",
            "profile": "baseline",
            "level": "4.1",
        }
        return stream

    def _publish_session(self):
        self._ensure_tunnel()
        output = av.open(
            self._rtsp_url(),
            mode="w",
            format="rtsp",
            options={"rtsp_transport": "tcp"},
        )
        stream = self._configure_stream(output)
        frame_number = 0
        next_frame_at = time.monotonic()
        byte_count = 0
        bytes_started = time.monotonic()
        try:
            while not self._stop.is_set():
                latest = get_latest(
                    self.stream_id,
                    max_age_seconds=config.WEBRTC_SOURCE_GRACE_SECONDS,
                )
                if latest is None:
                    raise CloudMediaError("board video source is offline")
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                    if self._stop.is_set():
                        break
                encode_started = time.perf_counter()
                frame = self._prepare_frame(latest, frame_number)
                packets = list(stream.encode(frame))
                with self._lock:
                    self._encode_ms = (
                        time.perf_counter() - encode_started
                    ) * 1000
                for packet in packets:
                    byte_count += packet.size
                    output.mux(packet)
                    self._ready.set()
                    self._consecutive_publish_failures = 0
                    self._set_state(error="")
                frame_number += 1
                with self._lock:
                    self._encoded_frames += 1
                    elapsed = time.monotonic() - bytes_started
                    if elapsed > 0:
                        self._upload_mbps = (
                            byte_count * 8 / elapsed / 1_000_000
                        )
                next_frame_at = max(
                    next_frame_at + 1 / config.WEBRTC_VIDEO_FPS,
                    time.monotonic(),
                )
        finally:
            try:
                for packet in stream.encode(None):
                    output.mux(packet)
            except Exception:
                pass
            output.close()

    def _run(self):
        backoff = 0.5
        self._set_state(running=True, error="")
        while not self._stop.is_set():
            if (
                get_latest(
                    self.stream_id,
                    max_age_seconds=config.STREAM_STALE_SECONDS,
                )
                is None
            ):
                self._ready.clear()
                self._stop.wait(0.2)
                continue
            try:
                self._publish_session()
                backoff = 0.5
            except Exception as exc:
                self._ready.clear()
                self._set_state(error=exc)
                if not (
                    isinstance(exc, CloudMediaError)
                    and str(exc) == "board video source is offline"
                ):
                    self._consecutive_publish_failures += 1
                    if (
                        not self._tunnel_online()
                        or self._consecutive_publish_failures >= 5
                    ):
                        self._invalidate_tunnel()
                        self._consecutive_publish_failures = 0
                if self._consecutive_publish_failures <= 1:
                    backoff = 0.5
                else:
                    backoff = min(backoff * 2, 2)
                self._stop.wait(backoff)
        self._ready.clear()
        self._set_state(running=False)


class NvencSrtPublisher(NvencRtspPublisher):
    """Publish the latest manager frame as low-latency MPEG-TS over SRT."""

    def __init__(self, stream_id):
        super().__init__(stream_id, lambda: None)

    @property
    def stats(self):
        result = super().stats
        result["transport"] = "srt"
        with self._lock:
            result["publisher_generation"] = getattr(
                self, "_publisher_generation", 0
            )
            result["publisher_reconnects"] = getattr(
                self, "_publisher_reconnects", 0
            )
        return result

    def _srt_url(self):
        if not config.MEDIA_MTX_SRT_HOST:
            raise CloudMediaError("MEDIA_MTX_SRT_HOST is not configured")
        stream_id = (
            f"publish:{self.stream_id}:"
            f"{config.MEDIA_MTX_PUBLISH_USERNAME}:"
            f"{config.MEDIA_MTX_PUBLISH_PASSWORD}"
        )
        query = {
            "mode": "caller",
            "transtype": "live",
            "latency": str(config.MEDIA_MTX_SRT_LATENCY_MS * 1000),
            "peerlatency": str(config.MEDIA_MTX_SRT_LATENCY_MS * 1000),
            "tlpktdrop": "1",
            "snddropdelay": "300000",
            "connect_timeout": "3000000",
            "rw_timeout": "500000",
            "pkt_size": "1316",
            "streamid": stream_id,
        }
        if config.MEDIA_MTX_SRT_PASSPHRASE:
            query["passphrase"] = config.MEDIA_MTX_SRT_PASSPHRASE
            query["pbkeylen"] = "32"
        return (
            f"srt://{config.MEDIA_MTX_SRT_HOST}:"
            f"{config.MEDIA_MTX_SRT_PORT}?"
            + urllib.parse.urlencode(query, safe=":")
        )

    @staticmethod
    def _pyav_supports_srt():
        return bool(
            av is not None
            and "srt" in getattr(av, "protocols_available", ())
        )

    @staticmethod
    def _resolve_ffmpeg():
        configured = config.FFMPEG_PATH
        if os.path.isfile(configured):
            return os.path.abspath(configured)
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise CloudMediaError(
            "PyAV has no SRT support and FFmpeg was not found; "
            "install an FFmpeg build with libsrt and NVENC"
        )

    def _ffmpeg_command(self):
        fps = max(1, config.WEBRTC_VIDEO_FPS)
        keyframe_interval = max(1, fps // 2)
        return [
            self._resolve_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "mjpeg",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            (
                f"scale={config.WEBRTC_VIDEO_WIDTH}:"
                f"{config.WEBRTC_VIDEO_HEIGHT}:"
                "flags=fast_bilinear,format=yuv420p"
            ),
            "-c:v",
            config.WEBRTC_VIDEO_ENCODER,
            "-flags",
            "low_delay",
            "-preset",
            "p1",
            "-tune",
            "ull",
            "-rc",
            "cbr",
            "-b:v",
            str(config.WEBRTC_VIDEO_BITRATE),
            "-maxrate",
            str(config.WEBRTC_VIDEO_MAX_BITRATE),
            "-bufsize",
            str(config.WEBRTC_VIDEO_MAX_BITRATE),
            "-g",
            str(keyframe_interval),
            "-keyint_min",
            str(keyframe_interval),
            "-bf",
            "0",
            "-rc-lookahead",
            "0",
            "-forced-idr",
            "1",
            "-profile:v",
            "baseline",
            "-level:v",
            "4.1",
            "-fps_mode",
            "cfr",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-mpegts_flags",
            "+resend_headers",
            "-flush_packets",
            "1",
            "-f",
            "mpegts",
            self._ffmpeg_srt_url(),
        ]

    def _ffmpeg_srt_url(self):
        parsed = urllib.parse.urlsplit(self._srt_url())
        query = [
            (key, value)
            for key, value in urllib.parse.parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if key != "rw_timeout"
        ]
        return urllib.parse.urlunsplit(
            parsed._replace(
                query=urllib.parse.urlencode(query, safe=":")
            )
        )

    @staticmethod
    def _configure_stream(output):
        stream = output.add_stream(
            config.WEBRTC_VIDEO_ENCODER,
            rate=config.WEBRTC_VIDEO_FPS,
        )
        codec = stream.codec_context
        codec.width = config.WEBRTC_VIDEO_WIDTH
        codec.height = config.WEBRTC_VIDEO_HEIGHT
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = config.WEBRTC_VIDEO_BITRATE
        codec.bit_rate_tolerance = max(
            0,
            config.WEBRTC_VIDEO_MAX_BITRATE
            - config.WEBRTC_VIDEO_BITRATE,
        )
        codec.time_base = Fraction(1, config.WEBRTC_VIDEO_FPS)
        codec.options = {
            "preset": "p1",
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "delay": "0",
            "bf": "0",
            "g": str(max(1, config.WEBRTC_VIDEO_FPS // 2)),
            "forced-idr": "1",
            "repeat_headers": "1",
            "profile": "baseline",
            "level": "4.1",
        }
        return stream

    def _publish_session_pyav(self):
        output = av.open(
            self._srt_url(),
            mode="w",
            format="mpegts",
            options={
                "mpegts_flags": "resend_headers",
                "muxdelay": "0",
                "muxpreload": "0",
                "flush_packets": "1",
            },
        )
        stream = self._configure_stream(output)
        with self._lock:
            self._publisher_generation = getattr(
                self, "_publisher_generation", 0
            ) + 1
            if self._publisher_generation > 1:
                self._publisher_reconnects = getattr(
                    self, "_publisher_reconnects", 0
                ) + 1
        frame_number = 0
        next_frame_at = time.monotonic()
        byte_count = 0
        bytes_started = time.monotonic()
        try:
            while not self._stop.is_set():
                latest = get_latest(
                    self.stream_id,
                    max_age_seconds=config.WEBRTC_SOURCE_GRACE_SECONDS,
                )
                if latest is None:
                    raise CloudMediaError("board video source is offline")
                delay = next_frame_at - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
                encode_started = time.perf_counter()
                frame = self._prepare_frame(latest, frame_number)
                packets = list(stream.encode(frame))
                with self._lock:
                    self._encode_ms = (
                        time.perf_counter() - encode_started
                    ) * 1000
                for packet in packets:
                    mux_started = time.perf_counter()
                    output.mux(packet)
                    mux_ms = (time.perf_counter() - mux_started) * 1000
                    if mux_ms > 500:
                        raise TimeoutError(
                            f"SRT mux exceeded 500ms ({mux_ms:.1f}ms)"
                        )
                    byte_count += packet.size
                    self._ready.set()
                    self._consecutive_publish_failures = 0
                    self._set_state(error="")
                frame_number += 1
                with self._lock:
                    self._encoded_frames += 1
                    elapsed = time.monotonic() - bytes_started
                    if elapsed > 0:
                        self._upload_mbps = (
                            byte_count * 8 / elapsed / 1_000_000
                        )
                next_frame_at = max(
                    next_frame_at + 1 / config.WEBRTC_VIDEO_FPS,
                    time.monotonic(),
                )
        finally:
            try:
                for packet in stream.encode(None):
                    output.mux(packet)
            except Exception:
                pass
            output.close()

    @staticmethod
    def _ffmpeg_error(process):
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read().decode(
                "utf-8", errors="replace"
            ).strip()
        except Exception:
            return ""

    def _publish_session_ffmpeg(self):
        creation_flags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        process = subprocess.Popen(
            self._ffmpeg_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creation_flags,
        )
        with self._lock:
            self._publisher_generation = getattr(
                self, "_publisher_generation", 0
            ) + 1
            if self._publisher_generation > 1:
                self._publisher_reconnects = getattr(
                    self, "_publisher_reconnects", 0
                ) + 1

        next_frame_at = time.monotonic()
        try:
            while not self._stop.is_set():
                latest = get_latest(
                    self.stream_id,
                    max_age_seconds=config.WEBRTC_SOURCE_GRACE_SECONDS,
                )
                if latest is None:
                    raise CloudMediaError("board video source is offline")
                delay = next_frame_at - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break

                exit_code = process.poll()
                if exit_code is not None:
                    detail = self._ffmpeg_error(process)
                    raise CloudMediaError(
                        f"FFmpeg SRT publisher exited ({exit_code}): "
                        f"{detail or 'no error output'}"
                    )
                if process.stdin is None:
                    raise CloudMediaError(
                        "FFmpeg SRT publisher has no input pipe"
                    )

                write_started = time.perf_counter()
                try:
                    process.stdin.write(latest["jpeg"])
                    process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    detail = self._ffmpeg_error(process)
                    raise CloudMediaError(
                        "FFmpeg SRT publisher pipe failed: "
                        f"{detail or exc}"
                    ) from exc
                write_ms = (
                    time.perf_counter() - write_started
                ) * 1000
                if write_ms > 500:
                    raise TimeoutError(
                        f"FFmpeg SRT input exceeded 500ms "
                        f"({write_ms:.1f}ms)"
                    )

                self._ready.set()
                self._consecutive_publish_failures = 0
                self._set_state(error="")
                with self._lock:
                    self._encoded_frames += 1
                    self._encode_ms = write_ms
                    self._upload_mbps = (
                        config.WEBRTC_VIDEO_BITRATE / 1_000_000
                    )
                next_frame_at = max(
                    next_frame_at
                    + 1 / max(1, config.WEBRTC_VIDEO_FPS),
                    time.monotonic(),
                )
        finally:
            self._ready.clear()
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stderr is not None:
                process.stderr.close()

    def _publish_session(self):
        if self._pyav_supports_srt():
            return self._publish_session_pyav()
        return self._publish_session_ffmpeg()


class CloudMediaService:
    """Proxy WHEP sessions and optionally own the legacy media publisher."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tunnel_process = None
        self._publishers = {}
        self._sessions = {}
        self._pending_sessions = 0
        atexit.register(self.stop)

    @property
    def available(self):
        if self._uses_direct_whep:
            return bool(
                config.MEDIA_MTX_READ_USERNAME
                and config.MEDIA_MTX_READ_PASSWORD
            )
        return bool(
            config.MEDIA_MTX_HOST
            and config.MEDIA_MTX_PUBLISH_USERNAME
            and config.MEDIA_MTX_PUBLISH_PASSWORD
            and config.MEDIA_MTX_READ_USERNAME
            and config.MEDIA_MTX_READ_PASSWORD
            and os.path.isfile(config.MEDIA_MTX_SSH_KEY)
        )

    @property
    def unavailable_reason(self):
        if self._uses_direct_whep:
            return "MediaMTX read credentials are incomplete"
        if not config.MEDIA_MTX_HOST:
            return "MEDIA_MTX_HOST is not configured"
        if not os.path.isfile(config.MEDIA_MTX_SSH_KEY):
            return "MediaMTX SSH key does not exist"
        return "MediaMTX credentials are incomplete"

    @property
    def peer_count(self):
        self._cleanup_expired_sessions()
        with self._lock:
            return len(self._sessions)

    @property
    def source_stats(self):
        with self._lock:
            return [
                publisher.stats for publisher in self._publishers.values()
            ]

    @property
    def tunnel_online(self):
        if self._uses_direct_whep:
            parsed = urllib.parse.urlsplit(self._whep_base_url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            host = parsed.hostname or "127.0.0.1"
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return True
            except OSError:
                return False
        return _port_open(
            config.MEDIA_MTX_LOCAL_RTSP_PORT
        ) and _port_open(config.MEDIA_MTX_LOCAL_WHEP_PORT)

    def _tunnel_command(self):
        return [
            config.MEDIA_MTX_SSH_BIN,
            "-NT",
            "-F",
            os.devnull,
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            (
                "ServerAliveInterval="
                f"{config.MEDIA_MTX_SSH_KEEPALIVE_SECONDS}"
            ),
            "-o",
            (
                "ServerAliveCountMax="
                f"{config.MEDIA_MTX_SSH_KEEPALIVE_COUNT}"
            ),
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={config.MEDIA_MTX_KNOWN_HOSTS}",
            "-i",
            config.MEDIA_MTX_SSH_KEY,
            "-L",
            (
                f"127.0.0.1:{config.MEDIA_MTX_LOCAL_RTSP_PORT}:"
                "127.0.0.1:8554"
            ),
            "-L",
            (
                f"127.0.0.1:{config.MEDIA_MTX_LOCAL_WHEP_PORT}:"
                "127.0.0.1:8889"
            ),
            f"{config.MEDIA_MTX_SSH_USER}@{config.MEDIA_MTX_HOST}",
        ]

    def ensure_tunnel(self):
        if not self.available:
            raise CloudMediaError(self.unavailable_reason)
        if self._uses_direct_whep:
            return
        with self._lock:
            process = self._tunnel_process
            if process is not None and process.poll() is not None:
                self._tunnel_process = None
                process = None
            if self.tunnel_online:
                return
            if process is None:
                creationflags = (
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                )
                process = subprocess.Popen(
                    self._tunnel_command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                self._tunnel_process = process
        deadline = (
            time.monotonic() + config.MEDIA_MTX_START_TIMEOUT_SECONDS
        )
        while time.monotonic() < deadline:
            if self.tunnel_online:
                return
            process = self._tunnel_process
            if process is not None and process.poll() is not None:
                raise CloudMediaError("MediaMTX SSH tunnel exited")
            time.sleep(0.1)
        self.invalidate_tunnel()
        raise CloudMediaError("MediaMTX SSH tunnel timed out")

    def invalidate_tunnel(self):
        """Discard a failed forwarding process so the next call reconnects."""
        if self._uses_direct_whep:
            return
        with self._lock:
            process = self._tunnel_process
            self._tunnel_process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _publisher(self, stream_id):
        with self._lock:
            publisher = self._publishers.get(stream_id)
            if publisher is None:
                publisher = NvencRtspPublisher(
                    stream_id,
                    self.ensure_tunnel,
                    self.invalidate_tunnel,
                    lambda: self.tunnel_online,
                )
                self._publishers[stream_id] = publisher
            publisher.start()
            return publisher

    def _whep_url(self, location=None, stream_id=None):
        base = self._whep_base_url
        if location:
            parsed = urllib.parse.urlsplit(location)
            path = parsed.path
            if parsed.query:
                path += f"?{parsed.query}"
            return base + path
        return f"{base}/{stream_id}/whep"

    @property
    def _uses_direct_whep(self):
        return bool(
            config.MEDIA_MTX_WHEP_BASE_URL
            or config.DEPLOYMENT_ROLE == "cloud"
        )

    @property
    def _whep_base_url(self):
        if config.MEDIA_MTX_WHEP_BASE_URL:
            return config.MEDIA_MTX_WHEP_BASE_URL
        if config.DEPLOYMENT_ROLE == "cloud":
            return "http://127.0.0.1:8889"
        return (
            f"http://127.0.0.1:{config.MEDIA_MTX_LOCAL_WHEP_PORT}"
        )

    @staticmethod
    def _authorization():
        raw = (
            f"{config.MEDIA_MTX_READ_USERNAME}:"
            f"{config.MEDIA_MTX_READ_PASSWORD}"
        ).encode()
        return "Basic " + base64.b64encode(raw).decode()

    def _whep_request(
        self,
        method,
        *,
        stream_id=None,
        location=None,
        sdp=None,
        etag=None,
    ):
        headers = {"Authorization": self._authorization()}
        data = None
        if sdp is not None:
            data = sdp.encode()
            headers["Content-Type"] = "application/sdp"
        if etag:
            headers["If-Match"] = etag
        request = urllib.request.Request(
            self._whep_url(location=location, stream_id=stream_id),
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(
            request,
            timeout=config.WEBRTC_OFFER_TIMEOUT_SECONDS,
        ) as response:
            return response.read().decode(), dict(response.headers)

    def create_answer(self, stream_id, sdp, description_type):
        if description_type != "offer":
            raise CloudMediaError("WHEP requires an SDP offer")
        self._cleanup_expired_sessions()
        with self._lock:
            if (
                len(self._sessions) + self._pending_sessions
                >= config.WEBRTC_MAX_VIEWERS
            ):
                raise CloudMediaCapacityError(
                    "WebRTC viewer limit reached "
                    f"({config.WEBRTC_MAX_VIEWERS})"
                )
            self._pending_sessions += 1
        try:
            self.ensure_tunnel()
            if config.DEPLOYMENT_ROLE != "cloud":
                publisher = self._publisher(stream_id)
                if (
                    config.MEDIA_MTX_REQUIRE_PUBLISHER_READY
                    and not publisher.wait_ready(
                        config.MEDIA_MTX_START_TIMEOUT_SECONDS
                    )
                ):
                    raise CloudMediaError(
                        publisher.last_error
                        or "MediaMTX publisher did not start"
                    )
            last_error = None
            for _ in range(10):
                try:
                    answer, headers = self._whep_request(
                        "POST",
                        stream_id=stream_id,
                        sdp=sdp,
                    )
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code not in {404, 409}:
                        raise CloudMediaError(
                            f"MediaMTX WHEP failed with HTTP {exc.code}"
                        ) from exc
                    time.sleep(0.2)
                except (urllib.error.URLError, TimeoutError, OSError):
                    self.invalidate_tunnel()
                    raise
            else:
                raise CloudMediaError(
                    f"MediaMTX stream is unavailable: {last_error}"
                )
            peer_id = uuid.uuid4().hex
            now = time.monotonic()
            with self._lock:
                self._sessions[peer_id] = {
                    "location": headers.get("Location")
                    or headers.get("location"),
                    "etag": headers.get("ETag") or headers.get("etag"),
                    "created": now,
                    "last_seen": now,
                }
            return {
                "peer_id": peer_id,
                "sdp": answer,
                "type": "answer",
            }
        finally:
            with self._lock:
                self._pending_sessions = max(
                    0, self._pending_sessions - 1
                )

    def _take_expired_sessions(self):
        cutoff = time.monotonic() - config.WEBRTC_SESSION_TIMEOUT_SECONDS
        expired = []
        with self._lock:
            for peer_id, session in list(self._sessions.items()):
                if session.get("last_seen", session.get("created", 0)) < cutoff:
                    expired.append(self._sessions.pop(peer_id))
        return expired

    def _delete_sessions_async(self, sessions):
        if not sessions:
            return

        def delete():
            for session in sessions:
                location = session.get("location")
                if not location:
                    continue
                try:
                    self._whep_request(
                        "DELETE",
                        location=location,
                        etag=session.get("etag"),
                    )
                except Exception:
                    pass

        threading.Thread(
            target=delete,
            name="mediamtx-whep-cleanup",
            daemon=True,
        ).start()

    def _cleanup_expired_sessions(self):
        expired = self._take_expired_sessions()
        if expired:
            self._delete_sessions_async(expired)

    def touch_peer(self, peer_id):
        self._cleanup_expired_sessions()
        with self._lock:
            session = self._sessions.get(peer_id)
            if session is None:
                return False
            session["last_seen"] = time.monotonic()
            return True

    def close_peer(self, peer_id):
        with self._lock:
            session = self._sessions.pop(peer_id, None)
        if session is None:
            return False
        location = session.get("location")
        if location:
            try:
                self._whep_request(
                    "DELETE",
                    location=location,
                    etag=session.get("etag"),
                )
            except Exception:
                pass
        return True

    def stop(self):
        with self._lock:
            publishers = list(self._publishers.values())
            sessions = list(self._sessions.values())
            process = self._tunnel_process
            self._tunnel_process = None
            self._publishers.clear()
            self._sessions.clear()
            self._pending_sessions = 0
        for session in sessions:
            location = session.get("location")
            if not location:
                continue
            try:
                self._whep_request(
                    "DELETE",
                    location=location,
                    etag=session.get("etag"),
                )
            except Exception:
                pass
        for publisher in publishers:
            publisher.stop()
        if process is not None:
            with self._lock:
                self._tunnel_process = process
            self.invalidate_tunnel()


cloud_media_service = CloudMediaService()
