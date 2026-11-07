import os
from datetime import timedelta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_even_int(name, default, minimum=None, maximum=None):
    value = _env_int(name, default, minimum, maximum)
    return value if value % 2 == 0 else value - 1


# Web/authentication
JWT_SECRET_KEY = os.environ.get(
    "JWT_SECRET_KEY", "rk3568-manager-development-key-change-me"
)
JWT_ACCESS_TOKEN_EXPIRES = timedelta(
    seconds=_env_int("JWT_ACCESS_TOKEN_EXPIRES", 86400, 300)
)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
VISITOR_USERNAME = os.environ.get("VISITOR_USERNAME", "").strip()
VISITOR_PASSWORD = os.environ.get("VISITOR_PASSWORD", "")
ALLOW_REGISTER = _env_bool("ALLOW_REGISTER", False)
_cors_value = os.environ.get("CORS_ORIGINS", "same-origin").strip()
CORS_ORIGINS = (
    None
    if not _cors_value or _cors_value.lower() == "same-origin"
    else [
        origin.strip()
        for origin in _cors_value.split(",")
        if origin.strip()
    ]
)
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = _env_int("SERVER_PORT", 5000, 1, 65535)
DEPLOYMENT_ROLE = os.environ.get(
    "DEPLOYMENT_ROLE", "standalone"
).strip().lower()
if DEPLOYMENT_ROLE not in {"cloud", "agent", "standalone"}:
    DEPLOYMENT_ROLE = "standalone"
JWT_COOKIE_SECURE = _env_bool(
    "JWT_COOKIE_SECURE",
    DEPLOYMENT_ROLE == "cloud",
)

# Cloud-to-device agent channel. The agent always initiates the connection.
AGENT_ID = os.environ.get("AGENT_ID", "rk3568-primary").strip()
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")
AGENT_CLOUD_URL = os.environ.get("AGENT_CLOUD_URL", "").rstrip("/")
AGENT_STALE_SECONDS = _env_int("AGENT_STALE_SECONDS", 6, 3, 60)
AGENT_RECONNECT_MAX_SECONDS = _env_int(
    "AGENT_RECONNECT_MAX_SECONDS", 10, 2, 60
)

# Persistent data
DATABASE_PATH = os.environ.get("DATABASE_PATH") or os.path.join(
    BASE_DIR, "db", "manager.db"
)
EVENT_SNAPSHOT_DIR = os.environ.get("EVENT_SNAPSHOT_DIR") or os.path.join(
    BASE_DIR, "db", "events"
)
EVENT_RETENTION_DAYS = _env_int("EVENT_RETENTION_DAYS", 30, 1)
EVENT_MAX_COUNT = _env_int("EVENT_MAX_COUNT", 10000, 100)
EVENT_COOLDOWN_SECONDS = _env_int("EVENT_COOLDOWN_SECONDS", 10, 1)
PLATE_STABLE_FRAMES = _env_int("PLATE_STABLE_FRAMES", 3, 1, 30)

# RK3568 connection. Credentials are server-side only.
SSH_HOST = os.environ.get("SSH_HOST", "192.168.137.216")
SSH_PORT = _env_int("SSH_PORT", 22, 1, 65535)
SSH_USER = os.environ.get("SSH_USER", "cat")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "")
SSH_CONNECT_TIMEOUT = _env_int("SSH_CONNECT_TIMEOUT", 10, 1, 60)
SSH_STOP_TIMEOUT = _env_int("SSH_STOP_TIMEOUT", 5, 1, 30)
SSH_START_CONFIRM_SECONDS = _env_int(
    "SSH_START_CONFIRM_SECONDS", 1, 1, 10
)
SSH_RESTART_ATTEMPTS = _env_int("SSH_RESTART_ATTEMPTS", 3, 0, 10)
SSH_RESTART_DELAY_SECONDS = _env_int(
    "SSH_RESTART_DELAY_SECONDS", 2, 1, 30
)
RK_WORK_DIR = os.environ.get("RK_WORK_DIR", "/home/cat/RK3568/final")

# Board-to-manager publish settings.
PUBLISH_TOKEN = os.environ.get("PUBLISH_TOKEN", "")
PUBLISH_URL = os.environ.get("PUBLISH_URL", "")
PUBLISH_FPS = _env_int("PUBLISH_FPS", 30, 1, 60)
PUBLISH_QUALITY = _env_int("PUBLISH_QUALITY", 50, 20, 95)
MAX_INGEST_BYTES = _env_int("MAX_INGEST_BYTES", 16 * 1024 * 1024, 1024)
MAX_INGEST_JSON_BYTES = _env_int(
    "MAX_INGEST_JSON_BYTES", 256 * 1024, 1024, MAX_INGEST_BYTES
)
STREAM_STALE_SECONDS = _env_int("STREAM_STALE_SECONDS", 3, 1, 60)

# Browser video transport. RK3568 publishing remains on direct Ethernet.
VIDEO_TRANSPORT = os.environ.get("VIDEO_TRANSPORT", "webrtc").strip().lower()
if VIDEO_TRANSPORT not in {"webrtc", "jpeg"}:
    VIDEO_TRANSPORT = "webrtc"
WEBRTC_VIDEO_FPS = _env_int("WEBRTC_VIDEO_FPS", 30, 1, 60)
WEBRTC_VIDEO_WIDTH = _env_even_int(
    "WEBRTC_VIDEO_WIDTH", 1920, 160, 3840
)
WEBRTC_VIDEO_HEIGHT = _env_even_int(
    "WEBRTC_VIDEO_HEIGHT", 1080, 120, 2160
)
WEBRTC_VIDEO_MAX_BITRATE = _env_int(
    "WEBRTC_VIDEO_MAX_BITRATE", 5_000_000, 500_000, 50_000_000
)
WEBRTC_VIDEO_BITRATE = _env_int(
    "WEBRTC_VIDEO_BITRATE",
    4_000_000,
    500_000,
    WEBRTC_VIDEO_MAX_BITRATE,
)
WEBRTC_VIDEO_ENCODER = os.environ.get(
    "WEBRTC_VIDEO_ENCODER", "h264_nvenc"
).strip().lower()
if WEBRTC_VIDEO_ENCODER not in {"h264_nvenc", "libx264"}:
    WEBRTC_VIDEO_ENCODER = "h264_nvenc"
WEBRTC_MAX_VIEWERS = _env_int("WEBRTC_MAX_VIEWERS", 12, 1, 100)
WEBRTC_PEER_QUEUE_FRAMES = _env_int(
    "WEBRTC_PEER_QUEUE_FRAMES", 60, WEBRTC_VIDEO_FPS, 300
)
WEBRTC_PROVIDER = os.environ.get(
    "WEBRTC_PROVIDER", "local"
).strip().lower()
if WEBRTC_PROVIDER not in {"mediamtx", "local"}:
    WEBRTC_PROVIDER = "local"
WEBRTC_STUN_URLS = os.environ.get(
    "WEBRTC_STUN_URLS",
    "stun:stun.cloudflare.com:3478,stun:stun.l.google.com:19302",
)
WEBRTC_TURN_URLS = os.environ.get("WEBRTC_TURN_URLS", "")
WEBRTC_TURN_USERNAME = os.environ.get("WEBRTC_TURN_USERNAME", "")
WEBRTC_TURN_CREDENTIAL = os.environ.get("WEBRTC_TURN_CREDENTIAL", "")
WEBRTC_OFFER_TIMEOUT_SECONDS = _env_int(
    "WEBRTC_OFFER_TIMEOUT_SECONDS", 6, 3, 60
)
WEBRTC_SOURCE_GRACE_SECONDS = _env_int(
    "WEBRTC_SOURCE_GRACE_SECONDS", 12, STREAM_STALE_SECONDS, 60
)
WEBRTC_SESSION_TIMEOUT_SECONDS = _env_int(
    "WEBRTC_SESSION_TIMEOUT_SECONDS", 60, 20, 600
)

# Cloud SFU. RTSP and WHEP are carried through an SSH tunnel; browser media
# goes directly to the MediaMTX ICE listener.
MEDIA_MTX_HOST = os.environ.get("MEDIA_MTX_HOST", "").strip()
MEDIA_MTX_SSH_USER = os.environ.get("MEDIA_MTX_SSH_USER", "root").strip()
MEDIA_MTX_SSH_BIN = os.environ.get("MEDIA_MTX_SSH_BIN", "ssh").strip()
MEDIA_MTX_SSH_KEY = os.environ.get("MEDIA_MTX_SSH_KEY") or os.path.join(
    BASE_DIR, "runtime", "mediamtx_ed25519"
)
MEDIA_MTX_KNOWN_HOSTS = (
    os.environ.get("MEDIA_MTX_KNOWN_HOSTS")
    or os.path.join(BASE_DIR, "runtime", "mediamtx_known_hosts")
)
if not os.path.isabs(MEDIA_MTX_SSH_KEY):
    MEDIA_MTX_SSH_KEY = os.path.join(BASE_DIR, MEDIA_MTX_SSH_KEY)
if not os.path.isabs(MEDIA_MTX_KNOWN_HOSTS):
    MEDIA_MTX_KNOWN_HOSTS = os.path.join(
        BASE_DIR, MEDIA_MTX_KNOWN_HOSTS
    )
MEDIA_MTX_LOCAL_RTSP_PORT = _env_int(
    "MEDIA_MTX_LOCAL_RTSP_PORT", 18554, 1024, 65535
)
MEDIA_MTX_LOCAL_WHEP_PORT = _env_int(
    "MEDIA_MTX_LOCAL_WHEP_PORT", 18889, 1024, 65535
)
MEDIA_MTX_PUBLISH_USERNAME = os.environ.get(
    "MEDIA_MTX_PUBLISH_USERNAME", ""
)
MEDIA_MTX_PUBLISH_PASSWORD = os.environ.get(
    "MEDIA_MTX_PUBLISH_PASSWORD", ""
)
MEDIA_MTX_READ_USERNAME = os.environ.get("MEDIA_MTX_READ_USERNAME", "")
MEDIA_MTX_READ_PASSWORD = os.environ.get("MEDIA_MTX_READ_PASSWORD", "")
MEDIA_MTX_START_TIMEOUT_SECONDS = _env_int(
    "MEDIA_MTX_START_TIMEOUT_SECONDS", 10, 3, 30
)
MEDIA_MTX_SSH_KEEPALIVE_SECONDS = _env_int(
    "MEDIA_MTX_SSH_KEEPALIVE_SECONDS", 5, 1, 30
)
MEDIA_MTX_SSH_KEEPALIVE_COUNT = _env_int(
    "MEDIA_MTX_SSH_KEEPALIVE_COUNT", 3, 1, 10
)
MEDIA_MTX_INGEST_TRANSPORT = os.environ.get(
    "MEDIA_MTX_INGEST_TRANSPORT", "legacy-rtsp-ssh"
).strip().lower()
if MEDIA_MTX_INGEST_TRANSPORT not in {"srt", "legacy-rtsp-ssh"}:
    MEDIA_MTX_INGEST_TRANSPORT = "legacy-rtsp-ssh"
MEDIA_MTX_SRT_HOST = os.environ.get(
    "MEDIA_MTX_SRT_HOST", MEDIA_MTX_HOST
).strip()
MEDIA_MTX_SRT_PORT = _env_int("MEDIA_MTX_SRT_PORT", 8890, 1, 65535)
MEDIA_MTX_SRT_LATENCY_MS = _env_int(
    "MEDIA_MTX_SRT_LATENCY_MS", 120, 80, 2000
)
MEDIA_MTX_SRT_PASSPHRASE = os.environ.get(
    "MEDIA_MTX_SRT_PASSPHRASE", ""
)
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg"
MEDIA_MTX_WHEP_BASE_URL = os.environ.get(
    "MEDIA_MTX_WHEP_BASE_URL", ""
).rstrip("/")
MEDIA_MTX_REQUIRE_PUBLISHER_READY = _env_bool(
    "MEDIA_MTX_REQUIRE_PUBLISHER_READY",
    DEPLOYMENT_ROLE == "standalone",
)
HLS_FALLBACK_BASE_URL = os.environ.get(
    "HLS_FALLBACK_BASE_URL", "/media/hls"
).rstrip("/")
SHARED_LOG_MAX_CHARS = _env_int(
    "SHARED_LOG_MAX_CHARS", 80000, 1000, 1000000
)

SOURCES = {
    "udp": {"name": "以太网（UDP）", "argv": ["--source", "udp"]},
    "hdmi": {"name": "HDMI采集卡", "argv": ["--source", "hdmi"]},
}

MODES = {
    "plate_serial": {
        "name": "串口车牌识别",
        "desc": "FPGA候选ROI + RK YOLO26 Pose角点检测 + Native OCR",
        "argv": [
            "sudo", "-E", "python3", "-u", "fpga_serial.py",
            "--result-log-interval", "60",
        ],
        "accepts_source": False,
        "publish": True,
    },
    "plate_ai": {
        "name": "AI车牌检测识别",
        "desc": "RK3568 NPU完成车牌检测与文字识别",
        "argv": [
            "python3", "-u", "fpga_ai.py",
            "--result-log-interval", "60",
        ],
        "accepts_source": False,
        "publish": True,
    },
    "people_redlight": {
        "name": "行人闯红灯检测",
        "desc": "行人、信号灯及闯红灯事件检测",
        "argv": [
            "python3", "-u", "people_redlight.py",
            "--result-log-interval", "60",
        ],
        "accepts_source": False,
        "publish": True,
    },
}
