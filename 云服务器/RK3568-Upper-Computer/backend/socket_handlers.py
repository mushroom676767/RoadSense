from flask import request
from flask_jwt_extended import decode_token
from flask_socketio import emit

import config
from backend.ingest import get_latest
from backend.stream_hub import stream_hub


def register_socket_handlers(socketio, ssh_manager):
    @socketio.on("connect")
    def handle_connect(auth=None):
        token = ""
        if isinstance(auth, dict):
            token = str(auth.get("token") or "")
        token = token or request.args.get("token", "")
        try:
            decode_token(token)
        except Exception:
            return False
        stream_hub.connect(request.sid)
        emit("log_snapshot", {"text": ssh_manager.get_log_snapshot()})
        status = ssh_manager.get_status()
        emit(
            "mode_status",
            {
                "mode": status.get("current_mode"),
                "running": status.get("running", False),
            },
        )
        emit("connected", {"message": "已连接到服务器"})

    @socketio.on("disconnect")
    def handle_disconnect(reason=None):
        stream_hub.disconnect(request.sid)

    @socketio.on("subscribe")
    def handle_subscribe(data):
        stream_id = str((data or {}).get("stream_id", ""))
        include_jpeg = (data or {}).get("include_jpeg", True) is not False
        jpeg_fps = (data or {}).get("jpeg_fps", 5)
        if stream_id not in config.MODES:
            emit("stream_status", {"stream_id": stream_id, "error": "未知流"})
            return
        stream_hub.subscribe(
            request.sid,
            stream_id,
            include_jpeg=include_jpeg,
            jpeg_fps=jpeg_fps,
        )
        latest = get_latest(stream_id)
        emit(
            "subscribed",
            {
                "stream_id": stream_id,
                "has_frame": latest is not None,
                "include_jpeg": include_jpeg,
            },
        )
        if latest is not None:
            stream_hub.publish(stream_id, latest["meta"], latest["jpeg"])

    @socketio.on("unsubscribe")
    def handle_unsubscribe(data):
        stream_id = str((data or {}).get("stream_id", ""))
        stream_hub.unsubscribe(request.sid, stream_id)
        emit("unsubscribed", {"stream_id": stream_id})
