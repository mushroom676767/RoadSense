import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required
from flask_socketio import SocketIO

import config
from backend.auth import (
    auth_bp,
    bootstrap_admin,
    bootstrap_visitor,
    control_required,
)
from backend.database import init_db
from backend.events import events_bp
from backend.ingest import ingest_bp
from backend.agent_gateway import register_agent_gateway
from backend.socket_handlers import register_socket_handlers
from backend.stream_hub import stream_hub
from backend.webrtc import webrtc_bp


def create_app(test_config=None):
    frontend_dist = os.path.join(config.BASE_DIR, "frontend", "dist")
    frontend_fallback = os.path.join(config.BASE_DIR, "frontend")
    static_folder = frontend_dist if os.path.isdir(frontend_dist) else frontend_fallback
    application = Flask(
        __name__,
        static_folder=static_folder,
        static_url_path="",
    )
    application.config.update(
        JWT_SECRET_KEY=config.JWT_SECRET_KEY,
        JWT_ACCESS_TOKEN_EXPIRES=config.JWT_ACCESS_TOKEN_EXPIRES,
        JWT_TOKEN_LOCATION=["headers"],
        JWT_COOKIE_SECURE=config.JWT_COOKIE_SECURE,
        JWT_COOKIE_SAMESITE="Strict",
        JWT_COOKIE_CSRF_PROTECT=False,
        MAX_CONTENT_LENGTH=config.MAX_INGEST_BYTES,
    )
    if test_config:
        application.config.update(test_config)

    if config.CORS_ORIGINS is not None:
        CORS(
            application,
            resources={r"/api/*": {"origins": config.CORS_ORIGINS}},
        )
    JWTManager(application)
    socket = SocketIO(
        application,
        cors_allowed_origins=config.CORS_ORIGINS,
        async_mode="threading",
        logger=False,
        engineio_logger=False,
        max_http_buffer_size=config.MAX_INGEST_BYTES,
    )
    stream_hub.init(socket)
    if config.DEPLOYMENT_ROLE == "cloud":
        from backend.remote_control import RemoteAgentControl

        control = RemoteAgentControl(socket)
        register_agent_gateway(socket)
    else:
        from backend.ssh_manager import SSHManager

        control = SSHManager(socket)

    init_db()
    bootstrap_admin()
    bootstrap_visitor()
    application.register_blueprint(auth_bp)
    application.register_blueprint(ingest_bp)
    application.register_blueprint(events_bp)
    application.register_blueprint(webrtc_bp)
    register_socket_handlers(socket, control)

    @application.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'",
        )
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        elif response.mimetype == "text/html":
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        if request.headers.get("X-Forwarded-Proto", "").lower() == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response

    @application.get("/")
    def index():
        return send_from_directory(static_folder, "index.html")

    @application.get("/<path:path>")
    def frontend_route(path):
        target = os.path.join(static_folder, path)
        if os.path.isfile(target):
            return send_from_directory(static_folder, path)
        return send_from_directory(static_folder, "index.html")

    @application.post("/api/mode/start")
    @control_required()
    def start_mode():
        data = request.get_json(silent=True) or {}
        mode_id = str(data.get("mode") or "")
        source = str(data.get("source") or "hdmi")
        result = control.start_mode(mode_id, source=source)
        return jsonify(result), 400 if "error" in result else 200

    @application.post("/api/mode/stop")
    @control_required()
    def stop_mode():
        result = control.stop_mode()
        return jsonify(result), 400 if "error" in result else 200

    @application.get("/api/mode/status")
    @jwt_required()
    def mode_status():
        return jsonify(control.get_status())

    @application.get("/api/modes")
    @jwt_required()
    def list_modes():
        public_modes = {
            key: {
                "name": value["name"],
                "desc": value["desc"],
                "accepts_source": value.get("accepts_source", False),
                "default_source": value.get("default_source", "hdmi"),
            }
            for key, value in config.MODES.items()
        }
        return jsonify(public_modes)

    @application.get("/api/sources")
    @jwt_required()
    def list_sources():
        return jsonify(
            {key: {"name": value["name"]} for key, value in config.SOURCES.items()}
        )

    @application.get("/api/config")
    def public_config():
        return jsonify({"allow_register": config.ALLOW_REGISTER})

    @application.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "请求体过大"}), 413

    application.extensions["socketio"] = socket
    application.extensions["ssh_manager"] = control
    application.extensions["control_manager"] = control
    return application, socket


app, socketio = create_app()


if __name__ == "__main__":
    from backend.single_instance import acquire_single_instance

    if not acquire_single_instance(config.BASE_DIR):
        raise SystemExit(
            "RK3568管理平台已有一个运行实例，请勿重复启动app.py。"
        )
    print("=" * 60)
    print("RK3568 三模式智能交通管理平台")
    print(f"访问地址: http://localhost:{config.SERVER_PORT}")
    print("=" * 60)
    socketio.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
