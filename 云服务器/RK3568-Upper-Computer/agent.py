"""Local-only RK3568 device agent entrypoint."""

import atexit

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify

import config
from backend.database import init_db
from backend.device_agent import DeviceAgent
from backend.ingest import ingest_bp


def create_agent_app():
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = config.MAX_INGEST_BYTES
    init_db()
    application.register_blueprint(ingest_bp)

    @application.get("/health")
    def health():
        return jsonify({"ok": True, "role": "agent"})

    return application


app = create_agent_app()
device_agent = DeviceAgent()
atexit.register(device_agent.stop)


if __name__ == "__main__":
    from waitress import serve

    if config.DEPLOYMENT_ROLE != "agent":
        raise SystemExit("DEPLOYMENT_ROLE must be agent")
    if not config.AGENT_CLOUD_URL or not config.AGENT_TOKEN:
        raise SystemExit("AGENT_CLOUD_URL and AGENT_TOKEN are required")
    device_agent.start()
    serve(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        threads=8,
        channel_timeout=30,
    )
