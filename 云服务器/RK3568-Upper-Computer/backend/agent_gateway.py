"""Socket.IO namespace used by the outbound Windows device agent."""

import time

from flask import request
from flask_socketio import emit

from backend.agent_state import agent_registry
from backend.events import save_external_event
from backend.stream_hub import stream_hub


AGENT_NAMESPACE = "/agent"


def register_agent_gateway(socketio):
    @socketio.on("connect", namespace=AGENT_NAMESPACE)
    def agent_connect(auth=None):
        auth = auth if isinstance(auth, dict) else {}
        if not agent_registry.authenticate(auth.get("token")):
            return False
        desired = agent_registry.connect(request.sid, auth)
        emit(
            "desired_state",
            desired,
            namespace=AGENT_NAMESPACE,
        )

    @socketio.on("disconnect", namespace=AGENT_NAMESPACE)
    def agent_disconnect(reason=None):
        agent_registry.disconnect(request.sid)
        socketio.emit(
            "agent_status",
            agent_registry.status(),
        )

    @socketio.on("heartbeat", namespace=AGENT_NAMESPACE)
    def agent_heartbeat(payload):
        if not agent_registry.touch(request.sid, payload):
            return {"ok": False}
        status = agent_registry.status()
        socketio.emit("agent_status", status)
        return {"ok": True, "desired": status["desired"]}

    @socketio.on("frame_meta", namespace=AGENT_NAMESPACE)
    def agent_frame_meta(payload):
        payload = payload if isinstance(payload, dict) else {}
        stream_id = str(payload.get("stream_id") or "")
        meta = payload.get("meta")
        if not agent_registry.update_stream(
            request.sid,
            stream_id,
            meta,
        ):
            return {"ok": False}
        stream_hub.publish(stream_id, meta, b"")
        return {"ok": True}

    @socketio.on("event", namespace=AGENT_NAMESPACE)
    def agent_event(payload):
        payload = payload if isinstance(payload, dict) else {}
        event = payload.get("event")
        jpeg = payload.get("jpeg")
        status = agent_registry.status()
        if request.sid != agent_registry.active_sid:
            return {"ok": False, "error": "agent is not active"}
        try:
            event_id = save_external_event(
                status.get("agent_id"),
                event,
                jpeg,
            )
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:
            return {"ok": False, "error": "event archive write failed"}
        return {
            "ok": True,
            "event_id": event_id,
            "external_event_id": event.get("external_event_id"),
            "received_at_ms": int(time.time() * 1000),
        }

    @socketio.on("log_output", namespace=AGENT_NAMESPACE)
    def agent_log(payload):
        payload = payload if isinstance(payload, dict) else {}
        text = str(payload.get("text") or "")
        if not agent_registry.append_log(request.sid, text):
            return {"ok": False}
        socketio.emit("log_output", {"text": text})
        return {"ok": True}

    @socketio.on("command_ack", namespace=AGENT_NAMESPACE)
    def agent_command_ack(payload):
        accepted = agent_registry.acknowledge(request.sid, payload)
        socketio.emit("mode_status", agent_registry.status())
        return {"ok": accepted}
