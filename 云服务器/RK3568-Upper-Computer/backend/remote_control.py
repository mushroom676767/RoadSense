"""Cloud-side mode controller backed by the connected device agent."""

import config
from backend.agent_gateway import AGENT_NAMESPACE
from backend.agent_state import agent_registry


class RemoteAgentControl:
    def __init__(self, socketio):
        self.socketio = socketio

    def _dispatch(self, desired):
        sid = agent_registry.active_sid
        if not sid:
            return False
        self.socketio.emit(
            "desired_state",
            desired,
            to=sid,
            namespace=AGENT_NAMESPACE,
        )
        return True

    def start_mode(self, mode_id, source="hdmi"):
        if mode_id not in config.MODES:
            return {"error": f"unknown mode: {mode_id}"}
        desired = agent_registry.set_desired(
            mode=mode_id,
            source=source,
            running=True,
        )
        dispatched = self._dispatch(desired)
        return {
            "success": True,
            "pending": True,
            "agent_online": dispatched,
            "mode": mode_id,
            "generation": desired["generation"],
            "name": config.MODES[mode_id]["name"],
        }

    def stop_mode(self):
        previous = agent_registry.status().get("current_mode")
        desired = agent_registry.set_desired(running=False)
        dispatched = self._dispatch(desired)
        return {
            "success": True,
            "pending": True,
            "agent_online": dispatched,
            "stopped": previous,
            "generation": desired["generation"],
        }

    def get_status(self):
        return agent_registry.status()

    def get_log_snapshot(self):
        return agent_registry.log_snapshot()

    def close(self):
        return None
