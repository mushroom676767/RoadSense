import unittest

from backend.agent_state import AgentRegistry


class AgentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.registry = AgentRegistry(
            token="agent-secret",
            stale_seconds=6,
            clock=lambda: self.now,
        )

    def test_authentication_and_stale_agent_state(self):
        self.assertFalse(self.registry.authenticate("wrong"))
        self.assertTrue(self.registry.authenticate("agent-secret"))
        desired = self.registry.connect(
            "sid-1",
            {"agent_id": "board", "capabilities": ["srt"]},
        )
        self.assertEqual(desired["generation"], 0)
        self.assertTrue(self.registry.status()["agent_online"])
        self.now += 7
        self.assertFalse(self.registry.status()["agent_online"])
        self.assertIsNone(self.registry.active_sid)

    def test_desired_state_is_monotonic_and_latest_wins(self):
        first = self.registry.set_desired(
            mode="plate_ai",
            source="hdmi",
            running=True,
        )
        second = self.registry.set_desired(running=False)
        self.assertEqual(first["generation"], 1)
        self.assertEqual(second["generation"], 2)
        self.assertFalse(second["running"])

    def test_old_ack_cannot_replace_new_ack(self):
        self.registry.connect("sid-1", {"agent_id": "board"})
        self.assertTrue(
            self.registry.acknowledge(
                "sid-1", {"generation": 4, "success": True}
            )
        )
        self.assertFalse(
            self.registry.acknowledge(
                "sid-1", {"generation": 3, "success": False}
            )
        )
        self.assertEqual(
            self.registry.status()["last_ack"]["generation"],
            4,
        )

    def test_heartbeat_recovers_a_lost_command_acknowledgement(self):
        self.registry.connect("sid-1", {"agent_id": "windows-agent"})
        self.assertTrue(
            self.registry.touch(
                "sid-1",
                {
                    "mode": None,
                    "running": False,
                    "command_ack": {
                        "generation": 9,
                        "success": True,
                        "error": "",
                        "status": {
                            "current_mode": None,
                            "running": False,
                        },
                    },
                },
            )
        )
        acknowledgement = self.registry.status()["last_ack"]
        self.assertEqual(acknowledgement["generation"], 9)
        self.assertTrue(acknowledgement["success"])
        self.assertFalse(acknowledgement["status"]["running"])
        next_desired = self.registry.set_desired(
            mode="plate_ai", source="hdmi", running=True
        )
        self.assertEqual(next_desired["generation"], 10)

    def test_stream_status_separates_board_and_publisher(self):
        self.registry.connect("sid-1", {"agent_id": "board"})
        self.registry.touch(
            "sid-1",
            {
                "mode": "plate_ai",
                "running": True,
                "board_online": True,
                "cloud_publisher_online": False,
                "publisher_generation": 7,
                "publisher_reconnects": 2,
                "active_stream": "plate_ai",
                "encoded_frames": 42,
                "encode_ms": 1.2,
                "upload_mbps": 3.8,
                "active_transport": "srt",
            },
        )
        self.registry.update_stream(
            "sid-1",
            "plate_ai",
            {"frame_id": 9, "mode": "plate_ai", "metrics": {}},
        )
        stream = self.registry.streams()["plate_ai"]
        self.assertTrue(stream["board_online"])
        self.assertFalse(stream["cloud_publisher_online"])
        self.assertEqual(stream["publisher_generation"], 7)
        self.assertEqual(stream["active_transport"], "srt")
        self.assertEqual(stream["active_stream"], "plate_ai")
        self.assertEqual(stream["encoded_frames"], 42)

    def test_publisher_online_only_applies_to_active_stream(self):
        self.registry.connect("sid-1", {"agent_id": "board"})
        self.registry.touch(
            "sid-1",
            {
                "active_stream": "plate_ai",
                "board_online": True,
                "cloud_publisher_online": True,
            },
        )
        self.registry.update_stream("sid-1", "plate_ai", {"frame_id": 1})
        self.registry.update_stream(
            "sid-1", "people_redlight", {"frame_id": 2}
        )
        streams = self.registry.streams()
        self.assertTrue(streams["plate_ai"]["cloud_publisher_online"])
        self.assertFalse(
            streams["people_redlight"]["cloud_publisher_online"]
        )


if __name__ == "__main__":
    unittest.main()
