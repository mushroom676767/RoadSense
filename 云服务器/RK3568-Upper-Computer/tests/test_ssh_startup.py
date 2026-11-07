import unittest
from unittest import mock

from backend.ssh_manager import SSHManager


class _Emitter:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


class _ExitedChannel:
    closed = False

    def __init__(self, stderr=b"error: unrecognized arguments: --source hdmi\n"):
        self.stderr = [stderr]
        self.stdout = []
        self.command = ""

    def get_pty(self, **_kwargs):
        return None

    def exec_command(self, command):
        self.command = command

    def recv_ready(self):
        return bool(self.stdout)

    def recv(self, _size):
        return self.stdout.pop(0)

    def recv_stderr_ready(self):
        return bool(self.stderr)

    def recv_stderr(self, _size):
        return self.stderr.pop(0)

    @staticmethod
    def exit_status_ready():
        return True

    @staticmethod
    def recv_exit_status():
        return 2

    def close(self):
        self.closed = True


class SshStartupTests(unittest.TestCase):
    def test_people_command_does_not_pass_removed_source_argument(self):
        manager = SSHManager(_Emitter())
        command, display, source_name = manager._build_command(
            "people_redlight", "hdmi"
        )
        self.assertNotIn("--source", command)
        self.assertNotIn("--source", display)
        self.assertEqual(source_name, "HDMI采集卡")

    def test_fast_argument_failure_returns_error_without_starting_reader(self):
        manager = SSHManager(_Emitter())
        channel = _ExitedChannel()
        transport = mock.Mock()
        transport.open_session.return_value = channel
        manager.client = mock.Mock()
        manager.client.get_transport.return_value = transport
        manager._connect_locked = mock.Mock()
        manager._cleanup_remote_processes_locked = mock.Mock()

        result = manager._start_locked(
            "people_redlight", "hdmi", restart_attempt=0
        )

        self.assertIn("error", result)
        self.assertIn("unrecognized arguments", result["error"])
        self.assertFalse(manager.running)
        self.assertIsNone(manager.channel)
        self.assertTrue(channel.closed)


if __name__ == "__main__":
    unittest.main()
