"""Current production entrypoint and module-boundary contracts."""

import importlib
import importlib.util
import queue
import unittest
from unittest import mock

from people_det import app as people_app
from plate_det import ai_runtime, serial_runtime


REMOVED_MODULES = (
    "fpga_serial_hybrid", "core.hdmi_native", "core.postprocess",
    "plate_det.plate_detector", "plate_det.plate_draw",
    "plate_det.serial_hybrid_runtime", "people_det.people_run", "rknn_infer",
)


class ProductionContractTests(unittest.TestCase):
    def test_three_production_runtimes_expose_cli(self):
        for runtime in (ai_runtime, serial_runtime, people_app):
            with self.subTest(runtime=runtime.__name__):
                self.assertTrue(callable(runtime.build_args))
                self.assertTrue(callable(runtime.main))

    def test_removed_modules_cannot_be_imported(self):
        for module_name in REMOVED_MODULES:
            with self.subTest(module=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

    def test_packages_do_not_reexport_runtime_or_ethernet_apis(self):
        core = importlib.import_module("core")
        plate = importlib.import_module("plate_det")
        people = importlib.import_module("people_det")
        for module, names in (
            (core, ("NativeMppHdmiReceiver", "udp_receiver_process")),
            (plate, ("prepare_plate_image", "recognize_rknn")),
            (people, ("RKNNDetector", "run_hdmi_native")),
        ):
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))

    def test_ethernet_module_remains_internal_and_functional(self):
        fpga_io = importlib.import_module("core.fpga_io")
        frame = b"\xf8\x00" * (fpga_io.FRAME_W * fpga_io.FRAME_H)
        decoded = fpga_io.decode_rgb565_big_endian(frame)
        self.assertEqual(decoded.shape, (480, 640, 3))
        self.assertEqual(decoded[0, 0].tolist(), [0, 0, 248])

        payload = b"\x12\x34" * (fpga_io.FRAME_W * fpga_io.FRAME_H)
        packet = fpga_io.FRAME_HEADER + b"\x00" * 4 + payload
        fake_socket = mock.Mock()
        fake_socket.recvfrom.side_effect = [
            (packet, (fpga_io.FPGA_IP, fpga_io.FPGA_PORT)), KeyboardInterrupt(),
        ]
        frame_queue = queue.Queue(maxsize=1)
        with mock.patch.object(fpga_io.socket, "socket", return_value=fake_socket):
            fpga_io.udp_receiver_process(
                frame_queue, fpga_io.LOCAL_IP, fpga_io.RK_PORT,
                fpga_io.FPGA_IP, fpga_io.FPGA_PORT,
            )
        self.assertEqual(frame_queue.get_nowait(), payload)

    def test_removed_recognition_options_are_rejected(self):
        for option in (
            "--rec-cache-ttl", "--rec-cache-iou", "--rec-stability-mode",
            "--rec-fast-confirm", "--no-rec-fast-confirm",
            "--rec-sample-interval-ms",
        ):
            with self.subTest(entrypoint="ai", option=option):
                with self.assertRaises(SystemExit):
                    ai_runtime.build_args([option])
        with self.assertRaises(SystemExit):
            serial_runtime.build_args(["--rec-stability-mode", "off"])


if __name__ == "__main__":
    unittest.main()
