import struct
import unittest

from core.serial_io import (
    SerialRoiProtocolError,
    read_plate_rois,
    validate_plate_rois,
)


class ChunkedSerial:
    def __init__(self, chunks):
        self.chunks = [bytearray(chunk) for chunk in chunks]

    def read(self, size):
        while self.chunks and not self.chunks[0]:
            self.chunks.pop(0)
        if not self.chunks:
            raise EOFError("test serial exhausted")
        take = min(int(size), len(self.chunks[0]))
        value = bytes(self.chunks[0][:take])
        del self.chunks[0][:take]
        return value


def packet(*rois):
    payload = b"".join(struct.pack("<4H", *roi) for roi in rois)
    return bytes((0xAA, len(rois))) + payload


class SerialRoiProtocolTests(unittest.TestCase):
    def test_count_zero_is_a_complete_packet(self):
        rois, raw = read_plate_rois(ChunkedSerial([b"\xaa", b"\x00"]), True)
        self.assertEqual(rois, [])
        self.assertEqual(raw, b"\xaa\x00")

    def test_short_reads_and_little_endian_rois(self):
        raw = packet((100, 200, 401, 301), (0, 0, 1920, 1080))
        serial = ChunkedSerial([b"noise", raw[:1], raw[1:3], raw[3:9], raw[9:]])
        rois, received = read_plate_rois(serial, True)
        self.assertEqual(received, raw)
        self.assertEqual(rois[0], {"x1": 100, "y1": 200, "x2": 401, "y2": 301})
        self.assertEqual(rois[1]["x2"], 1920)

    def test_invalid_count_resynchronizes_at_next_header(self):
        valid = packet((1, 2, 3, 4))
        serial = ChunkedSerial([bytes((0xAA, 5)), valid])
        self.assertEqual(read_plate_rois(serial)[0]["x1"], 1)

    def test_all_four_rois_are_preserved_in_wire_order(self):
        raw = packet((1, 1, 2, 2), (3, 3, 4, 4),
                     (5, 5, 6, 6), (7, 7, 8, 8))
        self.assertEqual(
            [roi["x1"] for roi in read_plate_rois(ChunkedSerial([raw]))],
            [1, 3, 5, 7],
        )

    def test_degenerate_or_out_of_bounds_geometry_rejects_whole_packet(self):
        for roi in ((10, 10, 10, 20), (10, 10, 20, 10),
                    (0, 0, 1921, 10), (0, 0, 10, 1081)):
            with self.subTest(roi=roi), self.assertRaises(SerialRoiProtocolError):
                read_plate_rois(ChunkedSerial([packet(roi)]))

    def test_validator_rejects_more_than_four_rois(self):
        with self.assertRaises(SerialRoiProtocolError):
            validate_plate_rois([(0, 0, 1, 1)] * 5)


if __name__ == "__main__":
    unittest.main()
