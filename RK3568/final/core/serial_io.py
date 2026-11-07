"""FPGA serial ROI protocol.

Wire format: ``0xAA, count, count * (x1, y1, x2, y2)``. ``count`` is 0..4
and coordinates are little-endian uint16 values in full-frame 1920x1080 XYXY
coordinates.  The right and bottom edges are exclusive.
"""

import struct

try:
    import serial
except ImportError:
    serial = None


SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115205
SERIAL_HEADER = 0xAA
SERIAL_MAX_ROIS = 4
SERIAL_FRAME_WIDTH = 1920
SERIAL_FRAME_HEIGHT = 1080


class SerialRoiProtocolError(ValueError):
    """A structurally complete packet contains invalid ROI geometry."""

    def __init__(self, message, raw_packet=b""):
        super().__init__(message)
        self.raw_packet = bytes(raw_packet)


def _read_exact(ser, size):
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = ser.read(remaining)
        if not chunk:
            continue
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def open_serial(port=SERIAL_PORT, baud=SERIAL_BAUD, timeout=None):
    if serial is None:
        raise ImportError("pyserial 未安装，请执行: pip3 install pyserial")
    handle = serial.Serial(port, baud, timeout=timeout)
    print(f"串口已打开: {port} @ {baud}")
    return handle


def validate_plate_rois(rois, width=SERIAL_FRAME_WIDTH,
                        height=SERIAL_FRAME_HEIGHT):
    """Return normalized ROI dictionaries or raise on invalid geometry."""
    normalized = []
    for index, roi in enumerate(rois):
        if isinstance(roi, dict):
            values = tuple(roi.get(key) for key in ("x1", "y1", "x2", "y2"))
        else:
            values = tuple(roi)
        if len(values) != 4 or any(value is None for value in values):
            raise SerialRoiProtocolError(f"ROI{index}字段不完整")
        x1, y1, x2, y2 = (int(value) for value in values)
        if not (0 <= x1 < x2 <= int(width)):
            raise SerialRoiProtocolError(
                f"ROI{index}横向坐标非法: ({x1}, {x2})"
            )
        if not (0 <= y1 < y2 <= int(height)):
            raise SerialRoiProtocolError(
                f"ROI{index}纵向坐标非法: ({y1}, {y2})"
            )
        normalized.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    if len(normalized) > SERIAL_MAX_ROIS:
        raise SerialRoiProtocolError("ROI数量超过4")
    return normalized


def read_plate_rois(ser, return_raw=False):
    """Block until one complete ROI packet has been decoded.

    Noise and invalid ``count`` bytes return to the header scan. Invalid
    geometry consumes the complete packet then raises so the runtime can clear
    stale results.
    """
    while True:
        header = ser.read(1)
        if not header or header[0] != SERIAL_HEADER:
            continue
        count_bytes = _read_exact(ser, 1)
        count = count_bytes[0]
        if count > SERIAL_MAX_ROIS:
            continue
        payload = _read_exact(ser, count * 8)
        raw_packet = bytes((SERIAL_HEADER, count)) + payload
        values = struct.unpack(f"<{count * 4}H", payload) if count else ()
        rois = [
            {
                "x1": values[offset],
                "y1": values[offset + 1],
                "x2": values[offset + 2],
                "y2": values[offset + 3],
            }
            for offset in range(0, len(values), 4)
        ]
        try:
            rois = validate_plate_rois(rois)
        except SerialRoiProtocolError as exc:
            exc.raw_packet = raw_packet
            raise
        if return_raw:
            return rois, raw_packet
        return rois
