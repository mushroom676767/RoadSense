"""FPGA 以太网 UDP 帧协议的内部实现。

该模块保留原有触发、组帧和 RGB565 解码能力，供以后在运行时内部接线。
当前实时入口不会导入或启动它，``core`` 包也不会重导出这些符号，因此加载
应用不会绑定端口、发送触发包或暴露输入源选项。
"""

import queue
import socket

import cv2
import numpy as np


FPGA_IP = "192.168.1.10"
FPGA_PORT = 61440
RK_PORT = 61440
LOCAL_IP = "192.168.1.102"
FRAME_HEADER = b"\xf0\x5a\xa5\x0f"
FRAME_W = 640
FRAME_H = 480
FRAME_BYTES = FRAME_W * FRAME_H * 2
HEADER_OFFSET = 8


def udp_receiver_process(q, local_ip, local_port, fpga_ip, fpga_port):
    """阻塞接收 UDP 数据、按帧头组帧，并以 latest-wins 方式写入队列。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((local_ip, local_port))
    sock.setblocking(True)

    print(f"接收进程启动，向 FPGA ({fpga_ip}:{fpga_port}) 发送触发信号...")
    sock.sendto(b"\x31", (fpga_ip, fpga_port))
    data_buffer = bytearray()

    try:
        while True:
            data, _peer = sock.recvfrom(65536)
            data_buffer.extend(data)

            idx = data_buffer.find(FRAME_HEADER)
            while idx != -1:
                frame_end = idx + HEADER_OFFSET + FRAME_BYTES
                if len(data_buffer) < frame_end:
                    if idx > 0:
                        del data_buffer[:idx]
                    break

                frame_bytes = bytes(
                    data_buffer[idx + HEADER_OFFSET:frame_end]
                )
                del data_buffer[:frame_end]
                try:
                    q.put_nowait(frame_bytes)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(frame_bytes)
                    except queue.Full:
                        pass
                idx = data_buffer.find(FRAME_HEADER)

            if idx == -1 and len(data_buffer) > FRAME_BYTES * 2:
                data_buffer.clear()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def decode_rgb565_big_endian(frame_bytes):
    """将一帧固定尺寸的大端 RGB565 数据解码为连续 BGR 图像。"""
    if len(frame_bytes) != FRAME_BYTES:
        raise ValueError(
            f"RGB565 frame size mismatch: expected {FRAME_BYTES}, "
            f"got {len(frame_bytes)}"
        )
    words = np.frombuffer(frame_bytes, dtype=">u2").astype(
        np.uint16, copy=False
    )
    packed = words.view(np.uint8).reshape((FRAME_H, FRAME_W, 2))
    return np.ascontiguousarray(cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR))
