import queue

# ==========================================
# HDMI 采集卡配置
# ==========================================
HDMI_DEVICE = '/dev/video9'
# 注意: 此 UVC 摄像头 (LRCP 1080P-60) 的 MJPG 仅支持 60fps 一种间隔,
# 请求 30fps 会被忽略并恒定输出 60fps。故默认请求值对齐为 60,避免
# 帧率协商警告与 30/60 不一致。YUYV 才有低帧率但 1080p 仅 5fps,不用于实时。
HDMI_WIDTH = 1920
HDMI_HEIGHT = 1080
HDMI_FPS = 60
HDMI_FOURCC = 'MJPG'


def put_latest_frame(q, frame):
    """向容量为1的队列发布帧，并在需要时丢弃更旧帧。"""
    dropped = 0
    while True:
        try:
            q.put_nowait(frame)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                continue
