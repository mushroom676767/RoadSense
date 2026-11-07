"""实时入口共享的线程安全原语。

模块只定义队列和所有权规则，不包含车型、车牌或行人业务：

- latest-wins队列永不对采集线程施加反压；
- ``OwnedLatestFrame``拥有一个Native帧引用，借出时额外retain；
- ``LatestValue``保存不含图像所有权的最新结果，并按单调时钟执行TTL；
- ``native_display_worker``持有已提交DRM但尚未退休的帧，Page Flip返回token后释放。
"""

from __future__ import annotations

import queue
import threading
import time


def release_frame(value):
    """释放帧或任务字典中的帧；无release方法的普通值保持不变。"""
    frame = value.get("frame") if isinstance(value, dict) else value
    release = getattr(frame, "release", None)
    if callable(release):
        release()


def put_latest(target_queue, value, release_dropped=False):
    """无背压发布最新任务，并按需释放被替换任务持有的帧。"""
    dropped = 0
    while True:
        try:
            target_queue.put_nowait(value)
            return dropped
        except queue.Full:
            try:
                stale = target_queue.get_nowait()
            except queue.Empty:
                continue
            dropped += 1
            if release_dropped:
                release_frame(stale)


def drain_latest(target_queue):
    """取走latest-wins队列中残留的全部项目，并按回调释放其帧所有权。"""
    latest = None
    while True:
        try:
            latest = target_queue.get_nowait()
        except queue.Empty:
            return latest


class OwnedLatestFrame:
    """Own one native frame reference and lend retained references to workers."""

    def __init__(self):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self._lock = threading.Lock()
        self._frame = None

    def publish(self, frame):
        """用新对象替换槽内旧对象；槽位负责释放被替换对象所持有的帧引用。"""
        with self._lock:
            stale = self._frame
            self._frame = frame
        if stale is not None:
            stale.release()

    def acquire(self):
        """返回最新对象的安全借用副本；Native帧会先retain再交给调用方。"""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.retain()

    def clear(self):
        """清空最新帧槽并释放槽位持有的Native引用。"""
        with self._lock:
            stale = self._frame
            self._frame = None
        if stale is not None:
            stale.release()


class LatestValue:
    """带锁的单槽结果；值本身不转移Native帧所有权。"""
    def __init__(self, value=None):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self._lock = threading.Lock()
        self._value = value

    def set(self, value):
        """在线程锁保护下更新最新业务值及其单调时钟时间戳。"""
        with self._lock:
            self._value = value

    def get(self):
        """在线程锁保护下读取最新业务值和更新时间。"""
        with self._lock:
            return self._value

    def get_fresh(self, ttl_ms, now=None):
        """只返回TTL内的最新业务值，超时结果视为不存在。"""
        value = self.get()
        if value is None or ttl_ms <= 0:
            return value
        timestamp = value.get("_ts", 0.0)
        current = time.monotonic() if now is None else float(now)
        return value if current - timestamp <= ttl_ms / 1000.0 else None


def native_display_worker(display, frame_queue, stop_event, error_queue=None,
                          counters=None):
    """持有队列NV12引用，直到KMS明确退休对应token。

    ``display.show``成功后帧进入``inflight``，不能立即释放；否则V4L2/MPP可能在
    显示控制器仍扫描时覆盖DMA-BUF。异常和退出路径会释放队列及inflight全部引用。
    """
    inflight = {}
    current_frame = None
    try:
        while not stop_event.is_set():
            if current_frame is None:
                try:
                    frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            else:
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    # Keep submitting the current token.  display.show()
                    # waits for the previous Page Flip, so this repeats the
                    # latest video frame at the vblank cadence even when a
                    # Python producer is briefly delayed.
                    frame = current_frame
            try:
                retired_token = display.show(frame)
            except Exception:
                if frame is not current_frame:
                    frame.release()
                raise
            inflight[int(frame.token)] = frame
            current_frame = frame
            if retired_token:
                retired = inflight.pop(int(retired_token), None)
                if retired is not None:
                    retired.release()
            if counters is not None:
                counters["displayed"] = counters.get("displayed", 0) + 1
    except Exception as exc:
        if error_queue is not None:
            put_latest(error_queue, exc)
        stop_event.set()
    finally:
        while True:
            try:
                release_frame(frame_queue.get_nowait())
            except queue.Empty:
                break
        for frame in inflight.values():
            frame.release()
