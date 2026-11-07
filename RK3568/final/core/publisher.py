"""Non-blocking latest-wins publisher for the RK3568 manager."""

import http.client
import json
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit


def add_publish_args(parser):
    """执行“add publish args”流程，保持原有输入输出和副作用不变。"""
    parser.add_argument(
        "--publish",
        default="",
        metavar="URL",
        help="将视频和识别元数据发布到上位机；默认关闭",
    )
    parser.add_argument("--publish-token", default="")
    parser.add_argument("--publish-stream", default="")
    parser.add_argument("--publish-fps", type=int, default=30)
    parser.add_argument("--publish-quality", type=int, default=50)
    return parser


def validate_publish_args(args):
    """校验publish args并报告不满足的约束。"""
    if not getattr(args, "publish", ""):
        return
    if not getattr(args, "publish_token", ""):
        raise ValueError("启用--publish时必须提供--publish-token")
    if not getattr(args, "publish_stream", ""):
        raise ValueError("启用--publish时必须提供--publish-stream")
    if not 1 <= int(args.publish_fps) <= 60:
        raise ValueError("publish-fps必须在1到60之间")
    if not 20 <= int(args.publish_quality) <= 95:
        raise ValueError("publish-quality必须在20到95之间")
    parsed = urlsplit(args.publish)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("publish必须是有效的HTTP或HTTPS地址")


def _builtin(value):
    """实现模块内部的“builtin”辅助逻辑，不作为新的外部接口。"""
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return _builtin(value.tolist())
    if hasattr(value, "item"):
        return _builtin(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def plate_overlays(results, source):
    """执行“plate overlays”流程，保持原有输入输出和副作用不变。"""
    overlays = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        bbox = result.get("bbox")
        landmarks = result.get("landmarks")
        item = {
            "type": "plate",
            "source": source,
            "label": str(result.get("plate_no") or ""),
            "score": float(
                result.get("detect_conf")
                or result.get("conf")
                or result.get("score")
                or 0
            ),
            "rec_score": float(result.get("rec_conf") or 0),
            "color": "#ef4444",
        }
        if bbox is not None:
            item["bbox"] = _builtin(bbox)
        if landmarks is not None:
            item["landmarks"] = _builtin(landmarks)
        overlays.append(item)
    return overlays


def people_overlays(items, class_names, violation_ids):
    """执行“people overlays”流程，保持原有输入输出和副作用不变。"""
    overlays = []
    violation_ids = set(violation_ids or ())
    for item in items or []:
        class_id = int(item.get("class_id", -1))
        violation = class_id in violation_ids
        label = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        overlays.append(
            {
                "type": "redlight_violation" if violation else "person",
                "source": "rk",
                "bbox": _builtin(item.get("bbox", [])),
                "label": label,
                "class_id": class_id,
                "score": float(item.get("score") or 0),
                "status": "violation" if violation else "normal",
                "color": "#ff4d5e" if violation else "#47e59b",
            }
        )
    return overlays


@dataclass
class PublisherStats:
    sent: int = 0
    failures: int = 0
    encoded: int = 0
    last_latency_ms: float = 0.0
    last_error: str = ""


class FramePublisher:
    def __init__(
        self,
        url,
        token,
        stream_id,
        mode,
        fps=30,
        quality=50,
        jpeg_provider=None,
        printer=print,
    ):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.url = str(url)
        self.token = str(token)
        self.stream_id = str(stream_id)
        self.mode = str(mode)
        self.session_id = uuid.uuid4().hex
        self.fps = max(1, int(fps))
        self.quality = int(quality)
        self.jpeg_provider = jpeg_provider
        self.printer = printer
        self.stats = PublisherStats()
        self._lock = threading.Lock()
        self._meta = {
            "overlays": [],
            "results": [],
            "events": [],
            "metrics": {},
            "serial_data": {},
        }
        self._pending_bgr = None
        self._bgr_sequence = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"manager-publisher-{stream_id}",
        )

    @classmethod
    def from_args(cls, args, mode, jpeg_provider=None, printer=print):
        """从命令行参数创建推流器；未启用发布时返回保持空操作语义的实例。"""
        if not getattr(args, "publish", ""):
            return None
        return cls(
            args.publish,
            args.publish_token,
            args.publish_stream,
            mode,
            fps=args.publish_fps,
            quality=args.publish_quality,
            jpeg_provider=jpeg_provider,
            printer=printer,
        )

    def start(self):
        """启动后台资源或工作线程；重复启动遵循原有幂等规则。"""
        self._thread.start()
        self.printer(
            f"上位机推流已启动: {self.stream_id} "
            f"{self.fps}FPS -> {self.url}"
        )
        return self

    def update(
        self,
        overlays=None,
        metrics=None,
        results=None,
        events=None,
        serial_data=None,
    ):
        """更新随下一帧发送的业务元数据，不立即执行网络I/O。"""
        with self._lock:
            if overlays is not None:
                self._meta["overlays"] = _builtin(overlays)
            if metrics is not None:
                self._meta["metrics"] = _builtin(metrics)
            if results is not None:
                self._meta["results"] = _builtin(results)
            if events is not None:
                self._meta["events"] = _builtin(events)
            if serial_data is not None:
                self._meta["serial_data"] = _builtin(serial_data)

    def submit_bgr(self, frame):
        """把BGR帧提交给发布器；编码与网络发送由后台线程完成。"""
        if frame is None:
            return
        with self._lock:
            self._pending_bgr = frame
            self._bgr_sequence += 1

    def snapshot_stats(self):
        """返回发送帧数、错误数和最近错误等发布统计快照。"""
        with self._lock:
            return PublisherStats(**vars(self.stats))

    def stop(self, timeout=3.0):
        """请求后台工作停止，并保持原有清理与状态通知顺序。"""
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _read_frame(self, last_sequence):
        """读取待发布的最新帧快照，并维持latest-wins语义。"""
        if self.jpeg_provider is not None:
            return self.jpeg_provider(last_sequence)
        with self._lock:
            if self._pending_bgr is None or self._bgr_sequence == last_sequence:
                return None
            frame = self._pending_bgr
            sequence = self._bgr_sequence
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        if not ok:
            raise RuntimeError("OpenCV JPEG encode failed")
        with self._lock:
            self.stats.encoded += 1
        return encoded.tobytes(), sequence, time.monotonic_ns()

    def _metadata(self, sequence, timestamp_ns):
        """构造与既有前端契约一致的帧元数据字典。"""
        with self._lock:
            state = _builtin(self._meta)
        return {
            "version": 1,
            "stream_id": self.stream_id,
            "mode": self.mode,
            "session_id": self.session_id,
            "frame_id": int(sequence),
            "jpeg_sequence": int(sequence),
            "capture_timestamp_ns": int(timestamp_ns),
            "sent_at_ms": int(time.time() * 1000),
            "overlays": state.get("overlays", []),
            "results": state.get("results", []),
            "events": state.get("events", []),
            "metrics": state.get("metrics", {}),
            "serial_data": state.get("serial_data", {}),
        }

    def _connection(self):
        """创建一次推流HTTP连接并发送既定二进制帧协议。"""
        parsed = urlsplit(self.url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, port, timeout=3.0)
        base_path = parsed.path.rstrip("/")
        if base_path.endswith("/api/ingest"):
            path = base_path
        else:
            path = f"{base_path}/api/ingest" if base_path else "/api/ingest"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return connection, path

    def _run(self):
        """执行对象的后台工作循环，直到收到停止请求或不可恢复错误。"""
        interval = 1.0 / self.fps
        next_frame = time.monotonic()
        last_sequence = 0
        connection = None
        path = None
        backoff = 0.5
        last_log = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_frame:
                self._stop.wait(min(next_frame - now, 0.05))
                continue
            next_frame = max(next_frame + interval, now)
            try:
                item = self._read_frame(last_sequence)
                if item is None:
                    continue
                jpeg, sequence, timestamp_ns = item
                if sequence == last_sequence:
                    continue
                meta = self._metadata(sequence, timestamp_ns)
                json_bytes = json.dumps(
                    meta,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                body = struct.pack(">I", len(json_bytes)) + json_bytes + jpeg
                if connection is None:
                    connection, path = self._connection()
                started = time.perf_counter()
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(body)),
                        "X-Publish-Token": self.token,
                    },
                )
                response = connection.getresponse()
                response.read()
                if response.status != 204:
                    raise RuntimeError(
                        f"manager ingest returned HTTP {response.status}"
                    )
                latency_ms = (time.perf_counter() - started) * 1000
                with self._lock:
                    self.stats.sent += 1
                    self.stats.last_latency_ms = latency_ms
                    self.stats.last_error = ""
                last_sequence = sequence
                backoff = 0.5
            except Exception as exc:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self.stats.failures += 1
                    self.stats.last_error = str(exc)
                now = time.monotonic()
                if now - last_log >= 5:
                    self.printer(f"上位机推流暂不可用: {exc}")
                    last_log = now
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 10.0)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
