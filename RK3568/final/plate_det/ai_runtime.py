"""RK车牌检测、识别与HDMI显示的完整运行时。

线程与所有权关系：

- Native采集线程发布NV12 DMA-BUF帧，队列满时采用latest-wins；
- 检测Worker只处理最新任务，完成288×512前处理和Pose26车牌检测；
- 识别Worker持有对应帧引用，提取小ROI后立即释放整帧；
- 显示Worker独立Page Flip，检测和识别变慢不能阻塞60Hz视频；
- 主线程只转发最新结果、更新Overlay、统计时间戳并处理异常恢复。

运行时不生成1080p BGR。队列中的字典若含``frame``就拥有一次引用，覆盖、异常
和退出分支都必须配对释放。

用户仍应通过``python3 fpga_ai.py``启动；本模块是领域实现，不新增CLI。
"""

import sys
import os
import time
import argparse
import threading
import queue as thread_queue
from collections import OrderedDict

from core.camera_fps import configure_camera_fps
from core.display_drm_atomic import AtomicNv12Display
from core.hdmi_io import HDMI_DEVICE, HDMI_WIDTH, HDMI_HEIGHT, HDMI_FPS, HDMI_FOURCC
from core.latency_metrics import (
    DisplayedLatencyTracker,
    DisplayedStageTracker,
    ai_latency_values,
    frame_e2e_source_timestamp_ns,
)
from core.rknn_native import NativeRknnModel
from core.native_capture import NativeMppHdmiReceiver
from core.publisher import (
    FramePublisher,
    add_publish_args,
    plate_overlays,
    validate_publish_args,
)
from plate_det.plate_pipeline import prepare_plate_image, process_prepared_plate
from plate_det.detection_filter import DetectionCandidateFilter, DetectionFilterConfig
from plate_det.result_utils import bbox_iou, is_complete_plate_result
from plate_det.result_stabilizer import PlateTextStabilizer
from plate_det.model_contracts import validate_pose26_model


POSE26_MODEL_PATH = "/home/cat/RK3568/final/rknn_models/plate_ai_pose26_512x288_int8.rknn"
REC_MODEL_PATH = "/home/cat/RK3568/final/rknn_models/plate_ocr_168x48_int8.rknn"


def build_args(argv=None):
    """构造命令行参数解析器，保持历史参数名称、选项和默认值不变。"""
    parser = argparse.ArgumentParser(description="车牌检测识别 (RKNN, 异步并行流水线)")
    parser.add_argument('--skip', type=int, default=0,
                        help="跳帧数（默认0），设为1则每隔1帧做检测，即每2帧检测1次，减轻NPU负载")
    parser.add_argument(
        '--font-path',
        default=None,
        help="C/FreeType标注字体路径，默认自动选择Noto CJK",
    )
    parser.add_argument('--hdmi-device', type=str, default=HDMI_DEVICE,
                        help=f"HDMI 采集设备 (默认: {HDMI_DEVICE})")
    parser.add_argument('--hdmi-width', type=int, default=HDMI_WIDTH,
                        help=f"HDMI 采集宽度 (默认: {HDMI_WIDTH})")
    parser.add_argument('--hdmi-height', type=int, default=HDMI_HEIGHT,
                        help=f"HDMI 采集高度 (默认: {HDMI_HEIGHT})")
    parser.add_argument('--hdmi-fps', type=int, default=HDMI_FPS,
                        help=f"HDMI 采集帧率 (默认: {HDMI_FPS})")
    parser.add_argument('--hdmi-fourcc', type=str, default=HDMI_FOURCC,
                        choices=['MJPG', 'YUYV'],
                        help=f"HDMI 采集格式 (默认: {HDMI_FOURCC})")
    parser.add_argument('--camera-fps-mode', type=str, default='auto-exposure',
                        choices=['stable30', 'auto-exposure', 'off'],
                        help="摄像头帧率策略: auto-exposure=亮度优先允许掉帧(默认,曝光正常不黑屏), stable30=锁定请求帧率优先(暗光会偏黑), off=不修改")
    parser.add_argument('--camera-exposure', type=int, default=None,
                        help="可选手动曝光值；启用后设置 exposure_auto=1 和 exposure_absolute=N")
    parser.add_argument('--camera-gain', type=int, default=None,
                        help="可选固定增益值")
    parser.add_argument('--camera-config-strict', action='store_true',
                        help="摄像头配置失败或实际FPS不匹配时直接退出")
    parser.add_argument('--debug', action='store_true',
                        help="输出中文调试统计：各阶段耗时和帧率")
    parser.add_argument('--draw-mode', type=str, default='fast', choices=['full', 'fast'],
                        help="绘制模式: full=完整标注, fast=低开销标注(默认)")
    parser.add_argument('--result-log-interval', type=int, default=1,
                        help="有识别结果时每 N 帧打印一次日志 (默认: 1)")
    parser.add_argument('--det-sensitivity-mode', type=str, default='balanced',
                        choices=['off', 'balanced', 'high'],
                        help="检测灵敏度: off=旧阈值, balanced=低阈值保守确认(默认), high=更高召回实验")
    parser.add_argument('--det-conf-high', type=float, default=0.30,
                        help="高置信检测阈值，高于该值立即画框 (默认: 0.30)")
    parser.add_argument('--det-conf-low', type=float, default=None,
                        help="低置信检测阈值；不设置时 balanced=0.18, high=0.12, off=高阈值")
    parser.add_argument('--det-low-confirm-frames', type=int, default=2,
                        help="低置信候选连续命中多少帧后画框 (默认: 2)")
    parser.add_argument('--det-max-candidates', type=int, default=4,
                        help="每帧最多处理/绘制的检测候选数 (默认: 4)")
    parser.add_argument('--queue-size', type=int, default=1,
                        help="内部队列大小 (默认: 1，实时模式建议保持 1)")
    parser.add_argument('--async-result-ttl', type=int, default=500,
                        help="识别结果最长显示时间，单位 ms (默认: 500)")
    add_publish_args(parser)
    return parser.parse_args(argv)


DEBUG_INTERVAL = 60


class AsyncCounters:
    def __init__(self):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.lock = threading.Lock()
        self.counts = {
            'received': 0,
            'submitted_detect': 0,
            'dropped_detect': 0,
            'finished_detect': 0,
            'submitted_rec': 0,
            'dropped_rec': 0,
            'finished_rec': 0,
            'rec_real_samples': 0,
            'draw_boxes': 0,
            'recognized_labels': 0,
            'low_conf_candidates': 0,
            'low_conf_confirmed': 0,
            'low_conf_geometry_dropped': 0,
            'low_conf_rec_dropped': 0,
            'dropped_result': 0,
            'composed': 0,
            'displayed': 0,
            'stale_result': 0,
        }
        self.times = {
            'wait_total_ms': 0.0,
            'wait_max_ms': 0.0,
            'decode_total_ms': 0.0,
            'decode_max_ms': 0.0,
            'decode_convert_total_ms': 0.0,
            'decode_convert_max_ms': 0.0,
            'submit_total_ms': 0.0,
            'submit_max_ms': 0.0,
            'detect_total_ms': 0.0,
            'detect_max_ms': 0.0,
            'plate_total_ms': 0.0,
            'plate_max_ms': 0.0,
            'draw_total_ms': 0.0,
            'draw_max_ms': 0.0,
            'display_total_ms': 0.0,
            'display_max_ms': 0.0,
            'loop_total_ms': 0.0,
            'loop_max_ms': 0.0,
            'latency_total_ms': 0.0,
            'latency_max_ms': 0.0,
            'latency_count': 0,
        }

    def inc(self, key, value=1):
        """在线程锁保护下增加指定计数器。"""
        with self.lock:
            self.counts[key] += value

    def add_time(self, total_key, max_key, value):
        """累计指定阶段的耗时和样本数，供周期性能日志计算平均值。"""
        with self.lock:
            self.times[total_key] += value
            self.times[max_key] = max(self.times[max_key], value)

    def add_latency(self, value):
        """记录一次端到端时延样本，保留既有统计窗口。"""
        with self.lock:
            self.times['latency_total_ms'] += value
            self.times['latency_max_ms'] = max(self.times['latency_max_ms'], value)
            self.times['latency_count'] += 1

    def snapshot(self, reset=False):
        """在锁保护下复制当前统计或状态，避免调用方观察到中间状态。"""
        with self.lock:
            counts = self.counts.copy()
            times = self.times.copy()
            if reset:
                for key in self.counts:
                    self.counts[key] = 0
                for key in self.times:
                    self.times[key] = 0.0
        return counts, times


def put_latest(q, item):
    """以latest-wins语义发布任务，并释放被新任务替换的帧引用。"""
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except thread_queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except thread_queue.Empty:
                continue


class NotifyingQueue(thread_queue.Queue):
    """入队时唤醒共享事件的Queue，保留标准Queue全部语义。"""

    def __init__(self, maxsize=0, wake_event=None):
        super().__init__(maxsize=maxsize)
        self.wake_event = wake_event or threading.Event()

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self.wake_event.set()


def release_native_frame(value):
    """释放任务持有的Native帧引用；空值和已释放对象可安全忽略。"""
    frame = value.get('frame') if isinstance(value, dict) else value
    release = getattr(frame, 'release', None)
    if callable(release):
        release()


def put_latest_owned(q, item):
    """以latest-wins方式发布任务；丢弃旧任务时同步释放其Native帧所有权。"""
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except thread_queue.Full:
            try:
                stale = q.get_nowait()
                release_native_frame(stale)
                dropped += 1
            except thread_queue.Empty:
                continue


def drain_latest(q):
    """清空队列中的待处理项目，并按所有权约定释放其中的帧。"""
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except thread_queue.Empty:
            return latest


def build_detection_filter_config(args):
    """根据CLI参数构造低置信度候选过滤配置。"""
    return DetectionFilterConfig(
        mode=args.det_sensitivity_mode,
        conf_low=args.det_conf_low,
        conf_high=args.det_conf_high,
        low_confirm_frames=args.det_low_confirm_frames,
        max_candidates=args.det_max_candidates,
    )


def detection_to_draw_result(det_result):
    """执行“detection to draw result”流程，保持原有输入输出和副作用不变。"""
    landmarks = det_result.get('landmarks', [])
    if hasattr(landmarks, 'tolist'):
        landmarks = landmarks.tolist()
    return {
        'bbox': det_result['bbox'],
        'landmarks': landmarks,
        'detect_conf': float(det_result.get('detect_conf', det_result.get('conf', 0.0))),
        'plate_type': det_result.get('plate_type', det_result.get('class', 0)),
        'det_conf_level': det_result.get('det_conf_level', 'high'),
    }


def _recognition_with_current_detection(recognition, draw_candidate):
    """把单次OCR字段合入当前帧检测几何。"""
    result = dict(recognition)
    result['bbox'] = draw_candidate['bbox']
    result['landmarks'] = draw_candidate.get('landmarks', [])
    result['detect_conf'] = draw_candidate.get('detect_conf', result.get('detect_conf', 0.0))
    result['plate_type'] = draw_candidate.get('plate_type', result.get('plate_type', 0))
    return result


def merge_draw_results(det_candidates, recognized_plates, min_iou=0.5):
    """合并检测与识别结果，AI模式继续使用原有bbox矩形框。"""
    draw_candidates = [detection_to_draw_result(item) for item in (det_candidates or [])]
    recognized_results = [
        dict(item) for item in (recognized_plates or [])
        if is_complete_plate_result(item)
    ]
    merged = []
    used_detection_indexes = set()

    for recognition in recognized_results:
        best_index = None
        best_iou = 0.0
        for idx, candidate in enumerate(draw_candidates):
            iou = bbox_iou(recognition['bbox'], candidate['bbox'])
            if iou > best_iou:
                best_index = idx
                best_iou = iou
        if best_index is not None and best_iou >= min_iou:
            recognition = _recognition_with_current_detection(
                recognition, draw_candidates[best_index]
            )
            used_detection_indexes.add(best_index)
        merged.append(recognition)

    for idx, candidate in enumerate(draw_candidates):
        if idx not in used_detection_indexes:
            merged.append(candidate)
    return merged


def wait_for_workers(status_q, expected, stop_event):
    """执行“wait for workers”流程，保持原有输入输出和副作用不变。"""
    ready = 0
    deadline = time.monotonic() + 60.0
    while ready < expected:
        if time.monotonic() > deadline:
            stop_event.set()
            raise RuntimeError("异步 worker 初始化超时")
        try:
            status = status_q.get(timeout=0.1)
        except thread_queue.Empty:
            continue
        kind, worker_name, message = status
        if kind == 'ready':
            ready += 1
            print(f"{worker_name} 已就绪: {message}")
        elif kind == 'error':
            stop_event.set()
            raise RuntimeError(f"{worker_name} 初始化失败: {message}")


def drain_worker_status(status_q, stop_event, error_phase='运行'):
    """执行“drain worker status”流程，保持原有输入输出和副作用不变。"""
    while True:
        try:
            status = status_q.get_nowait()
        except thread_queue.Empty:
            return
        kind, worker_name, message = status
        if kind == 'error':
            stop_event.set()
            raise RuntimeError(f"{worker_name} {error_phase}失败: {message}")
        if kind == 'recovered':
            print(f"{worker_name} 已重建RKNN上下文: {message}", flush=True)


class _NpuRoleLock:
    """把检测或识别角色绑定到统一NPU调度器，兼容普通Lock接口。"""

    def __init__(self, arbiter, role):
        """保存调度器和固定角色；每个实例只供对应Worker使用。"""
        self._arbiter = arbiter
        self._role = role

    def acquire(self):
        """等待调度器授予本角色一次NPU执行权。"""
        self._arbiter.acquire(self._role)
        return True

    def release(self):
        """归还本角色持有的NPU执行权并唤醒下一角色。"""
        self._arbiter.release(self._role)

    def reserve_next(self, role):
        """预约下一次NPU执行权，防止目标Worker尚未唤醒时被抢占。"""
        self._arbiter.reserve_next(role)


class NpuExecutionArbiter:
    """在单核NPU上交替调度检测和识别，避免两个Worker无序抢锁。

    只有一方等待时允许其连续运行；双方同时等待时严格在检测和识别之间
    交替。这样每次检测产生的识别任务都有机会在下一次检测前运行，同时
    检测队列仍保持latest-wins，不会累积过期帧。
    """

    _ROLES = ("detect", "recognition")

    def __init__(self):
        """初始化条件变量、等待计数和最近完成角色。"""
        self._condition = threading.Condition()
        self._active_role = None
        self._last_completed_role = None
        self._reserved_role = None
        self._waiting = {role: 0 for role in self._ROLES}

    def role_lock(self, role):
        """返回绑定角色且兼容acquire/release接口的轻量锁。"""
        if role not in self._waiting:
            raise ValueError(f"unknown NPU role: {role}")
        return _NpuRoleLock(self, role)

    def _can_run(self, role):
        """判断当前角色是否应获得下一次NPU执行权。"""
        if self._active_role is not None:
            return False
        if self._reserved_role is not None:
            return role == self._reserved_role
        other_role = (
            "recognition" if role == "detect" else "detect"
        )
        if self._waiting[other_role] == 0:
            return True
        if self._last_completed_role is None:
            return role == "detect"
        return role != self._last_completed_role

    def acquire(self, role):
        """登记等待并按交替规则取得NPU执行权。"""
        if role not in self._waiting:
            raise ValueError(f"unknown NPU role: {role}")
        with self._condition:
            self._waiting[role] += 1
            try:
                self._condition.wait_for(lambda: self._can_run(role))
                self._active_role = role
                if self._reserved_role == role:
                    self._reserved_role = None
            finally:
                self._waiting[role] -= 1

    def reserve_next(self, role):
        """强制下一次NPU执行权属于指定角色，直到该角色实际取得执行权。"""
        if role not in self._waiting:
            raise ValueError(f"unknown NPU role: {role}")
        with self._condition:
            self._reserved_role = role
            self._condition.notify_all()

    def cancel_reservation(self):
        """停止流水线时取消未消费的预约，避免Worker退出阶段互相等待。"""
        with self._condition:
            self._reserved_role = None
            self._condition.notify_all()

    def release(self, role):
        """完成当前角色并通知所有等待Worker重新竞争调度条件。"""
        with self._condition:
            if self._active_role != role:
                raise RuntimeError(
                    f"NPU role {role} released while "
                    f"{self._active_role!r} is active"
                )
            self._active_role = None
            self._last_completed_role = role
            self._condition.notify_all()


def run_with_rknn_recovery(
    model, model_path, worker_name, operation, status_q,
    execution_lock=None, execution_timing=None, model_loader=None,
):
    """执行一次NPU操作；首次失败时重建RKNN上下文并只重试一次。"""
    if execution_lock is not None:
        wait_started_ns = time.monotonic_ns()
        execution_lock.acquire()
        if execution_timing is not None:
            execution_timing["npu_wait_ms"] = max(
                0.0,
                (time.monotonic_ns() - wait_started_ns) / 1_000_000.0,
            )
    try:
        try:
            return operation(model), model
        except Exception as first_error:
            model.release()
            loader = load_rknn_model if model_loader is None else model_loader
            replacement = loader(model_path, worker_name)
            try:
                result = operation(replacement)
            except Exception as retry_error:
                replacement.release()
                raise RuntimeError(
                    f"RKNN重建后重试仍失败: {retry_error}; "
                    f"首次错误: {first_error}"
                ) from retry_error
            status_q.put(('recovered', worker_name, str(first_error)))
            return result, replacement
    finally:
        if execution_lock is not None:
            execution_lock.release()


class Pose26DoubleBufferPreprocessor:
    """用两个Native输入槽在后台执行latest-wins RGA前处理。"""

    def __init__(self, model, source_q, pipeline_stop_event):
        self.model = model
        self.source_q = source_q
        self.pipeline_stop_event = pipeline_stop_event
        self.stop_event = threading.Event()
        self.prepared_q = thread_queue.Queue(maxsize=1)
        self.free_slots = thread_queue.Queue(maxsize=2)
        self.free_slots.put(0)
        self.free_slots.put(1)
        self.errors = thread_queue.Queue(maxsize=1)
        self._permission = threading.Event()
        self._permission.set()
        self._deadline_lock = threading.Lock()
        self._prepare_not_before_ns = 0
        self._generation = 0
        self.thread = threading.Thread(
            target=self._run,
            name="pose26-rga-preprocessor",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _stopping(self):
        return self.stop_event.is_set() or self.pipeline_stop_event.is_set()

    def allow_next(self, delay_ms):
        """在当前NPU预计结束前一个RGA窗口放行下一帧预处理。"""
        deadline_ns = time.monotonic_ns() + int(
            max(0.0, float(delay_ms)) * 1_000_000.0
        )
        with self._deadline_lock:
            self._prepare_not_before_ns = deadline_ns
        self._permission.set()

    def discard_pending_and_allow(self, delay_ms):
        """废弃已预约的旧输入，并在新时刻预取latest-wins帧。"""
        deadline_ns = time.monotonic_ns() + int(
            max(0.0, float(delay_ms)) * 1_000_000.0
        )
        with self._deadline_lock:
            self._generation += 1
            self._prepare_not_before_ns = deadline_ns
            generation = self._generation
            self._permission.set()

        discarded = 0
        while True:
            try:
                job = self.prepared_q.get_nowait()
            except thread_queue.Empty:
                break
            if job is None:
                self.prepared_q.put_nowait(None)
                break
            release_native_frame(job["item"])
            self.free_slots.put(int(job["slot"]))
            discarded += 1
        return generation, discarded

    def _wait_until_allowed(self):
        while not self._stopping():
            if not self._permission.wait(timeout=0.05):
                continue
            with self._deadline_lock:
                deadline_ns = self._prepare_not_before_ns
                generation = self._generation
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return generation
            self.stop_event.wait(min(0.05, remaining_ns / 1_000_000_000.0))
        return None

    def _acquire_slot(self):
        while not self._stopping():
            try:
                return self.free_slots.get(timeout=0.05)
            except thread_queue.Empty:
                continue
        return None

    def _publish(self, prepared_job):
        while not self._stopping():
            try:
                self.prepared_q.put(prepared_job, timeout=0.05)
                return True
            except thread_queue.Full:
                continue
        return False

    def _run(self):
        active_item = None
        active_slot = None
        try:
            while not self._stopping():
                generation = self._wait_until_allowed()
                if generation is None:
                    return
                try:
                    item = self.source_q.get(timeout=0.1)
                except thread_queue.Empty:
                    continue
                if item is None:
                    self._publish(None)
                    return
                active_item = item
                with self._deadline_lock:
                    if generation != self._generation:
                        release_native_frame(active_item)
                        active_item = None
                        continue
                active_slot = self._acquire_slot()
                if active_slot is None:
                    return
                with self._deadline_lock:
                    if generation != self._generation:
                        release_native_frame(active_item)
                        active_item = None
                        self.free_slots.put(active_slot)
                        active_slot = None
                        continue
                prepare_started_ns = time.monotonic_ns()
                prepared = self.model.prepare_pose26_nv12(
                    item["frame"], active_slot
                )
                prepare_finished_ns = time.monotonic_ns()
                job = {
                    "item": item,
                    "slot": active_slot,
                    "prepared": prepared,
                    "prepare_started_ns": prepare_started_ns,
                    "prepare_finished_ns": prepare_finished_ns,
                    "generation": generation,
                }
                active_item = None
                active_slot = None
                with self._deadline_lock:
                    if generation == self._generation:
                        self._permission.clear()
                if not self._publish(job):
                    release_native_frame(item)
                    self.free_slots.put(job["slot"])
                    return
        except Exception as exc:
            if active_item is not None:
                release_native_frame(active_item)
            if active_slot is not None:
                self.free_slots.put(active_slot)
            try:
                self.errors.put_nowait(exc)
            except thread_queue.Full:
                pass

    def get(self, timeout=0.1):
        deadline = time.monotonic() + timeout
        while True:
            try:
                error = self.errors.get_nowait()
            except thread_queue.Empty:
                error = None
            if error is not None:
                raise error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise thread_queue.Empty
            try:
                job = self.prepared_q.get(timeout=remaining)
            except thread_queue.Empty:
                try:
                    error = self.errors.get_nowait()
                except thread_queue.Empty:
                    raise
                raise error
            if job is None:
                return None
            with self._deadline_lock:
                current_generation = self._generation
            if int(job.get("generation", 0)) == current_generation:
                return job
            release_native_frame(job["item"])
            self.free_slots.put(int(job["slot"]))

    def release_slot(self, job):
        self.free_slots.put(int(job["slot"]))

    def close(self):
        self.stop_event.set()
        self._permission.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("Pose26 RGA preprocessor did not stop")
        while True:
            try:
                job = self.prepared_q.get_nowait()
            except thread_queue.Empty:
                break
            if job is not None:
                release_native_frame(job["item"])
                self.free_slots.put(int(job["slot"]))


def pose26_preload_delay_ms(last_timing):
    """根据上一轮耗时让RGA在NPU结束后约1ms完成，优先保持结果新鲜。"""
    timing = last_timing or {}
    npu_ms = float(timing.get("npu_ms", 18.8))
    preprocess_ms = float(timing.get("preprocess_ms", 4.1))
    return max(0.0, min(25.0, npu_ms - preprocess_ms + 1.0))


def pose26_recognition_preload_delay_ms(candidate_count, last_timing):
    """按候选数估算OCR占用，让新检测输入在识别结束附近就绪。"""
    timing = last_timing or {}
    preprocess_ms = float(timing.get("preprocess_ms", 4.1))
    recognition_ms = max(1, int(candidate_count)) * 6.5
    return max(0.0, min(25.0, recognition_ms - preprocess_ms + 1.0))


def detect_worker(worker_id, args, detect_q, rec_q, draw_q, status_q,
                  stop_event, counters, npu_lock):
    """消费最新帧执行检测，把候选结果发布给识别与显示链路。"""
    worker_name = f"检测worker-{worker_id}"
    rknn = None
    preprocessor = None
    active_item = None
    detector_filter = DetectionCandidateFilter(build_detection_filter_config(args))
    try:
        rknn = load_pose26_model(POSE26_MODEL_PATH, worker_name)
        print(
            f"{worker_name}模型: {POSE26_MODEL_PATH} "
            f"{rknn.tensor_summary()}"
        )
        status_q.put(('ready', worker_name, POSE26_MODEL_PATH))
        preprocessor = Pose26DoubleBufferPreprocessor(
            rknn, detect_q, stop_event
        )
        preprocessor.start()

        while not stop_event.is_set():
            try:
                prepared_job = preprocessor.get(timeout=0.1)
            except thread_queue.Empty:
                continue
            if prepared_job is None:
                break
            item = prepared_job["item"]
            active_item = item

            detect_started_ns = int(prepared_job["prepare_started_ns"])
            detect_wait_started_ns = int(
                prepared_job["prepare_finished_ns"]
            )
            npu_lock.acquire()
            detect_npu_wait_ms = max(
                0.0,
                (
                    time.monotonic_ns() - detect_wait_started_ns
                ) / 1_000_000.0,
            )
            try:
                slot_owner = preprocessor
                slot_released = False
                slot_owner.allow_next(
                    pose26_preload_delay_ms(
                        getattr(rknn, "last_detection_timing", None)
                    )
                )
                try:
                    try:
                        det_results = rknn.detect_pose26_prepared(
                            prepared_job["prepared"],
                            args.det_conf_low,
                            max_results=32,
                        )
                    except Exception as first_error:
                        slot_owner.release_slot(prepared_job)
                        slot_released = True
                        slot_owner.close()
                        rknn.release()
                        replacement = load_pose26_model(
                            POSE26_MODEL_PATH, worker_name
                        )
                        try:
                            retry_prepared = replacement.prepare_pose26_nv12(
                                item["frame"], 0
                            )
                            det_results = replacement.detect_pose26_prepared(
                                retry_prepared,
                                args.det_conf_low,
                                max_results=32,
                            )
                        except Exception as retry_error:
                            replacement.release()
                            rknn = None
                            preprocessor = None
                            raise RuntimeError(
                                f"RKNN重建后重试仍失败: {retry_error}; "
                                f"首次错误: {first_error}"
                            ) from retry_error
                        rknn = replacement
                        preprocessor = Pose26DoubleBufferPreprocessor(
                            rknn, detect_q, stop_event
                        )
                        preprocessor.start()
                        status_q.put(
                            ('recovered', worker_name, str(first_error))
                        )
                finally:
                    if not slot_released:
                        slot_owner.release_slot(prepared_job)
                filter_result = detector_filter.filter(det_results)
                native_detection_timing = dict(
                    getattr(rknn, 'last_detection_timing', None) or {}
                )
                if (
                    filter_result.recognition_candidates
                    and hasattr(npu_lock, "reserve_next")
                ):
                    # 在释放检测执行权前预约识别，彻底消除下一帧检测
                    # 抢在识别Worker唤醒之前再次取得NPU的竞态。
                    npu_lock.reserve_next("recognition")
                    preprocessor.discard_pending_and_allow(
                        pose26_recognition_preload_delay_ms(
                            len(filter_result.recognition_candidates),
                            native_detection_timing,
                        )
                    )
            finally:
                npu_lock.release()
            detect_finished_ns = time.monotonic_ns()
            detect_ms = (
                detect_finished_ns - detect_started_ns
            ) / 1_000_000.0
            detect_compute_ms = max(
                0.0, detect_ms - detect_npu_wait_ms
            )
            source_timestamp_ns = int(
                item.get(
                    'source_timestamp_ns',
                    float(item['timestamp']) * 1_000_000_000.0,
                )
            )
            detection_latency_ms = max(
                0.0,
                (detect_finished_ns - source_timestamp_ns) / 1_000_000.0,
            )
            e2e_source_timestamp_ns = int(
                item.get("e2e_source_timestamp_ns", source_timestamp_ns)
            )
            submitted_ns = int(
                item.get('submitted_ns', detect_started_ns)
            )
            decoded_timestamp_ns = int(
                item.get('decoded_timestamp_ns', source_timestamp_ns)
            )
            decode_latency_ms = max(
                0.0,
                (decoded_timestamp_ns - source_timestamp_ns)
                / 1_000_000.0,
            )
            processing_decode_ms = max(
                0.0,
                (decoded_timestamp_ns - e2e_source_timestamp_ns)
                / 1_000_000.0,
            )
            detect_queue_ms = max(
                0.0,
                (detect_started_ns - submitted_ns) / 1_000_000.0,
            )
            counters.inc('finished_detect')
            counters.add_time('detect_total_ms', 'detect_max_ms', detect_ms)

            counters.inc('draw_boxes', len(filter_result.draw_candidates))
            counters.inc('low_conf_candidates', filter_result.stats.low_conf_candidates)
            counters.inc('low_conf_confirmed', filter_result.stats.low_conf_confirmed)
            counters.inc('low_conf_geometry_dropped', filter_result.stats.geometry_dropped)

            draw_item = {
                'frame_id': item['frame_id'],
                'timestamp': item['timestamp'],
                'det_results_for_draw': filter_result.draw_candidates,
                'detect_ms': detect_ms,
                'detect_npu_wait_ms': detect_npu_wait_ms,
                'detect_compute_ms': detect_compute_ms,
                'native_detection_timing': native_detection_timing,
                'source_timestamp_ns': source_timestamp_ns,
                'e2e_source_timestamp_ns': e2e_source_timestamp_ns,
                'detect_started_ns': detect_started_ns,
                'detect_finished_ns': detect_finished_ns,
                'detection_latency_ms': detection_latency_ms,
                'decoded_timestamp_ns': decoded_timestamp_ns,
                'submitted_ns': submitted_ns,
                'decode_latency_ms': decode_latency_ms,
                'processing_decode_ms': processing_decode_ms,
                'detect_queue_ms': detect_queue_ms,
                'finished_at': detect_finished_ns / 1_000_000_000.0,
            }
            dropped = put_latest(draw_q, draw_item)
            if dropped:
                counters.inc('dropped_result', dropped)

            if not filter_result.recognition_candidates:
                release_native_frame(item)
                active_item = None
                continue

            rec_item = {
                'frame_id': item['frame_id'],
                'timestamp': item['timestamp'],
                'frame': item['frame'],
                'det_results': filter_result.recognition_candidates,
                'det_results_for_draw': filter_result.draw_candidates,
                'detect_ms': detect_ms,
                'detect_npu_wait_ms': detect_npu_wait_ms,
                'detect_compute_ms': detect_compute_ms,
                'native_detection_timing': native_detection_timing,
                'source_timestamp_ns': source_timestamp_ns,
                'e2e_source_timestamp_ns': e2e_source_timestamp_ns,
                'detect_started_ns': detect_started_ns,
                'detect_finished_ns': detect_finished_ns,
                'detection_latency_ms': detection_latency_ms,
                'decoded_timestamp_ns': decoded_timestamp_ns,
                'submitted_ns': submitted_ns,
                'decode_latency_ms': decode_latency_ms,
                'processing_decode_ms': processing_decode_ms,
                'detect_queue_ms': detect_queue_ms,
            }
            dropped = put_latest_owned(rec_q, rec_item)
            active_item = None
            counters.inc('submitted_rec')
            if dropped:
                counters.inc('dropped_rec', dropped)
    except Exception as e:
        if active_item is not None:
            release_native_frame(active_item)
        status_q.put(('error', worker_name, str(e)))
    finally:
        if preprocessor is not None:
            preprocessor.close()
        if rknn is not None:
            rknn.release()


def rec_worker(worker_id, args, rec_q, result_q, status_q, stop_event,
               counters, npu_lock):
    """消费识别任务并立即采用单次OCR结果；完成后释放帧。"""
    worker_name = f"识别worker-{worker_id}"
    rknn = None
    active_item = None
    try:
        rknn = load_rknn_model(REC_MODEL_PATH, worker_name)
        status_q.put(('ready', worker_name, REC_MODEL_PATH))

        while not stop_event.is_set():
            try:
                item = rec_q.get(timeout=0.1)
            except thread_queue.Empty:
                continue
            if item is None:
                break
            active_item = item

            plate_results = []
            recognition_npu_ms = 0.0
            recognition_started_ns = time.monotonic_ns()
            frame = item.get('frame')
            prepared_plates = []
            try:
                for det_result in item['det_results']:
                    prepared_plates.append(
                        (det_result, prepare_plate_image(frame, det_result))
                    )
            finally:
                if frame is not None:
                    frame.release()
                item['frame'] = None

            recognition_wait_started_ns = time.monotonic_ns()
            npu_lock.acquire()
            recognition_npu_wait_ms = max(
                0.0,
                (
                    time.monotonic_ns() - recognition_wait_started_ns
                ) / 1_000_000.0,
            )
            try:
                # 一次识别任务持有一个完整NPU时间片；多车牌在同一时间片内
                # 串行完成，检测只能在整批识别结束后处理最新帧。
                for det_result, prepared_plate in prepared_plates:
                    plate_result, rknn = run_with_rknn_recovery(
                        rknn,
                        REC_MODEL_PATH,
                        worker_name,
                        lambda active_model, candidate=det_result:
                            process_prepared_plate(
                                prepared_plate, candidate, active_model
                            ),
                        status_q,
                    )
                    ocr_timing = getattr(rknn, 'last_ocr_timing', None) or {}
                    recognition_npu_ms += max(
                        0.0, float(ocr_timing.get('npu_ms', 0.0) or 0.0)
                    )
                    counters.inc('rec_real_samples')
                    if _drop_low_conf_recognition_if_needed(
                        det_result, plate_result, counters
                    ):
                        continue
                    if is_complete_plate_result(plate_result):
                        plate_results.append(plate_result)
            finally:
                npu_lock.release()
            recognition_finished_ns = time.monotonic_ns()
            plate_ms = (
                recognition_finished_ns - recognition_started_ns
            ) / 1_000_000.0
            plate_compute_ms = max(
                0.0, plate_ms - recognition_npu_wait_ms
            )
            source_timestamp_ns = int(item['source_timestamp_ns'])
            detect_finished_ns = int(item['detect_finished_ns'])
            recognition_latency_ms = max(
                0.0,
                (recognition_finished_ns - detect_finished_ns)
                / 1_000_000.0,
            )
            recognition_queue_ms = max(
                0.0,
                (recognition_started_ns - detect_finished_ns)
                / 1_000_000.0,
            )
            counters.inc('finished_rec')
            counters.inc('recognized_labels', len(plate_results))
            counters.add_time('plate_total_ms', 'plate_max_ms', plate_ms)

            result = {
                'kind': 'rec',
                'frame_id': item['frame_id'],
                'timestamp': item['timestamp'],
                'plate_results': plate_results,
                'det_results_for_draw': item.get(
                    'det_results_for_draw', item['det_results']
                ),
                'detect_ms': item['detect_ms'],
                'detect_npu_wait_ms': item.get(
                    'detect_npu_wait_ms', 0.0
                ),
                'detect_compute_ms': item.get(
                    'detect_compute_ms', item['detect_ms']
                ),
                'native_detection_timing': item.get(
                    'native_detection_timing', {}
                ),
                'plate_ms': plate_ms,
                'recognition_npu_ms': recognition_npu_ms,
                'recognition_npu_wait_ms': recognition_npu_wait_ms,
                'plate_compute_ms': plate_compute_ms,
                'recognition_candidate_count': len(prepared_plates),
                'source_timestamp_ns': source_timestamp_ns,
                'e2e_source_timestamp_ns': int(
                    item.get('e2e_source_timestamp_ns', source_timestamp_ns)
                ),
                'detect_finished_ns': detect_finished_ns,
                'recognition_started_ns': recognition_started_ns,
                'recognition_finished_ns': recognition_finished_ns,
                'detection_latency_ms': item['detection_latency_ms'],
                'recognition_latency_ms': recognition_latency_ms,
                'decoded_timestamp_ns': item['decoded_timestamp_ns'],
                'decode_latency_ms': item['decode_latency_ms'],
                'processing_decode_ms': item.get(
                    'processing_decode_ms', item['decode_latency_ms']
                ),
                'detect_queue_ms': item['detect_queue_ms'],
                'recognition_queue_ms': recognition_queue_ms,
                'ready_latency_ms': max(
                    0.0,
                    (recognition_finished_ns - source_timestamp_ns)
                    / 1_000_000.0,
                ),
                'finished_at': recognition_finished_ns / 1_000_000_000.0,
            }
            dropped = put_latest(result_q, result)
            if dropped:
                counters.inc('dropped_result', dropped)
            release_native_frame(item)
            active_item = None
    except Exception as e:
        if active_item is not None:
            release_native_frame(active_item)
        status_q.put(('error', worker_name, str(e)))
    finally:
        if rknn is not None:
            rknn.release()


def print_debug_stats(frame_no, interval_count, elapsed, counters,
                       recv_process=None, state=None, display=None):
    """按既有日志格式输出debug stats信息。"""
    counts, times = counters.snapshot(reset=True)
    display_fps = counts['displayed'] / max(elapsed, 1e-6)
    detect_avg = times['detect_total_ms'] / max(counts['finished_detect'], 1)
    plate_avg = times['plate_total_ms'] / max(counts['finished_rec'], 1)
    draw_avg = times['draw_total_ms'] / max(counts['composed'], 1)

    # mpp: 从采集器 native 累计 stats 取区间增量, 按区间解码帧数平均。
    mpp_avg = 0.0
    if recv_process is not None:
        snap = recv_process.get_latest_stats_snapshot()
        if snap is not None:
            prev = state.get('prev_native_stats') if state is not None else None
            if prev is not None:
                d_dec = max(int(snap.decoded) - int(prev.decoded), 1)
                mpp_avg = (snap.mpp_total_ms - prev.mpp_total_ms) / d_dec
            if state is not None:
                state['prev_native_stats'] = snap

    metric_result = None
    native_stats = None
    if display is not None:
        native_stats = display.get_stats()
    if state is not None:
        stage_tracker = state.get('displayed_stage_tracker')
        if stage_tracker is not None and native_stats is not None:
            metric_result = stage_tracker.resolve(
                native_stats,
                state.get('displayed_stage_ttl_ms', 0),
            )
        if metric_result is None:
            metric_result = (
                state.get('latest_recognition_result')
                or state.get('latest_detection_result')
            )
    decode_ms = mpp_avg
    detect_queue_ms = 0.0
    detect_npu_wait_ms = 0.0
    rec_queue_ms = 0.0
    rec_npu_wait_ms = 0.0
    rec_candidate_count = 0
    result_poll_ms = 0.0
    native_detection_timing = {}
    if metric_result is not None:
        decode_ms = float(
            metric_result.get(
                'processing_decode_ms',
                metric_result.get('decode_latency_ms', decode_ms),
            ) or 0.0
        )
        detect_queue_ms = float(
            metric_result.get('detect_queue_ms', 0.0) or 0.0
        )
        detect_npu_wait_ms = float(
            metric_result.get('detect_npu_wait_ms', 0.0) or 0.0
        )
        detect_avg = float(
            metric_result.get(
                'detect_compute_ms',
                metric_result.get('detect_ms', detect_avg),
            ) or 0.0
        )
        rec_queue_ms = float(
            metric_result.get('recognition_queue_ms', 0.0) or 0.0
        )
        rec_npu_wait_ms = float(
            metric_result.get('recognition_npu_wait_ms', 0.0) or 0.0
        )
        plate_avg = float(
            metric_result.get(
                'plate_compute_ms',
                metric_result.get('plate_ms', 0.0),
            ) or 0.0
        )
        rec_candidate_count = int(
            metric_result.get('recognition_candidate_count', 0) or 0
        )
        result_poll_ms = float(
            metric_result.get('result_poll_ms', 0.0) or 0.0
        )
        native_detection_timing = (
            metric_result.get('native_detection_timing', {}) or {}
        )
    overlay_ms = draw_avg
    display_ms = 0.0
    end_to_end_ms = float(
        (metric_result or {}).get('algorithm_e2e_ms', 0.0) or 0.0
    )
    if native_stats is not None:
        overlay_ms = (
            native_stats.overlay_total_ms
            / max(native_stats.overlay_updates, 1)
        )
        display_ms = native_stats.last_overlay_display_ms
    print(
        f"[调试 {frame_no}] 显示:{display_fps:.1f}FPS | "
        f"DQBUF→解码:{decode_ms:.1f}ms | "
        f"检测排队:{detect_queue_ms:.1f}ms | "
        f"检测NPU等待:{detect_npu_wait_ms:.1f}ms | "
        f"检测计算:{detect_avg:.1f}ms | "
        f"检测RGA:{float(native_detection_timing.get('preprocess_ms', 0.0)):.1f}ms | "
        f"检测NPU:{float(native_detection_timing.get('npu_ms', 0.0)):.1f}ms | "
        f"检测输出:{float(native_detection_timing.get('output_ms', 0.0)):.1f}ms | "
        f"检测解码:{float(native_detection_timing.get('decode_ms', 0.0)):.1f}ms | "
        f"检测NMS:{float(native_detection_timing.get('nms_ms', 0.0)):.1f}ms | "
        f"识别排队:{rec_queue_ms:.1f}ms | "
        f"识别NPU等待:{rec_npu_wait_ms:.1f}ms | "
        f"识别计算:{plate_avg:.1f}ms | "
        f"识别候选:{rec_candidate_count} | "
        f"结果调度:{result_poll_ms:.1f}ms | "
        f"Overlay:{overlay_ms:.2f}ms | "
        f"就绪→显示:{display_ms:.1f}ms | "
        f"端到端:{end_to_end_ms:.1f}ms | "
        f"DRM错误:{int(native_stats.commit_errors) if native_stats else 0}"
    )


def load_rknn_model(model_path, name):
    """AI模式固定使用板端Native C RKNN运行时。"""
    return NativeRknnModel(model_path)


def load_pose26_model(model_path, name):
    """加载并严格校验AI模式唯一允许的Pose26模型。"""
    model = NativeRknnModel(model_path)
    try:
        validate_pose26_model(model)
    except Exception:
        model.release()
        raise
    return model


def _is_low_conf_detection(det_result):
    """判断候选框置信度是否低于直接采用阈值。"""
    return det_result.get('det_conf_level') == 'low'


def _passes_low_conf_recognition_filter(plate_result):
    """实现模块内部的“passes low conf recognition filter”辅助逻辑，不作为新的外部接口。"""
    if not is_complete_plate_result(plate_result):
        return False
    if float(plate_result.get('rec_conf', 0.0)) < 0.45:
        return False
    if float(plate_result.get('color_conf', 0.0)) < 0.45:
        return False
    return len(str(plate_result.get('plate_no', ''))) >= 5


def _drop_low_conf_recognition_if_needed(det_result, plate_result, counters):
    """实现模块内部的“drop low conf recognition if needed”辅助逻辑，不作为新的外部接口。"""
    if not _is_low_conf_detection(det_result):
        return False
    if _passes_low_conf_recognition_filter(plate_result):
        return False
    if counters is not None:
        counters.inc('low_conf_rec_dropped')
    return True


def start_async_runtime(args, wake_event=None):
    # RK3568 单核 NPU: 推理串行, 多 worker 只增排队开销不增吞吐(实测),
    # 故检测/识别各固定 1 个 worker。
    """启动async runtime流程并返回其运行句柄。"""
    stop_event = threading.Event()
    counters = AsyncCounters()
    work_event = wake_event or threading.Event()
    status_q = NotifyingQueue(wake_event=work_event)
    npu_arbiter = NpuExecutionArbiter()
    detect_npu_lock = npu_arbiter.role_lock("detect")
    recognition_npu_lock = npu_arbiter.role_lock("recognition")
    detect_q = thread_queue.Queue(maxsize=args.queue_size)
    rec_q = thread_queue.Queue(maxsize=args.queue_size)
    draw_q = NotifyingQueue(
        maxsize=args.queue_size, wake_event=work_event
    )
    result_q = NotifyingQueue(
        maxsize=args.queue_size, wake_event=work_event
    )
    runtime = {
        'stop_event': stop_event,
        'counters': counters,
        'status_q': status_q,
        'detect_q': detect_q,
        'rec_q': rec_q,
        'draw_q': draw_q,
        'result_q': result_q,
        'threads': [],
        'det_workers': 1,
        'npu_lock': npu_arbiter,
        'wake_event': work_event,
    }
    threads = runtime['threads']

    try:
        t = threading.Thread(
            target=detect_worker,
            name="detect-worker-1",
            args=(
                1, args, detect_q, rec_q, draw_q, status_q, stop_event,
                counters, detect_npu_lock,
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=rec_worker,
            name="rec-worker-1",
            args=(
                1, args, rec_q, result_q, status_q, stop_event, counters,
                recognition_npu_lock,
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

        wait_for_workers(status_q, 2, stop_event)
        return runtime
    except BaseException:
        try:
            stop_async_runtime(runtime)
        except BaseException:
            pass
        raise


def stop_async_runtime(runtime):
    """停止async runtime流程并完成既定资源回收。"""
    runtime['stop_event'].set()
    npu_arbiter = runtime.get('npu_lock')
    if hasattr(npu_arbiter, 'cancel_reservation'):
        npu_arbiter.cancel_reservation()
    alive_threads = []
    for t in runtime['threads']:
        t.join(timeout=2.0)
        if t.is_alive():
            alive_threads.append(t.name)
    if alive_threads:
        raise RuntimeError(
            "异步 worker 未能在超时内退出: " + ", ".join(alive_threads)
        )


    for queue_name in ('detect_q', 'rec_q'):
        owned_queue = runtime[queue_name]
        while True:
            try:
                release_native_frame(owned_queue.get_nowait())
            except thread_queue.Empty:
                break


def build_async_state():
    """创建异步检测、识别和显示线程共享的队列、事件与统计状态。"""
    return {
        'frame_count': 0,
        'latest_detection_result': None,
        'latest_recognition_result': None,
        'profile_count': 0,
        'profile_last_t': time.perf_counter(),
        'frame_processed': False,
        'last_result_str': '',
        'last_result_log_frame': None,
        'prev_native_stats': None,
    }


def start_receiver(args):
    """启动receiver流程并返回其运行句柄。"""
    frame_queue = NotifyingQueue(maxsize=1)

    device_deadline = time.monotonic() + 5.0
    while (
        not os.path.exists(args.hdmi_device)
        and time.monotonic() < device_deadline
    ):
        time.sleep(0.1)

    camera_mode = getattr(args, 'camera_fps_mode', 'off')
    if camera_mode != 'off':
        camera_status = configure_camera_fps(
            device=args.hdmi_device,
            width=args.hdmi_width,
            height=args.hdmi_height,
            fps=args.hdmi_fps,
            fourcc=args.hdmi_fourcc,
            mode=camera_mode,
            exposure=getattr(args, 'camera_exposure', None),
            gain=getattr(args, 'camera_gain', None),
            strict=getattr(args, 'camera_config_strict', False),
            printer=print if getattr(args, 'debug', False) else (lambda _msg: None),
        )
        if camera_status.warning and not getattr(args, 'debug', False):
            print(f"摄像头帧率警告: {camera_status.message}")
    hdmi_kwargs = {
        'device': args.hdmi_device,
        'width': args.hdmi_width,
        'height': args.hdmi_height,
        'fps': args.hdmi_fps,
        'fourcc': args.hdmi_fourcc,
        'jpeg_tap_fps': (
            args.publish_fps if args.publish else 0
        ),
    }
    if getattr(args, 'debug', False):
        hdmi_kwargs['profile'] = True
        hdmi_kwargs['profile_interval'] = DEBUG_INTERVAL
    recv_process = None
    last_error = None
    for attempt in range(1, 5):
        recv_process = NativeMppHdmiReceiver(frame_queue, **hdmi_kwargs)
        try:
            recv_process.start()
            break
        except RuntimeError as exc:
            last_error = exc
            recv_process.terminate()
            recv_process.join(timeout=1.0)
            if attempt == 4:
                raise
            delay = 0.5 * attempt
            print(
                f"HDMI采集设备暂不可用，{delay:.1f}s后重试"
                f"({attempt}/4): {exc}",
                flush=True,
            )
            time.sleep(delay)
    if recv_process is None or not recv_process.is_alive():
        raise RuntimeError(f"HDMI采集启动失败: {last_error}")
    print("HDMI native-mpp 采集线程已启动！")

    return frame_queue, recv_process


def validate_args(args):
    """校验命令行参数组合，在启动硬件资源前报告配置错误。"""
    validate_publish_args(args)
    low_conf_defaults = {
        'off': args.det_conf_high,
        'balanced': 0.18,
        'high': 0.12,
    }
    if args.det_sensitivity_mode not in low_conf_defaults:
        raise ValueError("--det-sensitivity-mode 只能是 off、balanced 或 high")
    if args.det_sensitivity_mode == 'off':
        args.det_conf_low = args.det_conf_high
    elif args.det_conf_low is None:
        args.det_conf_low = low_conf_defaults[args.det_sensitivity_mode]

    if args.queue_size < 1:
        raise ValueError("--queue-size 必须 >= 1")
    if args.async_result_ttl < 1:
        raise ValueError("--async-result-ttl 必须 >= 1")
    if args.result_log_interval < 1:
        raise ValueError("--result-log-interval 必须 >= 1")
    if args.det_conf_high <= 0 or args.det_conf_high > 1:
        raise ValueError("--det-conf-high 必须在 0 到 1 之间，且不能为 0")
    if args.det_conf_low <= 0 or args.det_conf_low > args.det_conf_high:
        raise ValueError("--det-conf-low 必须在 0 到 --det-conf-high 之间，且不能为 0")
    if args.det_low_confirm_frames < 1:
        raise ValueError("--det-low-confirm-frames 必须 >= 1")
    if args.det_max_candidates < 1:
        raise ValueError("--det-max-candidates 必须 >= 1")
    camera_exposure = getattr(args, 'camera_exposure', None)
    if camera_exposure is not None and camera_exposure < 0:
        raise ValueError("--camera-exposure 必须 >= 0")
    camera_gain = getattr(args, 'camera_gain', None)
    if camera_gain is not None and camera_gain < 0:
        raise ValueError("--camera-gain 必须 >= 0")
    if args.hdmi_width <= 0:
        raise ValueError("--hdmi-width 必须 > 0")
    if args.hdmi_height <= 0:
        raise ValueError("--hdmi-height 必须 > 0")
    if args.hdmi_fps <= 0:
        raise ValueError("--hdmi-fps 必须 > 0")
    args.hdmi_fourcc = getattr(args, 'hdmi_fourcc', 'MJPG')


def should_log_result(frame_count, interval, log_state):
    """执行“should log result”流程，保持原有输入输出和副作用不变。"""
    interval = max(1, int(interval))
    last_frame = log_state.get('last_result_log_frame')
    if last_frame is None or frame_count - last_frame >= interval:
        log_state['last_result_log_frame'] = frame_count
        return True
    return False


def raise_if_receiver_failed(recv_process):
    """执行“raise if receiver failed”流程，保持原有输入输出和副作用不变。"""
    error = getattr(recv_process, "error", None)
    if error is None:
        return
    if isinstance(error, BaseException):
        raise RuntimeError(f"接收进程运行失败: {error}") from error
    raise RuntimeError(f"接收进程运行失败: {error}")



def native_display_worker(hdmi, display_q, stop_event, counters, status_q):
    """独立刷新DRM画面和Overlay，检测或识别变慢时仍维持视频显示。"""
    held_frames = {}
    current_frame = None
    try:
        while not stop_event.is_set() or not display_q.empty():
            if current_frame is None:
                try:
                    frame = display_q.get(timeout=0.05)
                except thread_queue.Empty:
                    continue
            else:
                try:
                    frame = display_q.get_nowait()
                except thread_queue.Empty:
                    frame = current_frame
            try:
                started = time.perf_counter()
                retired_token = hdmi.show(frame)
                held_frames[frame.token] = frame
                current_frame = frame
                counters.inc('displayed')
                counters.add_time(
                    'display_total_ms',
                    'display_max_ms',
                    (time.perf_counter() - started) * 1000,
                )
                retired = held_frames.pop(retired_token, None)
                if retired is not None:
                    retired.release()
            except Exception:
                if frame is not current_frame:
                    frame.release()
                raise
    except Exception as exc:
        status_q.put(('error', 'atomic-display-worker', str(exc)))
        stop_event.set()
    finally:
        while True:
            try:
                release_native_frame(display_q.get_nowait())
            except thread_queue.Empty:
                break
        hdmi.close()
        for frame in held_frames.values():
            frame.release()


def drain_display_status(status_q):
    """执行“drain display status”流程，保持原有输入输出和副作用不变。"""
    try:
        status = status_q.get_nowait()
    except thread_queue.Empty:
        return
    if status[0] == 'error':
        raise RuntimeError(f"{status[1]}运行失败: {status[2]}")


def _overlay_signature(results):
    """生成Overlay内容签名，仅在显示内容变化时更新标注缓冲。"""
    signature = []
    for result in results:
        bbox = tuple(int(value) for value in result.get('bbox', (0, 0, 0, 0)))
        signature.append((
            bbox,
            str(result.get('plate_no', '')),
            round(float(result.get('detect_conf', 0.0)), 3),
            round(float(result.get('rec_conf', 0.0)), 3),
        ))
    return tuple(signature)


def _native_wait_timeout(state, ttl_ms, now=None, health_interval=0.2):
    """返回下次TTL到期或采集线程健康检查前的阻塞时间。"""
    current = time.monotonic() if now is None else float(now)
    deadline = current + float(health_interval)
    ttl_s = max(0.0, float(ttl_ms) / 1000.0)
    for key in ('latest_detection_result', 'latest_recognition_result'):
        result = state.get(key)
        if result is not None:
            deadline = min(deadline, float(result['timestamp']) + ttl_s)
    return max(0.0, deadline - current)


def run_native_zero_copy(args, hdmi, frame_queue, recv_process):
    """运行车牌模式的NV12零拷贝采集、异步推理和Atomic显示链路。"""
    print("NV12 DMA-BUF + DRM Atomic零拷贝流水线已启动")
    wake_event = getattr(frame_queue, 'wake_event', threading.Event())
    runtime = start_async_runtime(args, wake_event=wake_event)
    state = build_async_state()
    text_stabilizer = PlateTextStabilizer(
        iou_threshold=0.5,
        first_confirmations=2,
        switch_confirmations=3,
        track_ttl_ms=1200,
    )
    display_q = thread_queue.Queue(maxsize=1)
    display_status_q = NotifyingQueue(wake_event=wake_event)
    display_stop = threading.Event()
    display_thread = threading.Thread(
        target=native_display_worker,
        name="atomic-display-worker",
        args=(
            hdmi,
            display_q,
            display_stop,
            runtime['counters'],
            display_status_q,
        ),
        daemon=True,
    )
    display_thread.start()
    last_overlay_signature = None
    overlay_results = []
    overlay_metric_result = None
    overlay_source_ns = 0
    overlay_ready_ns = 0
    last_stable_signature = ()
    publisher = FramePublisher.from_args(
        args,
        mode="plate_ai",
        jpeg_provider=recv_process.read_latest_jpeg,
    )
    if publisher is not None:
        publisher.start()
    publish_started = time.monotonic()
    page_flip_fps = 0.0
    latency_tracker = DisplayedLatencyTracker()
    displayed_stage_tracker = DisplayedStageTracker()
    state['displayed_stage_tracker'] = displayed_stage_tracker
    state['displayed_stage_ttl_ms'] = args.async_result_ttl
    publish_snapshots = OrderedDict()
    try:
        while True:
            # 先清除上次唤醒，再清空所有队列；清空期间的新入队会重新置位，
            # 因而不会在检查与阻塞之间丢失通知。
            wake_event.clear()
            drain_worker_status(runtime['status_q'], runtime['stop_event'])
            drain_display_status(display_status_q)
            frame_processed = False
            try:
                frame = frame_queue.get_nowait()
                frame_processed = True
            except thread_queue.Empty:
                if not recv_process.is_alive():
                    raise_if_receiver_failed(recv_process)
                    return
            if frame_processed:
                state['frame_count'] += 1
            frame_count = state['frame_count']
            if frame_processed:
                frame_timestamp = frame.timestamp_ns / 1_000_000_000.0
                runtime['counters'].inc('received')

                display_frame = frame.retain()
                put_latest_owned(display_q, display_frame)

                if (
                    args.skip == 0
                    or (frame_count - 1) % (args.skip + 1) == 0
                ):
                    detect_item = {
                        'frame_id': frame_count,
                        'timestamp': frame_timestamp,
                        'source_timestamp_ns': frame.timestamp_ns,
                        'e2e_source_timestamp_ns': (
                            frame_e2e_source_timestamp_ns(frame)
                        ),
                        'decoded_timestamp_ns': frame.decoded_timestamp_ns,
                        'submitted_ns': time.monotonic_ns(),
                        'frame': frame,
                    }
                    dropped = put_latest_owned(
                        runtime['detect_q'], detect_item
                    )
                    runtime['counters'].inc('submitted_detect')
                    if dropped:
                        runtime['counters'].inc('dropped_detect', dropped)
                else:
                    frame.release()

            overlay_dirty = False
            detection = drain_latest(runtime['draw_q'])
            if detection is not None:
                overlay_dirty = True
                detect_ready_ns = int(detection.get('detect_finished_ns', 0) or 0)
                detection['result_poll_ms'] = max(
                    0.0,
                    (time.monotonic_ns() - detect_ready_ns) / 1_000_000.0,
                ) if detect_ready_ns > 0 else 0.0
                state['latest_detection_result'] = detection
            recognition = drain_latest(runtime['result_q'])
            if recognition is not None:
                overlay_dirty = True
                recognition_ready_ns = int(
                    recognition.get(
                        'recognition_finished_ns',
                        recognition.get('detect_finished_ns', 0),
                    ) or 0
                )
                recognition['result_poll_ms'] = max(
                    0.0,
                    (time.monotonic_ns() - recognition_ready_ns) / 1_000_000.0,
                ) if recognition_ready_ns > 0 else 0.0
                state['latest_recognition_result'] = recognition

            now = time.monotonic()
            draw_detections = []
            if recognition is not None:
                text_stabilizer.update(
                    recognition.get('plate_results', []), now=now
                )
            recognized_results = text_stabilizer.current(now=now)
            stable_signature = _overlay_signature(recognized_results)
            if stable_signature != last_stable_signature:
                overlay_dirty = True
                last_stable_signature = stable_signature
            latest_detection = state['latest_detection_result']
            if latest_detection is not None:
                age_ms = (now - latest_detection['timestamp']) * 1000
                if age_ms <= args.async_result_ttl:
                    draw_detections = latest_detection.get('det_results_for_draw', [])
                else:
                    state['latest_detection_result'] = None
                    latest_detection = None
                    overlay_dirty = True
                    runtime['counters'].inc('stale_result')
            latest_recognition = state['latest_recognition_result']
            if latest_recognition is not None:
                age_ms = (now - latest_recognition['timestamp']) * 1000
                if age_ms > args.async_result_ttl:
                    state['latest_recognition_result'] = None
                    latest_recognition = None
                    overlay_dirty = True
                    runtime['counters'].inc('stale_result')

            # Do not combine geometry from a new detection frame with text
            # and timestamps from an older recognition frame.  A complete
            # recognition result already carries the bbox/landmarks of its
            # own source frame; otherwise publish the detection-only result.
            if overlay_dirty:
                if recognized_results:
                    overlay_results = merge_draw_results(
                        draw_detections or (
                            latest_recognition.get('det_results_for_draw', [])
                            if latest_recognition is not None else []
                        ),
                        recognized_results,
                    )
                    overlay_metric_result = (
                        latest_recognition or latest_detection
                    )
                else:
                    overlay_results = merge_draw_results(draw_detections, [])
                    overlay_metric_result = latest_detection
                overlay_source_ns = (
                    int(
                        overlay_metric_result.get(
                            'e2e_source_timestamp_ns',
                            overlay_metric_result['source_timestamp_ns'],
                        )
                    )
                    if overlay_metric_result is not None else 0
                )
                overlay_ready_ns = (
                    int(
                        overlay_metric_result.get(
                            'recognition_finished_ns',
                            overlay_metric_result.get(
                                'detect_finished_ns', 0
                            ),
                        )
                    )
                    if overlay_metric_result is not None else 0
                )
            detection_latency_ms, recognition_latency_ms = (
                ai_latency_values(overlay_metric_result)
            )
            latency_tracker.record(
                overlay_source_ns,
                overlay_ready_ns,
                detection_latency_ms,
                recognition_latency_ms,
            )
            if overlay_metric_result is not None:
                displayed_stage_tracker.record(
                    overlay_source_ns,
                    overlay_ready_ns,
                    {
                        'frame_id': overlay_metric_result.get('frame_id'),
                        'processing_decode_ms': overlay_metric_result.get(
                            'processing_decode_ms'
                        ),
                        'detect_queue_ms': overlay_metric_result.get(
                            'detect_queue_ms'
                        ),
                        'detect_ms': overlay_metric_result.get('detect_ms'),
                        'detect_npu_wait_ms': overlay_metric_result.get(
                            'detect_npu_wait_ms'
                        ),
                        'detect_compute_ms': overlay_metric_result.get(
                            'detect_compute_ms'
                        ),
                        'native_detection_timing': (
                            overlay_metric_result.get(
                                'native_detection_timing', {}
                            )
                        ),
                        'recognition_queue_ms': overlay_metric_result.get(
                            'recognition_queue_ms'
                        ),
                        'plate_ms': overlay_metric_result.get('plate_ms'),
                        'recognition_npu_ms': overlay_metric_result.get(
                            'recognition_npu_ms'
                        ),
                        'recognition_npu_wait_ms': (
                            overlay_metric_result.get(
                                'recognition_npu_wait_ms'
                            )
                        ),
                        'plate_compute_ms': overlay_metric_result.get(
                            'plate_compute_ms'
                        ),
                        'recognition_candidate_count': (
                            overlay_metric_result.get(
                                'recognition_candidate_count'
                            )
                        ),
                        'result_poll_ms': overlay_metric_result.get(
                            'result_poll_ms'
                        ),
                    },
                )
            if overlay_source_ns > 0:
                publish_snapshots.pop(overlay_source_ns, None)
                publish_snapshots[overlay_source_ns] = {
                    "overlays": plate_overlays(overlay_results, "rk"),
                    "results": [
                        {
                            "plate_no": result.get("plate_no", ""),
                            "confidence": result.get("rec_conf", 0.0),
                        }
                        for result in overlay_results
                        if result.get("plate_no")
                    ],
                }
                while len(publish_snapshots) > 64:
                    publish_snapshots.popitem(last=False)
            if publisher is not None and frame_processed:
                if frame_count % DEBUG_INTERVAL == 0:
                    page_flip_fps = hdmi.get_stats().display_fps
                counts, times = runtime['counters'].snapshot(reset=False)
                elapsed = max(time.monotonic() - publish_started, 1e-6)
                atomic_stats = hdmi.get_stats()
                published_latency = latency_tracker.resolve(
                    atomic_stats,
                    args.async_result_ttl,
                )
                published_snapshot = publish_snapshots.get(
                    int(
                        getattr(
                            atomic_stats,
                            "last_overlay_source_timestamp_ns",
                            0,
                        )
                        or 0
                    ),
                    {"overlays": [], "results": []},
                )
                publisher.update(
                    overlays=published_snapshot["overlays"],
                    results=published_snapshot["results"],
                    metrics={
                        "capture_fps": frame_count / elapsed,
                        "page_flip_fps": page_flip_fps,
                        "detect_fps": counts["finished_detect"] / elapsed,
                        "recognize_fps": counts["finished_rec"] / elapsed,
                        "detection_latency_ms": published_latency[
                            "detection_latency_ms"
                        ],
                        "recognition_latency_ms": published_latency[
                            "recognition_latency_ms"
                        ],
                        "total_latency_ms": published_latency[
                            "total_latency_ms"
                        ],
                        "dropped_frames": recv_process.native_drop_count(),
                        "skipped_frames": counts["dropped_detect"],
                    },
                )
            signature = (
                _overlay_signature(overlay_results),
                overlay_source_ns,
            )
            if signature != last_overlay_signature:
                started = time.perf_counter()
                hdmi.submit_overlay(
                    overlay_results,
                    ttl_ms=args.async_result_ttl,
                    draw_mode=args.draw_mode,
                    source_timestamp_ns=overlay_source_ns,
                    ready_timestamp_ns=overlay_ready_ns,
                )
                runtime['counters'].add_time(
                    'draw_total_ms',
                    'draw_max_ms',
                    (time.perf_counter() - started) * 1000,
                )
                runtime['counters'].inc('composed')
                last_overlay_signature = signature

            result_str = " ".join(
                str(result.get('plate_no', ''))
                for result in overlay_results
                if result.get('plate_no')
            )
            state['last_result_str'] = result_str

            if args.debug and frame_processed:
                state['profile_count'] += 1
                if state['profile_count'] >= DEBUG_INTERVAL:
                    profile_now = time.perf_counter()
                    print_debug_stats(
                        frame_count,
                        state['profile_count'],
                        profile_now - state['profile_last_t'],
                        runtime['counters'],
                        recv_process,
                        state,
                        hdmi,
                    )
                    state['profile_count'] = 0
                    state['profile_last_t'] = profile_now
            if frame_processed and result_str and should_log_result(
                frame_count, args.result_log_interval, state
            ):
                print(f"[{frame_count}] 车牌: {result_str}")
            wake_event.wait(
                _native_wait_timeout(state, args.async_result_ttl)
            )
    finally:
        if publisher is not None:
            publisher.stop()
        display_stop.set()
        display_thread.join(timeout=3.0)
        display_error = None
        try:
            drain_display_status(display_status_q)
        except RuntimeError as exc:
            display_error = exc
        stop_async_runtime(runtime)
        if display_error is not None:
            raise display_error



def main():
    """解析命令行参数并启动当前程序的既定运行流程。"""
    args = build_args()
    try:
        validate_args(args)
    except ValueError as e:
        print(f"参数错误: {e}")
        sys.exit(2)

    print("正在初始化运行环境...")

    hdmi = None
    recv_process = None
    exit_code = 0
    try:
        if args.hdmi_fourcc != 'MJPG':
            raise RuntimeError("AI模式仅支持1080p MJPG采集")
        hdmi = AtomicNv12Display(
            device="/dev/dri/card0",
            font_path=args.font_path,
        )
        print("底层模块全部加载完毕！")

        frame_queue, recv_process = start_receiver(args)

        run_native_zero_copy(args, hdmi, frame_queue, recv_process)
    except KeyboardInterrupt:
        print("\n程序被手动终止！正在清理资源...")
    except RuntimeError as e:
        exit_code = 1
        print(f"ERROR: {e}")
    finally:
        if recv_process is not None and recv_process.is_alive():
            recv_process.terminate()
            recv_process.join()
        if hdmi is not None:
            hdmi.close()
        print("资源清理完毕，安全退出。")
    if exit_code:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
