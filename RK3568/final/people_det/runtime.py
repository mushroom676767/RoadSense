"""行人闯红灯检测的异步Worker和各输入源运行时。

HDMI模式包含采集、显示、推理三个独立执行单元：显示逐帧Page Flip，推理按
``skip``获取最新帧。推理变慢只增加主动丢弃计数，不能阻塞采集或显示。
本地视频则保持逐帧语义，由OpenCV独立解码。

Native任务字典拥有一次帧引用。放入latest-wins队列、被覆盖、推理完成或线程
异常时都必须恰好释放一次；结果槽只保存坐标和时间戳，不保存整帧。
"""

import queue
import threading
import time

import cv2

from core.latency_metrics import (
    DisplayedLatencyTracker,
    DisplayedStageTracker,
    frame_e2e_source_timestamp_ns,
    people_latency_values,
)
from core.realtime_pipeline import (
    LatestValue,
    native_display_worker,
    put_latest,
    release_frame,
)
from core.publisher import FramePublisher, people_overlays


def _percentile(samples, quantile):
    """实现模块内部的“percentile”辅助逻辑，不作为新的外部接口。"""
    if not samples:
        return 0.0
    ordered = sorted(float(value) for value in samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.5))
    return ordered[index]


def _counts(detector, detections):
    """实现模块内部的“counts”辅助逻辑，不作为新的外部接口。"""
    return {
        "person": sum(
            detection["class_id"] in detector.person_ids
            for detection in detections
        ),
        "run_red_light": sum(
            detection["class_id"] in detector.run_red_ids
            for detection in detections
        ),
    }


def _print_result(frame_count, detector, detections, result_log_interval):
    """按既有日志格式输出result信息。"""
    counts = _counts(detector, detections)
    if counts["run_red_light"] > 0:
        print(
            f"[{frame_count}] 闯红灯! 行人:{counts['person']} "
            f"违规:{counts['run_red_light']}"
        )
    elif frame_count % max(1, result_log_interval) == 0:
        print(
            f"frames:{frame_count} person:{counts['person']} "
            f"run_red_light:{counts['run_red_light']}"
        )


def _inference_worker(detector, inference_queue, result_slot, stop_event,
                      error_queue, stats):
    """异步消费最新检测帧并执行预处理、NPU推理和后处理。"""
    rebuilds = 0
    try:
        while not stop_event.is_set():
            try:
                item = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame = item["frame"]
            frame_released = False
            inference_started_ns = time.monotonic_ns()
            submitted_ns = int(
                item.get("submitted_ns", inference_started_ns)
            )
            inference_queue_ms = max(
                0.0,
                (inference_started_ns - submitted_ns) / 1_000_000.0,
            )
            try:
                prepare_started = time.perf_counter()
                try:
                    try:
                        prepared = detector.prepare_nv12(frame)
                    except Exception:
                        if rebuilds >= 1:
                            raise
                        rebuilds += 1
                        detector.rebuild()
                        prepared = detector.prepare_nv12(frame)
                finally:
                    # RGA调用是同步的；模型输入准备完成后，
                    # NPU推理不再需要持有1080p DMA-BUF。
                    frame.release()
                    frame_released = True
                stats["prepare_ms"] = (
                    time.perf_counter() - prepare_started
                ) * 1000.0
                try:
                    detections = detector.infer_prepared(*prepared)
                except Exception:
                    if rebuilds >= 1:
                        raise
                    rebuilds += 1
                    detector.rebuild()
                    # 已准备的输入属于失败的RKNN上下文；
                    # 本次只跳过当前检测任务，下一张最新帧
                    # 会验证重建后的上下文，同时不阻塞视频显示。
                    detections = []
            finally:
                if not frame_released:
                    frame.release()
            inference_finished_ns = time.monotonic_ns()
            finished = inference_finished_ns / 1_000_000_000.0
            infer_ms = (
                inference_finished_ns - inference_started_ns
            ) / 1_000_000.0
            stats["infer_ms"] = infer_ms
            source_timestamp_ns = int(
                item.get(
                    "source_timestamp_ns",
                    float(item["source_ts"]) * 1_000_000_000.0,
                )
            )
            detection_latency_ms = max(
                0.0,
                (inference_finished_ns - source_timestamp_ns)
                / 1_000_000.0,
            )
            e2e_source_timestamp_ns = int(
                item.get("e2e_source_timestamp_ns", source_timestamp_ns)
            )
            decoded_timestamp_ns = int(
                item.get("decoded_timestamp_ns", source_timestamp_ns)
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
            processing_detection_latency_ms = max(
                0.0,
                (inference_finished_ns - e2e_source_timestamp_ns)
                / 1_000_000.0,
            )
            prepare_ms = float(stats.get("prepare_ms", 0.0))
            inference_compute_ms = max(0.0, infer_ms - prepare_ms)
            stats["detection_latency_ms"] = detection_latency_ms
            stats["processing_detection_latency_ms"] = (
                processing_detection_latency_ms
            )
            stats["decode_latency_ms"] = decode_latency_ms
            stats["processing_decode_ms"] = processing_decode_ms
            stats["inference_queue_ms"] = inference_queue_ms
            stats["inference_compute_ms"] = inference_compute_ms
            age_samples = stats.setdefault("age_samples", [])
            age_samples.append(detection_latency_ms)
            if len(age_samples) > 4096:
                del age_samples[:2048]
            stats["completed"] = stats.get("completed", 0) + 1
            result_slot.set({
                "detections": detections,
                "_ts": finished,
                "source_ts": item["source_ts"],
                "source_timestamp_ns": source_timestamp_ns,
                "e2e_source_timestamp_ns": e2e_source_timestamp_ns,
                "infer_ms": infer_ms,
                "inference_started_ns": inference_started_ns,
                "inference_finished_ns": inference_finished_ns,
                "detection_latency_ms": detection_latency_ms,
                "processing_detection_latency_ms": (
                    processing_detection_latency_ms
                ),
                "decoded_timestamp_ns": decoded_timestamp_ns,
                "decode_latency_ms": decode_latency_ms,
                "processing_decode_ms": processing_decode_ms,
                "inference_queue_ms": inference_queue_ms,
                "prepare_ms": prepare_ms,
                "inference_compute_ms": inference_compute_ms,
            })
    except Exception as exc:
        put_latest(error_queue, exc)
        stop_event.set()
    finally:
        while True:
            try:
                item = inference_queue.get_nowait()
            except queue.Empty:
                break
            release_frame(item)



def _start_inference(detector):
    """启动inference流程并返回其运行句柄。"""
    inference_queue = queue.Queue(maxsize=1)
    result_slot = LatestValue()
    error_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    stats = {
        "submitted": 0, "dropped": 0, "completed": 0,
        "infer_ms": 0.0, "detection_latency_ms": 0.0,
        "processing_detection_latency_ms": 0.0,
        "decode_latency_ms": 0.0, "inference_queue_ms": 0.0,
        "processing_decode_ms": 0.0,
        "prepare_ms": 0.0, "inference_compute_ms": 0.0,
        "age_samples": [],
    }
    thread = threading.Thread(
        target=_inference_worker,
        args=(
            detector, inference_queue, result_slot, stop_event, error_queue,
            stats,
        ),
        daemon=True,
        name="people-inference",
    )
    thread.start()
    return inference_queue, result_slot, error_queue, stop_event, stats, thread


def _result_or_empty(result_slot, ttl_ms):
    """实现模块内部的“result or empty”辅助逻辑，不作为新的外部接口。"""
    result = result_slot.get_fresh(ttl_ms)
    if result is not None and "result_poll_ms" not in result:
        ready_ns = int(result.get("inference_finished_ns", 0) or 0)
        result["result_poll_ms"] = max(
            0.0,
            (time.monotonic_ns() - ready_ns) / 1_000_000.0,
        ) if ready_ns > 0 else 0.0
    return result if result is not None else {"detections": []}


def _raise_worker_error(error_queue):
    """实现模块内部的“raise worker error”辅助逻辑，不作为新的外部接口。"""
    try:
        error = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"检测线程失败: {error}")


def _submit_detection_frame(inference_queue, item, stats):
    """把当前帧以latest-wins方式交给检测线程，主动释放被覆盖任务。"""
    dropped = put_latest(
        inference_queue, item, release_dropped=True
    )
    stats["submitted"] += 1
    stats["dropped"] += dropped


def _overlay_signature(items):
    """生成Overlay内容签名，仅在显示内容变化时更新标注缓冲。"""
    return tuple(
        (
            tuple(int(value) for value in item["bbox"]),
            int(item["class_id"]),
            round(float(item["score"]), 3),
        )
        for item in items
    )


def run_hdmi_native(detector, display, args):
    """运行HDMI NV12零拷贝采集、异步检测和DRM双Plane显示链路。"""
    from core.native_capture import NativeMppHdmiReceiver

    capture_queue = queue.Queue(maxsize=1)
    receiver = NativeMppHdmiReceiver(
        capture_queue,
        device=args.hdmi_device,
        width=args.hdmi_width,
        height=args.hdmi_height,
        fps=args.hdmi_fps,
        fourcc=args.hdmi_fourcc,
        profile=args.debug,
        profile_interval=60,
        jpeg_tap_fps=args.publish_fps if args.publish else 0,
    )
    receiver.start()
    values = _start_inference(detector)
    infer_queue, result_slot, error_queue, stop_event, stats, infer_thread = values
    display_queue = queue.Queue(maxsize=1)
    display_error = queue.Queue(maxsize=1)
    display_stats = {}
    display_thread = threading.Thread(
        target=native_display_worker,
        args=(
            display, display_queue, stop_event, display_error, display_stats
        ),
        daemon=True,
        name="people-atomic-display",
    )
    display_thread.start()
    frame_count = 0
    last_signature = None
    publisher = FramePublisher.from_args(
        args,
        mode="people_redlight",
        jpeg_provider=receiver.read_latest_jpeg,
    )
    if publisher is not None:
        publisher.start()
    publish_started = time.monotonic()
    page_flip_fps = 0.0
    latency_tracker = DisplayedLatencyTracker()
    displayed_stage_tracker = DisplayedStageTracker()
    try:
        while not stop_event.is_set():
            _raise_worker_error(error_queue)
            try:
                error = display_error.get_nowait()
                raise RuntimeError(f"Atomic显示线程失败: {error}")
            except queue.Empty:
                pass
            frame_processed = False
            try:
                frame = capture_queue.get(timeout=0.002)
                frame_processed = True
            except queue.Empty:
                if not receiver.is_alive():
                    raise RuntimeError(f"HDMI采集线程退出: {receiver.error}")
            if frame_processed:
                frame_count += 1
                source_ts = frame.timestamp_ns / 1_000_000_000.0
                if frame_count % (args.skip + 1) == 0:
                    _submit_detection_frame(
                        infer_queue,
                        {
                            "frame": frame.retain(),
                            "source_ts": source_ts,
                            "source_timestamp_ns": frame.timestamp_ns,
                            "e2e_source_timestamp_ns": (
                                frame_e2e_source_timestamp_ns(frame)
                            ),
                            "decoded_timestamp_ns": frame.decoded_timestamp_ns,
                            "submitted_ns": time.monotonic_ns(),
                        },
                        stats,
                    )
                put_latest(display_queue, frame, release_dropped=True)
            result = _result_or_empty(result_slot, args.async_result_ttl)
            overlay_items = detector.overlay_detections(
                result["detections"]
            )
            overlay_source_ns = int(
                result.get(
                    "e2e_source_timestamp_ns",
                    result.get("source_timestamp_ns", 0),
                )
            )
            overlay_ready_ns = int(
                result.get("inference_finished_ns", 0)
            )
            latency_tracker.record(
                overlay_source_ns,
                overlay_ready_ns,
                people_latency_values(result),
            )
            displayed_stage_tracker.record(
                overlay_source_ns,
                overlay_ready_ns,
                {
                    "processing_decode_ms": result.get(
                        "processing_decode_ms"
                    ),
                    "inference_queue_ms": result.get(
                        "inference_queue_ms"
                    ),
                    "prepare_ms": result.get("prepare_ms"),
                    "inference_compute_ms": result.get(
                        "inference_compute_ms"
                    ),
                    "detection_latency_ms": result.get(
                        "processing_detection_latency_ms"
                    ),
                    "result_poll_ms": result.get("result_poll_ms"),
                },
            )
            if frame_processed and frame_count % 60 == 0:
                page_flip_fps = display.get_stats().display_fps
            if publisher is not None and frame_processed:
                elapsed = max(time.monotonic() - publish_started, 1e-6)
                kms_stats = display.get_stats()
                published_latency = latency_tracker.resolve(
                    kms_stats,
                    args.async_result_ttl,
                )
                publisher.update(
                    overlays=people_overlays(
                        overlay_items,
                        detector.class_names,
                        detector.run_red_ids,
                    ),
                    metrics={
                        "capture_fps": frame_count / elapsed,
                        "page_flip_fps": page_flip_fps,
                        "detect_fps": stats["completed"] / elapsed,
                        "detection_latency_ms": published_latency[
                            "detection_latency_ms"
                        ],
                        "recognition_latency_ms": None,
                        "total_latency_ms": published_latency[
                            "total_latency_ms"
                        ],
                        "skipped_frames": stats["dropped"],
                        "dropped_frames": receiver.native_drop_count(),
                    },
                )
            signature = (
                _overlay_signature(overlay_items),
                overlay_source_ns,
            )
            if signature != last_signature:
                display.submit_detections(
                    overlay_items,
                    class_names=detector.class_names,
                    ttl_ms=args.async_result_ttl,
                    draw_mode=args.draw_mode,
                    source_timestamp_ns=overlay_source_ns,
                    ready_timestamp_ns=int(
                        overlay_ready_ns
                    ),
                )
                last_signature = signature
            if frame_processed and result["detections"]:
                _print_result(
                    frame_count, detector, result["detections"],
                    args.result_log_interval,
                )
            if args.debug and frame_processed and frame_count % 60 == 0:
                kms = display.get_stats()
                capture_stats = receiver.get_latest_stats_snapshot()
                displayed = displayed_stage_tracker.resolve(
                    kms, args.async_result_ttl
                ) or {}
                algorithm_e2e_ms = float(
                    displayed.get("algorithm_e2e_ms") or 0.0
                )
                print(
                    f"[调试 {frame_count}] "
                    f"PageFlip:{kms.display_fps:.1f}FPS | "
                    f"DQBUF→解码:{float(displayed.get('processing_decode_ms') or 0.0):.1f}ms | "
                    f"检测排队:{float(displayed.get('inference_queue_ms') or 0.0):.1f}ms | "
                    f"RGA输入:{float(displayed.get('prepare_ms') or 0.0):.1f}ms | "
                    f"NPU+后处理:{float(displayed.get('inference_compute_ms') or 0.0):.1f}ms | "
                    f"DQBUF→检测:{float(displayed.get('detection_latency_ms') or 0.0):.1f}ms "
                    f"P95:{_percentile(stats['age_samples'], 0.95):.1f}ms | "
                    f"结果调度:{float(displayed.get('result_poll_ms') or 0.0):.1f}ms | "
                    f"Overlay:{kms.overlay_total_ms / max(kms.overlay_updates, 1):.2f}ms | "
                    f"就绪→显示:{kms.last_overlay_display_ms:.1f}ms | "
                    f"端到端:{algorithm_e2e_ms:.1f}ms | "
                    f"主动丢弃:{stats['dropped']} "
                    f"| 压缩丢帧:{int(getattr(capture_stats, 'compressed_drops', 0))} "
                    f"| 帧池不足:{int(getattr(capture_stats, 'result_drops', 0))}"
                )
    finally:
        if publisher is not None:
            publisher.stop()
        stop_event.set()
        receiver.terminate()
        receiver.join(timeout=3.0)
        while True:
            try:
                release_frame(capture_queue.get_nowait())
            except queue.Empty:
                break
        display_thread.join(timeout=3.0)
        infer_thread.join(timeout=3.0)



def run_video(detector, display, source, show_window, args):
    """逐帧处理本地视频文件，不采用实时latest-wins丢帧策略。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")
    frame_count = 0
    detections = []
    time_acc = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            started = time.perf_counter()
            if frame_count % (args.skip + 1) == 0:
                detections = detector.infer(frame)
            output, counts = detector.draw(
                frame, detections, draw_mode=args.draw_mode
            )
            if display is not None:
                display.show(output)
            if show_window:
                cv2.imshow("people_redlight", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
            time_acc += time.perf_counter() - started
            if counts["run_red_light"] > 0:
                _print_result(
                    frame_count, detector, detections,
                    args.result_log_interval,
                )
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
    if frame_count:
        print(
            f"处理完成: {frame_count}帧, "
            f"平均{frame_count / max(time_acc, 1e-6):.2f} FPS"
        )
