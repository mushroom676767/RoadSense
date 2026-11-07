import re
import subprocess
from dataclasses import dataclass


class CameraFpsConfigError(RuntimeError):
    pass


@dataclass
class CameraFpsStatus:
    requested_fps: float
    actual_fps: float = None
    exposure_auto_priority: int = None
    exposure_auto: int = None
    manual_exposure: int = None
    warning: bool = False
    message: str = ""


def _run_v4l2(cmd, runner):
    """运行v4l2处理流程，保持现有数据与错误语义。"""
    return runner(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _parse_actual_fps(text):
    """解析actual fps输入并返回规范化结果。"""
    match = re.search(r"Frames per second:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
    return float(match.group(1)) if match else None


def _parse_exposure_auto_priority(text):
    """解析exposure auto priority输入并返回规范化结果。"""
    match = re.search(r"exposure_auto_priority[^\n]*value=([0-9]+)", text or "")
    return int(match.group(1)) if match else None


def _parse_exposure_auto(text):
    # 避免匹配到 exposure_auto_priority，用负向前瞻排除后面跟 "_priority"
    """解析exposure auto输入并返回规范化结果。"""
    match = re.search(r"exposure_auto(?!_priority)[^\n]*value=([0-9]+)", text or "")
    return int(match.group(1)) if match else None


def _check_result(result, command_desc, strict):
    """检查result结果并按原规则报告异常。"""
    if result.returncode == 0:
        return None
    message = f"{command_desc} failed: {result.stderr.strip() or result.stdout.strip()}"
    if strict:
        raise CameraFpsConfigError(message)
    return message


def configure_camera_fps(
    device,
    width,
    height,
    fps,
    fourcc,
    mode="stable30",
    exposure=None,
    gain=None,
    strict=False,
    runner=subprocess.run,
    printer=print,
):
    """配置UVC采集控制项，使输入帧节奏保持稳定。"""
    mode = str(mode or "stable30")
    fourcc = str(fourcc).upper()
    requested_fps = float(fps)
    warnings = []

    if mode not in {"stable30", "auto-exposure", "off"}:
        raise CameraFpsConfigError(f"unsupported camera FPS mode: {mode}")

    def run(cmd, desc):
        """执行“run”流程，保持原有输入输出和副作用不变。"""
        result = _run_v4l2(cmd, runner)
        warning = _check_result(result, desc, strict)
        if warning:
            warnings.append(warning)
        return result

    if mode == "stable30":
        priority_result = _run_v4l2(
            ["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_auto_priority=0"],
            runner,
        )
        if priority_result.returncode != 0:
            warning = _check_result(priority_result, "set exposure_auto_priority=0", strict)
            if warning:
                warnings.append(warning)
    elif mode == "auto-exposure":
        run(["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_auto_priority=1"],
            "set exposure_auto_priority=1")
        # 强制打开自动曝光（UVC: 3=光圈优先自动曝光），不再依赖摄像头默认值
        run(["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_auto=3"],
            "set exposure_auto=3")

    if mode != "off":
        effective_exposure = exposure
        if effective_exposure is not None:
            run(["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_auto=1"],
                "set manual exposure mode")
            run(["v4l2-ctl", "-d", device, "--set-ctrl", f"exposure_absolute={int(effective_exposure)}"],
                "set exposure_absolute")
        if gain is not None:
            run(["v4l2-ctl", "-d", device, "--set-ctrl", f"gain={int(gain)}"],
                "set gain")

        run([
            "v4l2-ctl",
            "-d",
            device,
            f"--set-fmt-video=width={int(width)},height={int(height)},pixelformat={fourcc}",
            f"--set-parm={int(fps)}",
        ], "set camera format/fps")

    status_result = run(["v4l2-ctl", "-d", device, "--all"], "read camera status")
    status_text = (status_result.stdout or "") + "\n" + (status_result.stderr or "")
    status = CameraFpsStatus(
        requested_fps=requested_fps,
        actual_fps=_parse_actual_fps(status_text),
        exposure_auto_priority=_parse_exposure_auto_priority(status_text),
        exposure_auto=_parse_exposure_auto(status_text),
        manual_exposure=int(effective_exposure) if mode != "off" and effective_exposure is not None else None,
    )

    if status.actual_fps is None:
        warnings.append("actual FPS is unknown")
    elif abs(status.actual_fps - requested_fps) > 0.25:
        warnings.append(f"actual FPS {status.actual_fps:.3f} != requested {requested_fps:.3f}")

    if strict and warnings:
        raise CameraFpsConfigError("; ".join(warnings))

    status.warning = bool(warnings)
    status.message = "; ".join(warnings)
    priority = "unknown" if status.exposure_auto_priority is None else status.exposure_auto_priority
    exp_auto = "unknown" if status.exposure_auto is None else status.exposure_auto
    actual = "unknown" if status.actual_fps is None else f"{status.actual_fps:.3f}"
    line = (
        f"[CAMERA-FPS] mode:{mode} requested:{requested_fps:.3f} "
        f"actual:{actual} exposure_auto:{exp_auto} exposure_auto_priority:{priority}"
    )
    if status.manual_exposure is not None:
        line += f" manual_exposure:{status.manual_exposure}"
    if status.warning:
        line += f" WARNING: {status.message}"
    printer(line)
    return status
