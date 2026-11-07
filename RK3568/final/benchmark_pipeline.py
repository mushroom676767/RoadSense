#!/usr/bin/env python3
"""Run fpga_ai for a fixed window and emit a machine-readable benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from tools.perf_metrics import (
    distribution,
    parse_capture_sample,
    parse_display_sample,
    summarize,
)


POSE26_MODEL = Path(
    "/home/cat/RK3568/final/rknn_models/"
    "plate_ai_pose26_512x288_int8.rknn"
)


def build_args():
    """构造命令行参数解析器，保持历史参数名称、选项和默认值不变。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("no-plate", "plate"), required=True)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("app_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def read_npu_load():
    """执行“read npu load”流程，保持原有输入输出和副作用不变。"""
    path = Path("/sys/kernel/debug/rknpu/load")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        completed = subprocess.run(
            ["sudo", "-n", "cat", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        text = completed.stdout
    percentages = [int(value) for value in re.findall(r"(\d+)\s*%", text)]
    if percentages:
        return sum(percentages) / len(percentages)
    values = [int(value) for value in re.findall(r"\d+", text)]
    return values[-1] if values else None


def read_temperature_c():
    """读取板上最高有效thermal zone温度。"""
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text().strip())
        except (OSError, ValueError):
            continue
        values.append(value / 1000.0 if value > 200 else value)
    return max(values) if values else None


def file_sha256(path):
    """流式计算模型哈希，避免把RKNN文件整体读入内存。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def git_commit(project_dir):
    """返回本次benchmark对应的代码提交。"""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    return completed.stdout.strip() or None


def main():
    """解析命令行参数并启动当前程序的既定运行流程。"""
    args = build_args()
    if args.warmup < 0 or args.duration <= 0:
        raise SystemExit("warmup must be >= 0 and duration must be > 0")

    project_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-u",
        str(project_dir / "fpga_ai.py"),
        "--debug",
        "--result-log-interval",
        "100000",
    ]
    app_args = list(args.app_args)
    if app_args[:1] == ["--"]:
        app_args.pop(0)
    command.extend(app_args)

    process = subprocess.Popen(
        command,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    timing = {"measure_started": None, "deadline": None}
    captures = []
    displays = []
    npu_samples = []
    cpu_samples = []
    temperature_samples = []
    output_lines = []

    def sampler():
        """执行“sampler”流程，保持原有输入输出和副作用不变。"""
        clock_ticks = os.sysconf("SC_CLK_TCK")
        previous_cpu = None
        previous_time = None
        while process.poll() is None:
            time.sleep(1.0)
            measure_started = timing["measure_started"]
            deadline = timing["deadline"]
            if deadline is not None and time.monotonic() >= deadline:
                return
            if measure_started is not None and time.monotonic() >= measure_started:
                value = read_npu_load()
                if value is not None:
                    npu_samples.append(value)
                temperature = read_temperature_c()
                if temperature is not None:
                    temperature_samples.append(temperature)
                try:
                    fields = Path(f"/proc/{process.pid}/stat").read_text().split()
                    cpu_ticks = int(fields[13]) + int(fields[14])
                    sample_time = time.monotonic()
                    if previous_cpu is not None:
                        cpu_samples.append(
                            100.0
                            * (cpu_ticks - previous_cpu)
                            / clock_ticks
                            / max(sample_time - previous_time, 1e-6)
                        )
                    previous_cpu = cpu_ticks
                    previous_time = sample_time
                except (OSError, ValueError, IndexError):
                    pass

    sampling_thread = threading.Thread(target=sampler, daemon=True)
    sampling_thread.start()

    try:
        assert process.stdout is not None
        while True:
            deadline = timing["deadline"]
            if deadline is not None and time.monotonic() >= deadline:
                break
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            sys.stdout.write(line)
            output_lines.append(line.rstrip())
            if timing["measure_started"] is None and (
                "零拷贝流水线已启动" in line
            ):
                timing["measure_started"] = time.monotonic() + args.warmup
                timing["deadline"] = timing["measure_started"] + args.duration
            measure_started = timing["measure_started"]
            if measure_started is None or time.monotonic() < measure_started:
                continue
            capture = parse_capture_sample(line)
            if capture is not None:
                captures.append(capture)
            display = parse_display_sample(line)
            if display is not None:
                displays.append(display)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        sampling_thread.join(timeout=2)

    result = summarize(captures, displays)
    result.update(
        {
            "scenario": args.scenario,
            "warmup_seconds": args.warmup,
            "duration_seconds": args.duration,
            "resource_usage": {
                "process_cpu_percent": (
                    distribution(cpu_samples) if cpu_samples else None
                ),
                "npu_load_percent": (
                    distribution(npu_samples) if npu_samples else None
                ),
                "temperature_c": {
                    "min": min(temperature_samples),
                    "max": max(temperature_samples),
                    "avg": sum(temperature_samples) / len(temperature_samples),
                } if temperature_samples else None,
            },
            "environment": {
                "git_commit": git_commit(project_dir),
                "pose26_model": str(POSE26_MODEL),
                "pose26_sha256": file_sha256(POSE26_MODEL),
                "system": platform.platform(),
                "python": platform.python_version(),
            },
            "exit_code": process.returncode,
            "command": command,
        }
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print("\n=== BENCHMARK JSON ===")
    print(encoded)
    return 0 if captures and displays else 2


if __name__ == "__main__":
    raise SystemExit(main())
