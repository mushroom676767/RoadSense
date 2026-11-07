#!/usr/bin/env python3
"""测量RKNN模型的纯NPU执行时间，不包含图像前后处理。"""

import argparse
from pathlib import Path
import sys

import numpy as np

FINAL_ROOT = Path(__file__).resolve().parents[1]
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from core.rknn_native import NativeRknnModel


DEFAULT_MODELS = (
    "/home/cat/RK3568/final/rknn_models/plate_detect_320_int8.rknn",
    "/home/cat/RK3568/final/rknn_models/plate_detect_640_int8.rknn",
)


def build_args():
    """构造可重复运行的模型性能测试参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def input_geometry(model):
    """从Native模型元数据读取NHWC测试输入尺寸。"""
    shape = tuple(model.input_info["shape"])
    if model.input_info["format"] == 1:
        batch, height, width, channels = shape
    elif model.input_info["format"] == 0:
        batch, channels, height, width = shape
    else:
        raise RuntimeError(f"不支持的输入布局: {model.input_info}")
    if batch != 1 or channels != 3:
        raise RuntimeError(f"测试仅支持单batch三通道模型: {shape}")
    return int(height), int(width)


def benchmark(path, warmup, iterations):
    """运行模型并返回驱动上报的纯NPU耗时统计。"""
    model = NativeRknnModel(path)
    try:
        height, width = input_geometry(model)
        tensor = np.full((1, height, width, 3), 114, dtype=np.uint8)
        for _ in range(warmup):
            model.inference([tensor], data_format="nhwc")

        samples = []
        for _ in range(iterations):
            model.inference([tensor], data_format="nhwc")
            samples.append(model.last_run_duration_ms())
        values = np.asarray(samples, dtype=np.float64)
        return {
            "input": f"{width}x{height}",
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()),
        }
    finally:
        model.release()


def main():
    """依次测试所有模型并以便于对比的单行格式输出。"""
    args = build_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup必须非负且iterations必须大于0")
    for model_path in args.models:
        result = benchmark(model_path, args.warmup, args.iterations)
        print(
            f"{Path(model_path).name} input={result['input']} "
            f"NPU mean={result['mean_ms']:.3f}ms "
            f"P50={result['p50_ms']:.3f}ms "
            f"P95={result['p95_ms']:.3f}ms "
            f"max={result['max_ms']:.3f}ms"
        )


if __name__ == "__main__":
    main()
