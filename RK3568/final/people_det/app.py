"""行人闯红灯应用配置与启动编排。

这里集中维护CLI、模型路径和类别文件。HDMI源固定进入NV12零拷贝运行时；
本地视频文件由OpenCV解码并通过DRM-RGA显示。入口只选择输入源，不参与逐帧
检测或显示。
"""

import argparse
import os
from pathlib import Path

from core.display_drm_atomic import AtomicNv12Display
from core.display_drm_rga import RgaDrmDisplay
from core.hdmi_io import HDMI_DEVICE
from core.publisher import add_publish_args, validate_publish_args
from people_det.people_detector import DEFAULT_CLASSES, RKNNDetector
from people_det.runtime import run_hdmi_native, run_video


def load_class_names(path):
    """从类别文件加载非空类别名称，并保持原有顺序。"""
    if not path or not os.path.exists(path):
        return DEFAULT_CLASSES
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return names or DEFAULT_CLASSES


def get_model_path():
    """解析命令行模型路径；未指定时使用行人闯红灯默认模型。"""
    script_dir = Path(__file__).parent.resolve()
    model_names = (
        "people_redlight_yolov5_384x640_int8.rknn",
    )
    model_path = Path("/home/cat/RK3568/final/rknn_models") / model_names[0]
    for candidate in (
        *(Path("/home/cat/RK3568/final/rknn_models") / name
          for name in model_names),
        *(script_dir / "rknn_models" / name for name in model_names),
        *(script_dir / name for name in model_names),
        *(Path("rknn_models") / name for name in model_names),
    ):
        if candidate.exists():
            return candidate
    return model_path


def build_args(argv=None):
    """构造命令行参数解析器，保持历史参数名称、选项和默认值不变。"""
    parser = argparse.ArgumentParser(description="行人闯红灯检测 - RKNN")
    parser.add_argument("video", nargs="?", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--rknn-backend",
        choices=["native-c", "python-toolkit"],
        default="native-c",
    )
    parser.add_argument(
        "--draw-mode", choices=["fast", "full"], default="fast"
    )
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--hdmi-device", default=HDMI_DEVICE)
    parser.add_argument("--hdmi-width", type=int, default=1920)
    parser.add_argument("--hdmi-height", type=int, default=1080)
    parser.add_argument("--hdmi-fps", type=int, default=60)
    parser.add_argument("--hdmi-fourcc", default="MJPG")
    parser.add_argument("--async-result-ttl", type=int, default=200)
    parser.add_argument("--result-log-interval", type=int, default=60)
    parser.add_argument(
        "--no-fps",
        action="store_false",
        dest="show_fps",
        help="关闭HDMI画面左上角的实时帧率",
    )
    parser.add_argument("--debug", action="store_true")
    add_publish_args(parser)
    return parser.parse_args(argv)


def validate_args(args):
    """校验命令行参数组合，在启动硬件资源前报告配置错误。"""
    validate_publish_args(args)
    if args.skip < 0:
        raise ValueError("skip不能为负数")
    if args.async_result_ttl < 0:
        raise ValueError("async-result-ttl不能为负数")
    if args.result_log_interval < 1:
        raise ValueError("result-log-interval必须大于0")
    if not args.video:
        if (args.hdmi_width, args.hdmi_height) != (1920, 1080):
            raise ValueError("native-zero-copy当前要求1920x1080输入")
        if args.hdmi_fourcc.upper() not in {"MJPG", "MJPEG"}:
            raise ValueError("native-zero-copy当前要求MJPEG输入")


def main(argv=None):
    """解析命令行参数并启动当前程序的既定运行流程。"""
    args = build_args(argv)
    validate_args(args)
    model_path = Path(args.model) if args.model else get_model_path()
    if not model_path.exists():
        raise RuntimeError(f"模型文件不存在: {model_path}")
    class_names = load_class_names(args.class_names)
    detector = RKNNDetector(
        model_path=str(model_path),
        class_names=class_names,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        backend=args.rknn_backend,
    )
    display = None
    print("=" * 60)
    print("行人闯红灯检测系统")
    print(
        f"输入:{'本地视频' if args.video else 'HDMI native-zero-copy'} "
        f"RKNN:{args.rknn_backend} "
        f"绘制:{args.draw_mode}"
    )
    print("=" * 60)
    try:
        if args.video:
            try:
                display = RgaDrmDisplay(show_fps=args.show_fps)
            except Exception:
                print("HDMI输出不可用，本地视频继续以窗口/无头模式运行")
            run_video(detector, display, args.video, args.show, args)
        else:
            display = AtomicNv12Display(font_path=args.font_path)
            run_hdmi_native(detector, display, args)
    except KeyboardInterrupt:
        print("\n程序被终止，正在清理...")
    finally:
        if display is not None:
            display.close()
        detector.close()
        print("资源清理完毕。")


if __name__ == "__main__":
    main()
