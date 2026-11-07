"""Fixed RKNN tensor contracts used by production plate runtimes."""

import math


POSE26_INPUT_SHAPE = (288, 512)
POSE26_STRIDES = (4, 8, 16, 32)
POSE26_CHANNELS = 18


def validate_pose26_model(model):
    """Strictly validate the fixed AI-mode Pose26 RKNN tensor contract."""
    input_info = getattr(model, "input_info", None)
    output_info = getattr(model, "output_info", None)
    if not input_info or output_info is None:
        raise ValueError("Pose26 AI mode requires native RKNN tensor metadata")
    if int(input_info.get("format", -1)) != 1:
        raise ValueError("Pose26 input must use NHWC layout")
    if tuple(int(value) for value in input_info.get("shape", ())) != (1, 288, 512, 3):
        raise ValueError(
            "Pose26 input must be RGB [1,288,512,3], got "
            f"{input_info.get('shape')}"
        )
    expected = tuple(
        (1, POSE26_CHANNELS, POSE26_INPUT_SHAPE[0] // stride,
         POSE26_INPUT_SHAPE[1] // stride)
        for stride in POSE26_STRIDES
    )
    actual = tuple(
        tuple(int(value) for value in item.get("shape", ()))
        for item in output_info
    )
    if actual != expected:
        raise ValueError(f"Pose26 outputs must be {expected}, got {actual}")
    for index, item in enumerate(output_info):
        if int(item.get("type", -1)) != 2 or int(item.get("quantization", -1)) != 2:
            raise ValueError(f"Pose26 output {index} must be affine INT8")
        scale = float(item.get("scale", 0.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Pose26 output {index} has invalid quantization scale")
    return POSE26_INPUT_SHAPE
