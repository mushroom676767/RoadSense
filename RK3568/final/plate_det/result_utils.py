"""车牌结果和边界框的通用判断工具。"""

import numpy as np


def bbox_iou(box_a, box_b):
    """计算两个 ``xyxy`` 边界框的交并比。"""
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def is_complete_plate_result(result):
    """车牌号有效即可显示；颜色只用于内部质量判断。"""
    return bool(result and result.get("plate_no"))
