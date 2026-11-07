import numpy as np
import cv2


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox缩放，保持宽高比，添加灰边填充。返回 (image, scale, pad_left, pad_top)"""
    h, w = image.shape[:2]
    new_w, new_h = int(new_shape[0]), int(new_shape[1])

    scale = min(new_w / w, new_h / h)
    resize_w, resize_h = int(round(w * scale)), int(round(h * scale))

    dw = (new_w - resize_w) / 2
    dh = (new_h - resize_h) / 2

    if (w, h) != (resize_w, resize_h):
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, scale, left, top


def letterbox_detect(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """旧车牌检测letterbox兼容接口。

    Native实时主链路已使用RGA版本；批处理或外部脚本仍可能依赖这里与YOLOv5
    一致的``auto/scaleFill/scaleup``行为，因此保留原签名和返回格式。
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def xywh2xyxy(boxes):
    """将 [x,y,w,h] 转换为 [x1,y1,x2,y2]"""
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def clip_boxes(boxes, img_w, img_h):
    """裁剪边界框到图像范围内"""
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h - 1)
    return boxes
