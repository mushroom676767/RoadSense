import cv2
import numpy as np

from .plate_recognition import plateName, color_labels


def numpy_softmax(x, axis=-1):
    """Compute a numerically stable softmax for the Python toolkit backend."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# ==========================================
# 车牌识别预处理
# ==========================================
def preprocess_rec(img):
    """车牌识别预处理: 168x48 BGR -> NCHW float32"""
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = img[None]
    return img


# ==========================================
# CTC 解码
# ==========================================
def ctc_decode(preds):
    """CTC解码: 去重去空白"""
    index = preds.argmax(axis=-1)
    pre = 0
    newPreds = []
    for i in range(len(index)):
        if index[i] != 0 and index[i] != pre:
            newPreds.append(index[i])
        pre = index[i]

    plate = "".join([plateName[i] for i in newPreds if i < len(plateName)])
    return plate


# ==========================================
# 车牌识别
# ==========================================
def recognize_rknn(rknn, plate_roi):
    """使用 RKNN 模型识别车牌文字和颜色"""
    native_ocr = getattr(rknn, "recognize_plate_bgr", None)
    if callable(native_ocr):
        indices, color_idx, text_conf, color_conf = native_ocr(plate_roi)
        plate_text = "".join(
            plateName[index]
            for index in indices
            if 0 <= index < len(plateName)
        )
        plate_color = (
            color_labels[color_idx]
            if 0 <= color_idx < len(color_labels)
            else "未知"
        )
        return plate_text, plate_color, text_conf, color_conf

    img = preprocess_rec(plate_roi)

    outputs = rknn.inference(inputs=[img], data_format='nchw')
    text_logits = outputs[0]
    color_logits = outputs[1]

    text_probs = numpy_softmax(text_logits[0], axis=-1)
    color_probs = numpy_softmax(color_logits[0], axis=-1)

    plate_text = ctc_decode(text_probs)
    text_conf = text_probs.max(axis=-1).mean()

    color_idx = color_probs.argmax()
    plate_color = color_labels[color_idx] if color_idx < len(color_labels) else '未知'
    color_conf = color_probs[color_idx]

    return plate_text, plate_color, text_conf, color_conf
