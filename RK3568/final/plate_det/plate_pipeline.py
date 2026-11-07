import cv2
import numpy as np

from .plate_recognizer import recognize_rknn
from .plate_recognition import get_split_merge

try:
    import core.rga_preprocess as _rga
except Exception:
    _rga = None


# ==========================================
# 透视变换
# ==========================================
def four_point_transform(image, pts):
    """透视变换校正车牌"""
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def rectify_plate_roi(img, landmarks):
    """按需提取小ROI，并依据四个角点完成车牌透视矫正。"""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("plate landmarks must be a finite 4x2 array")
    if hasattr(img, "dma_fd") and hasattr(img, "hor_stride"):
        if _rga is None or not _rga.available():
            raise RuntimeError("NV12 plate ROI requires RGA preprocessing")
        margin = 8
        x1 = max(0, int(np.floor(points[:, 0].min())) - margin)
        y1 = max(0, int(np.floor(points[:, 1].min())) - margin)
        x2 = min(int(img.width), int(np.ceil(points[:, 0].max())) + margin)
        y2 = min(int(img.height), int(np.ceil(points[:, 1].max())) + margin)
        crop, origin = _rga.nv12_roi_bgr(
            img, x1, y1, max(1, x2 - x1), max(1, y2 - y1)
        )
        local_points = points - np.asarray(origin, dtype=np.float32)
        return four_point_transform(crop, local_points)
    return four_point_transform(img, points)


def crop_plate_bbox(img, bbox):
    """Crop an axis-aligned fallback ROI when Pose26 landmarks are unreliable."""
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("plate bbox must be a finite xyxy array")
    x1, y1, x2, y2 = values
    if hasattr(img, "dma_fd") and hasattr(img, "hor_stride"):
        if _rga is None or not _rga.available():
            raise RuntimeError("NV12 plate ROI requires RGA preprocessing")
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(int(img.width), int(np.ceil(x2)))
        bottom = min(int(img.height), int(np.ceil(y2)))
        if right <= left or bottom <= top:
            raise ValueError("plate bbox is empty")
        crop, _origin = _rga.nv12_roi_bgr(
            img, left, top, right - left, bottom - top
        )
        return crop
    height, width = img.shape[:2]
    left = int(np.clip(np.floor(x1), 0, width))
    top = int(np.clip(np.floor(y1), 0, height))
    right = int(np.clip(np.ceil(x2), 0, width))
    bottom = int(np.clip(np.ceil(y2), 0, height))
    if right <= left or bottom <= top:
        raise ValueError("plate bbox is empty")
    return img[top:bottom, left:right].copy()


def pose26_landmarks_are_valid(det_result, visibility_threshold=0.25):
    """Return whether all Pose26 corners are visible and form a useful quad."""
    visibility = det_result.get("landmark_visibility")
    if visibility is None:
        return True
    visibility = np.asarray(visibility, dtype=np.float32).reshape(-1)
    points = np.asarray(det_result.get("landmarks"), dtype=np.float32)
    if visibility.shape != (4,) or points.shape != (4, 2):
        return False
    if not np.isfinite(visibility).all() or not np.isfinite(points).all():
        return False
    if np.any(visibility < float(visibility_threshold)):
        return False
    width = max(np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[3] - points[0]),
                 np.linalg.norm(points[2] - points[1]))
    return bool(width >= 8.0 and height >= 4.0)


# ==========================================
# 端到端车牌处理流水线
# ==========================================
def prepare_plate_image(img, det_result):
    """提取并校正单个车牌，返回不再引用整张Native帧的小图。"""
    landmarks = det_result['landmarks']
    class_label = det_result['class']

    if pose26_landmarks_are_valid(det_result):
        roi_img = rectify_plate_roi(img, landmarks)
    else:
        roi_img = crop_plate_bbox(img, det_result['bbox'])

    if class_label == 1:
        roi_img = get_split_merge(roi_img)
    return roi_img


def process_prepared_plate(roi_img, det_result, rec_rknn):
    """识别已经提取完成的小型车牌图，不再访问源NV12帧。"""
    class_label = det_result['class']
    landmarks = det_result['landmarks']
    if hasattr(landmarks, "tolist"):
        landmarks = landmarks.tolist()

    plate_text, plate_color, rec_conf, color_conf = recognize_rknn(rec_rknn, roi_img)

    result = {
        'plate_no': plate_text,
        'plate_color': plate_color,
        'rec_conf': float(rec_conf),
        'color_conf': float(color_conf),
        'plate_type': class_label,
        'bbox': det_result['bbox'],
        'landmarks': landmarks,
        'detect_conf': det_result['conf'],
    }
    if 'landmark_visibility' in det_result:
        visibility = det_result['landmark_visibility']
        result['landmark_visibility'] = (
            visibility.tolist() if hasattr(visibility, 'tolist') else visibility
        )
    return result


def process_plate(img, det_result, rec_rknn):
    """处理单个检测到的车牌。"""
    roi_img = prepare_plate_image(img, det_result)
    return process_prepared_plate(roi_img, det_result, rec_rknn)
