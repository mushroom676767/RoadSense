import cv2
import numpy as np

from core.image_utils import letterbox, xywh2xyxy, clip_boxes
from core.rga_preprocess import letterbox_nv12_fd_size, letterbox_rgb_size
from core.rknn_native import NativeRknnModel


# ==========================================
# 配置
# ==========================================
DEFAULT_CLASSES = [
    "person",
    "red_light",
    "green_light",
    "run_red_light",
    "yellow_light",
    "car",
    "bicycle",
    "motorcycle",
]

MODEL_INPUT_W = 384
MODEL_INPUT_H = 640


# ==========================================
# RKNN 行人闯红灯检测器
# ==========================================
class RKNNDetector:
    """基于 RKNN 的 YOLOv5 行人闯红灯检测器"""

    def __init__(self, model_path, class_names, conf_thres=0.35,
                 iou_thres=0.45, backend="native-c"):
        """初始化实例状态；只建立本对象所需资源，不改变外部调用契约。"""
        self.model_path = model_path
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.backend = backend
        self.in_w = MODEL_INPUT_W
        self.in_h = MODEL_INPUT_H
        self._input = np.empty(
            (1, self.in_h, self.in_w, 3), dtype=np.uint8
        )
        self.rknn = self._open_model()

        print(f"RKNN 模型加载成功: {model_path}")
        print(f"   输入尺寸: {self.in_w}x{self.in_h}")
        print(f"   类别数量: {len(self.class_names)}")

        self.person_ids = set()
        self.run_red_ids = set()
        for i, name in enumerate(self.class_names):
            norm = name.strip().lower()
            if norm in {"person", "人"}:
                self.person_ids.add(i)
            elif norm in {"run_red_light", "闯红灯", "违规"}:
                self.run_red_ids.add(i)

    def _open_model(self):
        """加载RKNN模型并校验输入输出信息，失败时保持原错误语义。"""
        if self.backend == "native-c":
            model = NativeRknnModel(self.model_path)
            expected = self.in_w * self.in_h * 3
            if model.input_info["elements"] != expected:
                shape = model.input_info["shape"]
                model.release()
                raise RuntimeError(
                    f"模型输入尺寸与{self.in_w}x{self.in_h}不匹配: {shape}"
                )
            return model
        if self.backend != "python-toolkit":
            raise ValueError(f"未知RKNN后端: {self.backend}")
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError("python-toolkit后端需要rknn-toolkit2") from exc
        model = RKNN(verbose=False)
        if model.load_rknn(self.model_path) != 0:
            raise RuntimeError(f"加载RKNN模型失败: {self.model_path}")
        if model.init_runtime(target="rk3568") != 0:
            model.release()
            raise RuntimeError("初始化RKNN运行时失败")
        return model

    def rebuild(self):
        """释放失败的RKNN上下文并按原配置重建一次。"""
        self.close()
        self.rknn = self._open_model()

    def close(self):
        """释放 RKNN 资源（关闭 NPU 设备）"""
        if hasattr(self, 'rknn') and self.rknn:
            self.rknn.release()

    def preprocess(self, frame):
        """预处理: letterbox + 转RGB + uint8"""
        if self.backend == "native-c":
            tensor, scale, padding = letterbox_rgb_size(
                frame, self.in_w, self.in_h, out=self._input
            )
            return tensor, scale, padding[0], padding[1], "nhwc"
        lb_img, scale, pad_left, pad_top = letterbox(frame, (self.in_w, self.in_h))
        rgb = lb_img[:, :, ::-1]
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.uint8))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale, pad_left, pad_top, "nchw"

    def preprocess_nv12(self, frame):
        """由RGA把NV12 DMA-BUF转换为模型所需letterbox输入。"""
        tensor, scale, padding = letterbox_nv12_fd_size(
            frame, self.in_w, self.in_h, out=self._input
        )
        return tensor, scale, padding[0], padding[1], "nhwc"

    def postprocess(self, pred, scale, pad_left, pad_top, orig_w, orig_h):
        """后处理: 解析输出 + NMS"""
        pred = np.squeeze(pred)

        if pred.ndim == 3:
            pred = pred[0]

        if pred.ndim == 1:
            cols = 5 + len(self.class_names)
            if pred.size % cols == 0:
                pred = pred.reshape(-1, cols)
            else:
                return []

        if pred.ndim != 2 or pred.shape[1] < 6:
            if pred.shape[0] > 20 and pred.shape[0] > pred.shape[1]:
                pred = pred.T

        if pred.ndim != 2 or pred.shape[1] < 6:
            return []

        boxes = pred[:, :4]
        obj_conf = pred[:, 4]
        cls_conf = pred[:, 5:]

        if cls_conf.shape[1] == 0:
            return []

        class_ids = np.argmax(cls_conf, axis=1)
        class_scores = cls_conf[np.arange(len(class_ids)), class_ids]
        scores = obj_conf * class_scores

        keep = scores >= self.conf_thres
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes /= scale
        boxes = clip_boxes(boxes, orig_w, orig_h)

        selected_indices = []
        unique_cls = np.unique(class_ids)

        for cls_id in unique_cls:
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            nms_boxes = cls_boxes.copy()
            nms_boxes[:, 2] -= nms_boxes[:, 0]
            nms_boxes[:, 3] -= nms_boxes[:, 1]
            nms_idx = cv2.dnn.NMSBoxes(nms_boxes.tolist(), cls_scores.tolist(),
                                        self.conf_thres, self.iou_thres)
            if len(nms_idx) == 0:
                continue

            for idx in np.array(nms_idx).reshape(-1):
                selected_indices.append(cls_indices[idx])

        if not selected_indices:
            return []

        selected_indices = sorted(selected_indices)
        detections = []
        for idx in selected_indices:
            x1, y1, x2, y2 = boxes[idx]
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(scores[idx]),
                "class_id": int(class_ids[idx]),
            })
        return detections

    def _infer_preprocessed(self, input_tensor, scale, pad_left, pad_top,
                            data_format, width, height):
        """对已准备的持久输入缓冲执行RKNN推理和后处理。"""
        outputs = self.rknn.inference(
            inputs=[input_tensor], data_format=data_format
        )

        if isinstance(outputs, list) and len(outputs) > 0:
            pred = outputs[0]
        else:
            pred = outputs

        return self.postprocess(
            pred, scale, pad_left, pad_top, width, height
        )

    def infer(self, frame):
        """执行BGR推理。"""
        h, w = frame.shape[:2]
        values = self.preprocess(frame)
        return self._infer_preprocessed(*values, w, h)

    def infer_nv12(self, frame):
        """直接从NV12 DMA-BUF执行RGA前处理和推理。"""
        values = self.prepare_nv12(frame)
        return self.infer_prepared(*values)

    def prepare_nv12(self, frame):
        """RGA生成持久模型输入；返回后原NV12帧即可释放。"""
        if self.backend != "native-c":
            raise RuntimeError("分离式NV12前处理只支持native-c后端")
        scale, padding = self.rknn.prepare_nv12(frame)
        return (
            scale, padding[0], padding[1],
            int(frame.width), int(frame.height),
        )

    def infer_prepared(self, scale, pad_left, pad_top, width, height):
        """对已准备的RKNN输入推理，不再引用原始视频帧。"""
        outputs = self.rknn.inference_prepared()
        prediction = outputs[0] if isinstance(outputs, list) else outputs
        return self.postprocess(
            prediction, scale, pad_left, pad_top, width, height
        )

    def draw(self, frame, detections, draw_mode="fast"):
        """绘制行人框: 闯红灯红框，普通行人绿框"""
        counts = {"person": 0, "run_red_light": 0}

        for det in detections:
            cls_id = det["class_id"]

            if cls_id in self.run_red_ids:
                counts["run_red_light"] += 1
                color = (0, 0, 255)
            elif cls_id in self.person_ids:
                counts["person"] += 1
                color = (0, 255, 0)
            else:
                continue

            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if draw_mode == "full":
                label = self.class_names[cls_id] if 0 <= cls_id < len(
                    self.class_names
                ) else str(cls_id)
                cv2.putText(
                    frame, f"{label} {det['score']:.2f}",
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA,
                )

        return frame, counts

    def overlay_detections(self, detections):
        """把模型检测结果转换为通用Atomic Overlay项目。"""
        selected = []
        for detection in detections:
            class_id = detection["class_id"]
            if class_id in self.run_red_ids:
                rgba = (255, 0, 0, 255)
            elif class_id in self.person_ids:
                rgba = (0, 255, 0, 255)
            else:
                continue
            item = dict(detection)
            item["rgba"] = rgba
            if 0 <= class_id < len(self.class_names):
                item["label"] = self.class_names[class_id]
            selected.append(item)
        return selected
