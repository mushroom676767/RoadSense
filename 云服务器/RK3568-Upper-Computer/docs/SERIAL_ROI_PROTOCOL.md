# FPGA串口ROI与上位机元数据契约

## 串口线协议

```text
0xAA
count: uint8，范围0～4
count × [x1, y1, x2, y2]: uint16 little-endian
```

坐标是1920×1080图像中的绝对像素XYXY，右边界和下边界按排他处理。每个ROI必须满足：

```text
0 <= x1 < x2 <= 1920
0 <= y1 < y2 <= 1080
```

`count=0`是有效清空包。RK收到后立即清空识别结果；`count>4`重新搜索帧头；任一ROI
越界或退化时拒绝整包并清空旧结果。

## RK处理链路

```text
FPGA 1～4个XYXY ROI
→ 约20%外扩并裁剪到1080p边界
→ RGA四槽160×320 RGB画布
→ YOLO26 Pose检测车牌框、四角点和单双层类别
→ OpenCV透视校正、双层拼接、168×48 BGR resize
→ Native OCR
→ HDMI及上位机
```

FPGA原始ROI仅用于候选区域和遥测展示，不直接作为最终车牌框绘制。视频画面使用
`source: "rk-roi"`的RK Pose完整画面坐标。

## 上位机`serial_data`

```json
{
  "online": true,
  "sequence": 1,
  "received_at_ms": 0,
  "age_ms": 0,
  "raw_hex": "AA 01 64 00 C8 00 90 01 2C 01",
  "count": 1,
  "rois": [
    {"x1": 100, "y1": 200, "x2": 400, "y2": 300}
  ]
}
```

Windows Agent和云端状态缓存透明转发该对象，不重排、外扩或改写坐标。前端验证数量、
几何范围和字段类型；异常数据只显示协议错误，不绘制ROI。`count=0`必须与空
`overlays`、空`results`一起覆盖上一帧状态。

串口性能面板使用RK当前发布的`detect_rga_ms`、`detect_npu_ms`和`detect_total_ms`；
不再使用旧接管状态或旧的合成时延字段。
