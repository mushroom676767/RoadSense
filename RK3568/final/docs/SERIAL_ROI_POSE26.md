# FPGA ROI串口车牌模式

## 启动

模型固定为：

```text
/home/cat/RK3568/final/rknn_models/plate_serial_roi_pose26_320x160_w8a8_fp16.rknn
SHA-256: FB90CA333E4E5DEC61E8231B887D1F11E90927DE340258E19A504BECF560B10C
```

生产和调试命令：

```bash
cd /home/cat/RK3568/final
python3 fpga_serial.py --port /dev/ttyUSB0
python3 fpga_serial.py --port /dev/ttyUSB0 --debug
```

可调参数以`python3 fpga_serial.py --help`为准。检测阈值默认`0.25`，可用
`--det-threshold`调整；识别模型可用`--rec-model`指定。串口检测和OCR后端固定
为Native C，不提供模型尺寸、布局模型、全帧fallback或Python Toolkit开关。

## 串口协议

```text
AA count [x1 y1 x2 y2] × count
```

- `count`为uint8，范围0～4。
- 每个坐标为little-endian uint16，基于1920×1080完整图像。
- XYXY的`x2/y2`为右/下排他边界。
- `count=0`立即清空车牌结果；非法数量重新搜索`AA`，非法坐标拒绝整包并清空。

上位机只收到`count`和`rois`，不再收到旧`points`、handover或IoU匹配状态。

## 模型合同

- 输入：`[1,160,320,3]` RGB UINT8 NHWC。
- 四个80×160槽按ROI0右上、ROI1左上、ROI2左下、ROI3右下排列。
- 输出：P2～P5四个`[1,18,H,W]` FP16张量。
- 类别0为单层，类别1为双层；角点顺序TL、TR、BR、BL。
- 后处理执行两阶段全局Top-K 300、按bbox中心分槽、每槽最高分，不执行NMS。

## 当前验证

- RKNN Runtime 2.3.2、驱动0.8.2下模型张量合同及哈希校验通过。
- Native C强制重编译无告警；全量99项Python测试通过。
- 新模型纯NPU复测（50次预热、300次运行）：mean 7.465ms、P50 7.436ms、
  P95 7.770ms、P99 7.895ms、max 8.036ms，相对既有约7.2/7.6ms基线回退
  小于5%。
- 当前板上`/dev/video9`未出现，未能在本次自动验证中采集HDMI帧；最终仍需接入
  HDMI和FPGA后完成零框、1～4牌、单双层、快速进出及59FPS/零丢帧现场验收。
