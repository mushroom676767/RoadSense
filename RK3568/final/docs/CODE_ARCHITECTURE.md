# RK3568生产代码结构与运行链路

## 模块边界

三个顶层脚本只负责进入对应的生产`main()`：

- `fpga_ai.py` → `plate_det.ai_runtime`
- `fpga_serial.py` → `plate_det.serial_runtime`
- `people_redlight.py` → `people_det.app`与`people_det.runtime`

`core`保存硬件和通用基础设施，包括V4L2/MPP采集、DRM Atomic显示、RGA、
Native RKNN、latest-wins队列、帧引用计数、上位机Publisher和时延统计。各包的
`__init__.py`不重导出实现，生产代码直接从职责模块导入。`core.fpga_io`仅作为
内部以太网实现保留，不出现在CLI或`core`公共命名空间。

## RK独立车牌模式

```text
V4L2 MJPEG → MPP NV12 DMA-BUF帧池 ─→ DRM视频Plane
                                  └→ RGA双缓冲Pose26输入 → NPU检测
                                      → OpenCV透视/拼接/168×48 BGR resize
                                      → Native OCR → ARGB Overlay
```

采集、检测、识别和显示异步运行。显示逐帧提交；检测与识别采用latest-wins，
不会阻塞60Hz视频。识别结果单次采用，颜色仅在内部质量过滤中使用。

## FPGA串口ROI车牌模式

```text
FPGA 1～4个1080p XYXY ROI → 绑定最新NV12帧 → RGA双输入缓冲四槽组图
→ 160×320 Pose26 → 全图四角点/单双层 → OpenCV车牌预处理 → Native OCR
→ HDMI Overlay与上位机rk-roi结果
```

ROI槽位固定为右上、左上、左下、右下，空槽填RGB 114。RGA准备线程与单NPU
线程并行，检测及最多四次OCR在同一NPU线程内串行。`count=0`和协议错误建立
清空屏障，旧任务完成后也不能重新发布。FPGA原始ROI和车牌颜色均不显示。

## 行人闯红灯模式

HDMI输入走同一NV12零拷贝采集与DRM Atomic显示链路，RGA生成384×640模型
输入，检测Worker独立运行。本地视频由OpenCV解码并通过DRM-RGA显示，显式
`python-toolkit`后端继续保留。默认`--skip=0`，即逐帧提交检测任务。

## 帧所有权与显示退休

`NativeNv12Frame`代表C帧池的一次引用。跨线程发布前必须`retain()`；队列覆盖、
异常、处理完成和退出都必须配对`release()`。DRM提交成功不代表帧可复用，只有
Page Flip返回退休token后才能释放扫描引用。关闭顺序为停止生产者、清空队列与
inflight、关闭显示，最后关闭采集上下文。

## Native生产接口

保留的Native能力包括NV12准备、双缓冲Pose26、四槽ROI Pose26、Native OCR、
通用RKNN推理、V4L2/MPP采集、DRM Atomic及本地BGR显示。旧NV16、JPEG显示、
RGA通用RGB resize、单调用NV12检测和调试输出复制接口不再导出。
