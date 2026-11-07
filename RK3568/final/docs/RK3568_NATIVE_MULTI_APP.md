# RK3568 多应用 Native 1080p60 流水线

## 固定后端

以下三个入口固定使用NV12零拷贝与DRM Atomic显示：

```bash
cd /home/cat/RK3568/final
python3 fpga_ai.py
python3 fpga_serial.py --port /dev/ttyUSB0
python3 people_redlight.py
```

本地视频没有NV12 DMA-BUF，由OpenCV解码后直接通过DRM-RGA显示；HDMI模式
固定使用Native路径。Native初始化失败会退出，不会自动降级。

## 数据流

串口识别：

```text
V4L2 MJPEG → MPP NV12 DMA-BUF ─→ DRM NV12 Plane
                              └→ 最新帧引用 + FPGA ROI
                                  → RGA 160×320四槽组图
                                  → Pose26四角点检测 → Native OCR
检测及识别结果 → C/FreeType ARGB Plane
```

行人闯红灯：

```text
V4L2 MJPEG → MPP NV12 DMA-BUF ─→ DRM NV12 Plane
                              └→ latest-wins → RGA 384×640 RGB → RKNN
检测框/类别/颜色 → C/FreeType ARGB Plane
```

采集、显示和NPU Worker互不等待。显示始终消费最新视频帧；检测或识别变慢时，只丢弃过期NPU任务。NV12帧由原子引用计数保护，DRM退休、RGA输入完成后才复用。

## 构建与验证

```bash
cd /home/cat/RK3568/final
make -C core/native clean all
python3 -m unittest discover -s tests -p 'test_*.py'
```

2026-07-23板端实测，预热10秒后连续60秒：

| 入口 | 采集 | Page Flip | NPU | Overlay | 丢帧 |
|---|---:|---:|---:|---:|---:|
| `fpga_ai.py` | 59.3–59.5 FPS | 59.3–59.5 FPS | 检测14.8–15.7 ms | 结果未变化时0 ms | 0 |
| `fpga_serial.py` | 59.3–59.4 FPS | 59.3–59.5 FPS | 识别4.9–8.6 ms | 平均约0.48 ms | 0 |
| `people_redlight.py` | 59.3–59.4 FPS | 59.3–59.4 FPS | 检测45.9–48.0 ms | 无结果更新时0 ms | 0 |

Native C与Python Toolkit使用同一输入的回归结果：

- 行人模型原始输出最大绝对误差：`0.0`。
- 车牌文字、颜色完全一致，文字及颜色置信度误差：`0.0`。
- Native行人程序连续启动/停止20次：`20/20`成功，无残留进程或设备占用。
- 串口坐标识别延迟P95约`7.3 ms`，C Overlay P99约`0.81 ms`。
- 行人检测结果年龄P95约`49.6 ms`。

性能日志和临时测试图像不进入Git。

## 故障处理

- Atomic、RGA或Native RKNN初始化失败：程序打印错误并退出。
- NPU运行错误：Worker重建模型一次，视频显示继续；第二次失败后安全退出。
- Overlay结果超过TTL：C侧清除标注，不影响视频Plane。
- 回退单阶段修改：在 `/home/cat/RK3568` 使用对应提交的 `git revert <commit>`。
