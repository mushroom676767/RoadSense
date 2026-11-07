# RK3568实时视觉系统

仓库只保留三套正式运行模式，均在RK3568板端的`/home/cat/RK3568/final`运行。

## 启动

```bash
cd /home/cat/RK3568/final
python3 fpga_ai.py
python3 fpga_serial.py --port /dev/ttyUSB0
python3 people_redlight.py
```

查看完整参数：

```bash
python3 fpga_ai.py --help
python3 fpga_serial.py --help
python3 people_redlight.py --help
```

三个入口均可使用`--publish URL`向上位机发送MJPEG和结果。行人模式还支持本地
视频文件及显式`--rknn-backend python-toolkit`。

## 目录

```text
final/
├── fpga_ai.py              RK独立车牌识别入口
├── fpga_serial.py          FPGA ROI串口车牌识别入口
├── people_redlight.py      行人闯红灯检测入口
├── plate_det/              车牌运行时、透视/拼接、OCR及模型合同
├── people_det/             行人模型、视频和HDMI运行时
├── core/                   采集、显示、RKNN、RGA、发布及并发基础设施
│   └── native/             五个生产动态库的C源码和Makefile
├── rknn_models/            四个部署模型（模型二进制不提交Git）
├── tests/                  生产合同和单元测试
├── tools/                  benchmark解析与性能工具
├── docs/                   当前架构、部署及性能文档
└── benchmark_pipeline.py   1080p60实时流水线benchmark
```

`core.fpga_io`保留为内部以太网模块，但不从`core`包或CLI公开。

## 模型

```text
people_redlight_yolov5_384x640_int8.rknn
plate_ai_pose26_512x288_int8.rknn
plate_ocr_168x48_int8.rknn
plate_serial_roi_pose26_320x160_w8a8_fp16.rknn
```

## 构建与测试

板端使用`/usr/lib/librknnrt.so`，对应的`rknn_api.h`安装在
`/home/cat/.local/include`，无需保留完整RKNN参考仓。

```bash
cd /home/cat/RK3568/final
make -C core/native clean all
python3 -m unittest discover -s tests -p 'test_*.py' -q
cd /home/cat/RK3568
python3 -m compileall -q final
```

完整HDMI和串口运行需要`/dev/video9`与`/dev/ttyUSB0`。设备未接入时，只执行
模型加载/推理冒烟、单元测试和静态验证，不以模拟设备代替实机验收。
