# 集创赛 · RK3568 实时视觉与云端管理系统

基于 RK3568、FPGA ROI 串口输入和云端管理平台的实时视觉项目，包含车牌识别、
行人闯红灯检测，以及 Windows Agent 视频转发与远程控制。

## 功能

- **AI 车牌识别**：RK3568 NPU 运行车牌检测、Pose 定位和 OCR。
- **FPGA ROI 车牌识别**：接收 FPGA 串口候选区域，完成后续检测与识别。
- **行人闯红灯检测**：处理目标、违规事件和性能数据。
- **云端管理**：Flask / Vue 管理平台，支持识别结果、事件和远程模式控制。
- **视频转发**：Windows Agent 通过 NVENC / H.264 / SRT 上行，浏览器通过 WebRTC 或 LL-HLS 查看。

## 目录

| 目录 | 内容与说明 |
| --- | --- |
| [RK3568](RK3568/README.md) | 板端 Python / C 源码、原生库构建文件、测试和部署文档 |
| [云服务器/RK3568-Upper-Computer](云服务器/RK3568-Upper-Computer/README.md) | 上位机、Windows Agent、Vue 前端、云部署模板与测试 |
| [模型说明](RK3568/final/rknn_models/README.md) | 运行所需模型名称和放置位置 |

## 板端启动

需要 RK3568 Linux 设备、匹配的 RKNN Runtime、RGA / DRM / V4L2 开发环境。
默认板端目录为 `/home/cat/RK3568/final`，默认设备包括 `/dev/video9` 和 `/dev/ttyUSB0`；
请根据自己的硬件调整参数。先准备模型并在板端构建原生库：

```bash
cd /home/cat/RK3568/final
make -C core/native all
# 以下三种模式选择一种运行
python3 fpga_ai.py
python3 fpga_serial.py --port /dev/ttyUSB0
python3 people_redlight.py
```

使用 `--help` 查看各入口参数。完整说明见 [板端 README](RK3568/README.md)。

## 上位机和云端

进入 `云服务器/RK3568-Upper-Computer`，按照其 [README](云服务器/RK3568-Upper-Computer/README.md)
安装依赖、复制 `.env.example` 并填写自己的配置。公网部署见
[云端部署文档](云服务器/RK3568-Upper-Computer/docs/CLOUD_AGENT_DEPLOYMENT.md)。
文档中的 `203.0.113.10` 为示例地址，不能用于实际连接。

## 发布范围

本仓库发布板端和上位机源码。上位机源码从本地 `RK3568-Upper-Computer.zip` 整理而来；
同目录的 `rk3568-manager-4b36e7e.tar.gz` 是另一份部署快照，没有重复上传。

下列本地资源未纳入 Git：

- `CRPD.zip`、`plate_pose.7z`、`plate_pose_roi.7z`：数据集、图片和标注，合计约 33 GB。
- RKNN 模型权重、编译生成的 `.so`、Python 缓存和依赖环境。
- 数据库、事件截图、日志、运行时配置、密钥及真实连接凭据。

数据集和模型需自行准备；仓库不附带下载链接。完整推理和视频链路验收需要对应硬件。

## 验证

板端逻辑测试在 `RK3568/final` 目录运行：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

上位机测试和前端构建在 `云服务器/RK3568-Upper-Computer` 目录运行：

```bash
python -m unittest discover -s tests -v
npm ci --prefix frontend
npm test --prefix frontend
npm run build --prefix frontend
```

硬件依赖和运行环境要求见各子目录说明；本地静态检查不能替代板端实机验证。

## 许可证

项目自有源码按 [MIT License](LICENSE) 开源。第三方库、硬件 SDK、模型及数据集
遵循各自的许可证，MIT 许可证不重新授权这些第三方资源。
