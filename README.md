# 2026年集创赛紫光同创赛道国家二等奖

**RoadSense · 基于紫光 FPGA + RK3568 的视频流智能交通识别系统**

本项目将车牌识别、行人闯红灯检测和云端管理结合起来：RK3568 完成视频采集与 NPU 推理，FPGA 在协同模式下提供车牌候选区域，云端网页用于查看视频、管理事件和远程切换运行模式。

[快速开始](#快速开始) · [部署与联调](docs/DEPLOYMENT.md) · [技术文档](#技术文档) · [数据集](#数据集) · [问题反馈](https://github.com/mushroom676767/jichuang-rk3568-vision/issues)

## 项目亮点

- **三种运行模式**：RK3568 独立车牌识别、FPGA ROI 串口协同识别、行人闯红灯检测。
- **板端视频与推理分离**：使用 V4L2 / MPP 采集、RGA 预处理、RKNN 推理及 DRM 显示，异步处理并优先保留最新任务。
- **端云协同**：Windows Agent 将板端视频编码为 H.264，经 SRT 上传；浏览器通过 WebRTC 查看，云端支持 LL-HLS 备用播放。
- **可管理的识别结果**：云平台提供管理员/访客权限、模式控制、日志、事件归档、截图及 CSV 导出；Agent 持久化待发送事件，云端确认后移除。
- **配套资料**：三个比赛阶段的原始技术文档，以及合计 32.91 GB 的数据包、校验清单和下载还原工具。

默认采集/显示参数为 **1920×1080、MJPG、60 fps**。该参数不等同于模型推理或公网播放帧率，实际性能需在对应硬件和网络环境下测量。

## 目录

- [系统组成](#系统组成)
- [开源内容](#开源内容)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [技术文档](#技术文档)
- [数据集](#数据集)
- [项目结构](#项目结构)
- [测试与验证](#测试与验证)
- [参与贡献](#参与贡献)
- [问题反馈](#问题反馈)
- [许可证](#许可证)
- [致谢与参考](#致谢与参考)

## 系统组成

| 部分 | 职责 | 代码位置 |
| --- | --- | --- |
| RK3568 端 | 视频采集、检测/OCR、本地显示、识别结果发布 | [`RK3568/final/`](RK3568/final/) |
| FPGA 端 | 串口协同模式下提供 0～4 个候选 ROI | 本仓库提供[接收协议](RK3568/final/core/serial_io.py)，不包含 FPGA 硬件工程 |
| 云服务器端 | 网页、认证、远程控制、事件归档和媒体分发 | [`云服务器/RK3568-Upper-Computer/`](云服务器/RK3568-Upper-Computer/) |
| Windows Agent | 接收板端数据、转发视频、通过 SSH 执行云端控制命令 | 同目录下的 [`agent.py`](云服务器/RK3568-Upper-Computer/agent.py) |

Windows Agent 是云端方案的配套组件，运行在连接 RK3568 的 Windows 电脑上，与 Linux 云服务器分开部署。

运行时，RK3568 向 Agent 发布 JPEG 与识别元数据；Agent 经 SRT 将视频发送到 MediaMTX，经 WSS 同步状态和事件。浏览器通过云端平台观看和控制；控制命令经 Agent 的 SSH 连接落实到板端。

## 开源内容

| 内容 | 发布方式 |
| --- | --- |
| RK3568、云平台和 Agent 源码 | 本仓库，可通过 Git 克隆 |
| 初赛、分赛区决赛、全国总决赛技术文档 | [`docs/technical/`](docs/technical/README.md) 中的原始 PDF |
| CRPD、整帧 Pose、ROI Pose 数据包 | [Datasets v1 Release](https://github.com/mushroom676767/jichuang-rk3568-vision/releases/tag/datasets-v1) 附件 |
| FPGA 硬件工程、模型权重 | 不随本仓库提供 |

## 快速开始

### 1. 获取源码

```bash
git clone https://github.com/mushroom676767/jichuang-rk3568-vision.git
cd jichuang-rk3568-vision
```

如果只阅读方案，可以直接打开下方[技术文档](#技术文档)；如果只需要数据，可按[数据集](#数据集)一节下载，不必先部署整套系统。

### 2. 在 RK3568 上运行

需要 RK3568 Linux 板卡、匹配的 RKNN Runtime 与 MPP/RGA/DRM 等开发库，以及相应的模型。Python 侧使用 NumPy、OpenCV，串口模式另需 pyserial。

将 `RK3568/` 目录部署到板端 `/home/cat/RK3568/`，把[模型清单](RK3568/final/rknn_models/README.md)中的文件放入 `final/rknn_models/`。在板端执行：

```bash
cd /home/cat/RK3568/final
make -C core/native all
python3 fpga_ai.py
```

默认视频节点为 `/dev/video9`。设备路径、SDK 库、模型固定路径及构建参数见[详细部署说明](docs/DEPLOYMENT.md#rk3568-端)。完整运行需要实际设备和模型。

### 3. 准备 Windows Agent

在连接 RK3568 的 Windows 电脑上，进入 `云服务器/RK3568-Upper-Computer`。建议 Python 3.12、Node.js 20.19+ 或 22.12+；视频转发需要 NVIDIA GPU 和支持 NVENC/SRT 的 FFmpeg。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-agent.txt
Copy-Item .env.example .env
npm ci --prefix frontend
npm run build --prefix frontend
```

编辑 `.env`，填写自己的 RK3568 SSH、Agent 地址、Token 和云端媒体配置，然后生成配置并启动：

```powershell
.\.venv\Scripts\python.exe scripts\render-deployment.py --role agent
.\scripts\start-agent.ps1
Invoke-RestMethod http://127.0.0.1:5000/health
```

正常健康响应包含 `ok: true`、`role: agent`。`.env.example` 中的占位值和示例地址必须替换。

### 4. 部署云端

云端使用 Flask/Vue、Nginx、MediaMTX、Coturn 和 systemd。安装、证书、端口、配置生成、上线与回滚步骤见[云端部署文档](云服务器/RK3568-Upper-Computer/docs/CLOUD_AGENT_DEPLOYMENT.md)。

本项目还有 `standalone` 独立上位机模式。角色区别及完整联调顺序见[部署与联调](docs/DEPLOYMENT.md#云服务器端)。

## 使用示例

### 选择板端模式

在板端 `/home/cat/RK3568/final` 中，以下三种模式选择一种运行：

| 任务 | 命令 | 发布流 ID |
| --- | --- | --- |
| 独立车牌识别 | `python3 fpga_ai.py` | `plate_ai` |
| FPGA ROI 协同识别 | `python3 fpga_serial.py --port /dev/ttyUSB0 --baud 115205` | `plate_serial` |
| 行人闯红灯检测 | `python3 people_redlight.py` | `people_redlight` |

串口默认波特率是源码中的 **115205**；应与 FPGA 实际发送配置一致。协议使用 `0xAA` 帧头、0～4 个全图 XYXY ROI、小端 `uint16` 坐标，详细约定见[串口说明](RK3568/final/docs/SERIAL_ROI_POSE26.md)。

行人模式也支持本地视频文件：

```bash
python3 people_redlight.py /path/to/video.mp4
```

### 将板端结果发布给 Agent

设置与 Agent 一致的共享 Token 后，以独立车牌模式为例：

```bash
export PUBLISH_TOKEN='替换为与Agent一致的共享Token'
python3 fpga_ai.py \
  --publish http://192.168.137.1:5000 \
  --publish-token "$PUBLISH_TOKEN" \
  --publish-stream plate_ai \
  --publish-fps 30
```

`192.168.137.1` 是示例 Agent 地址，应替换为板端实际可达地址。不同模式的 `--publish-stream` 必须与上表一致。发布默认关闭，启用时 Token 和流 ID 均必填。

所有入口均支持 `--help`。模型、串口格式、发布协议和常见问题集中在[部署与联调](docs/DEPLOYMENT.md)。

## 技术文档

作品团队为 **ZLC**，队伍编号 **CICC1001862**。三个阶段的 PDF 保留原始内容：

| 阶段 | 文档 | 页数 |
| --- | --- | --- |
| 初赛 | [初赛技术文档](docs/technical/preliminary.pdf) | 53 |
| 分赛区决赛 | [分赛区决赛技术文档](docs/technical/regional-final.pdf) | 96 |
| 全国总决赛 | [全国总决赛技术文档](docs/technical/national-final.pdf) | 63 |

建议先读全国总决赛文档了解整体方案，再结合前两个阶段查看设计演进。历史方案与性能记录对应各自比赛阶段，当前命令和接口以源码为准。文件校验值见[文档索引](docs/technical/README.md)。

## 数据集

数据包总大小为 **32,906,449,741 字节（32.91 GB）**，通过 [Datasets v1 Release](https://github.com/mushroom676767/jichuang-rk3568-vision/releases/tag/datasets-v1) 发布。

| 数据包 | 大小 | 附件数 | 内容 |
| --- | --- | --- | --- |
| `CRPD.zip` | 19.42 GB | 11 | 单车牌、双层车牌及多车牌图片与标注 |
| `plate_pose.7z` | 13.13 GB | 7 | 整帧车牌 Pose 数据及配置 |
| `plate_pose_roi.7z` | 0.36 GB | 1 | ROI 车牌 Pose 数据及配置 |

在仓库根目录使用 Python 3.11+ 下载：

```bash
# 下载全部数据，校验分卷后自动还原原始压缩包
python datasets/download.py

# 只下载较小的 ROI 数据包
python datasets/download.py --dataset plate_pose_roi.7z
```

也可以仅从 Release 下载 `download.py` 和 `manifest.json`，放在同一目录后运行 `python download.py`。

工具支持断点续传，校验每个附件及完整压缩包的 SHA-256。默认保存到 `datasets/downloads/`，还原成功后默认删除分卷并保留完整压缩包；全部下载和还原建议预留至少 **50 GB** 空间，解压还需额外空间。手动还原、统计与校验值见[数据集说明](datasets/README.md)。

## 项目结构

```text
.
├── RK3568/final/
│   ├── fpga_ai.py              # 独立车牌识别入口
│   ├── fpga_serial.py          # FPGA ROI 串口协同入口
│   ├── people_redlight.py      # 行人闯红灯检测入口
│   ├── plate_det/             # 车牌定位、预处理与 OCR
│   ├── people_det/            # 行人与违规目标检测
│   ├── core/native/           # 采集、显示、RGA、RKNN 原生 C 库
│   ├── rknn_models/           # 模型放置目录（权重不提交）
│   ├── tests/                 # 板端测试
│   └── docs/                  # 板端架构、接口和性能说明
├── 云服务器/RK3568-Upper-Computer/
│   ├── app.py                 # 云平台 / 独立上位机入口
│   ├── agent.py               # Windows Agent 入口
│   ├── backend/               # 认证、控制、事件和媒体处理
│   ├── frontend/              # Vue 前端及测试
│   ├── deploy/                # 云服务与 Agent 配置模板
│   ├── scripts/               # 部署、启动和数据迁移脚本
│   └── .env.example           # 配置示例
├── docs/
│   ├── DEPLOYMENT.md          # 完整部署、协议与联调说明
│   └── technical/             # 三个比赛阶段的原始 PDF
├── datasets/                  # 下载工具、附件清单与工具测试
└── LICENSE
```

更多代码说明：[RK3568 README](RK3568/README.md) · [板端架构](RK3568/final/docs/CODE_ARCHITECTURE.md) · [上位机 README](云服务器/RK3568-Upper-Computer/README.md)。

## 测试与验证

板端逻辑测试，在 `RK3568/final` 执行：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

上位机后端与前端测试，在 `云服务器/RK3568-Upper-Computer` 执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test --prefix frontend
npm run build --prefix frontend
```

数据下载工具测试，在仓库根目录执行：

```bash
python -m unittest discover -s datasets/tests -v
```

单元测试和静态检查不替代硬件验收。完整测试需要实际 RK3568、FPGA、视频采集设备及模型；性能指标应注明设备、驱动、模型和测量方法。

## 参与贡献

欢迎提交文档修正、问题复现、兼容性改进和代码修复：

1. 先通过 [Issues](https://github.com/mushroom676767/jichuang-rk3568-vision/issues) 说明问题或修改目标。
2. Fork 仓库，创建自己的修改分支，保持一次提交围绕一个问题。
3. 运行与修改相关的测试，必要时补充能验证行为的测试。
4. 提交 Pull Request，说明改动原因、验证结果和实机测试环境；未完成的硬件验证请注明。

数据包使用 Release 分发，依赖环境、模型权重、编译产物和本地运行数据不纳入源码提交。

## 问题反馈

请在 [GitHub Issues](https://github.com/mushroom676767/jichuang-rk3568-vision/issues) 提交问题或建议。硬件相关问题请附上板卡与系统镜像、RKNN/驱动版本、实际启动命令、复现步骤、预期结果和实际结果。

先排查设备节点、模型路径与校验值、串口参数、Agent 地址/Token，以及媒体服务状态。更完整的[常见问题](docs/DEPLOYMENT.md#常见问题)可用于定位。

## 许可证

项目自有源码采用 [MIT License](LICENSE)，使用、修改和分发时请保留版权及许可声明。

第三方库、硬件 SDK、模型和数据集遵循各自许可证。数据包未附独立许可证，源码 MIT 许可证不重新授权数据集；模型权重和 FPGA 硬件工程不在本次发布范围内。

## 致谢与参考

本项目使用 Rockchip RKNN/MPP/RGA、Flask、Vue、MediaMTX 等软硬件生态组件。感谢提供相关工具、库与技术资料的开发者。

README 的内容组织参考了[《如何编写 README.md：打造出色的开源项目文档》](https://zhuanlan.zhihu.com/p/679934993)，结合本项目补充了上手步骤、使用示例、贡献方式与反馈入口。
