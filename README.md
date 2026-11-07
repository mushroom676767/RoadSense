# 集创赛 · RK3568 + FPGA 智能交通视觉系统

面向车牌识别与行人闯红灯检测的端云协同项目。系统由 **RK3568 端、FPGA 端和云服务器端**组成，提供本地视频显示、车牌定位与识别、违规目标检测、远程模式控制和事件归档。

云服务器端配套一个运行在 Windows 电脑上的 **Device Agent**：它连接 RK3568、转发视频并执行云端下发的控制命令。Agent 与云平台共用 `云服务器/RK3568-Upper-Computer/` 中的代码，部署在不同机器上。

> **源码范围**：本仓库包含 RK3568 程序、云平台和 Windows Agent，不提供 FPGA 硬件工程。FPGA 对接协议依据现有接收端实现说明；整体硬件方案可参考下方三个比赛阶段的技术文档。

## 目录

- [系统组成与运行链路](#系统组成与运行链路)
- [源码目录](#源码目录)
- [技术文档](#技术文档)
- [RK3568 端](#rk3568-端)
- [FPGA 端](#fpga-端)
- [云服务器端](#云服务器端)
- [端间通信与联调](#端间通信与联调)
- [测试与验证](#测试与验证)
- [常见问题](#常见问题)
- [开源范围与许可证](#开源范围与许可证)

## 系统组成与运行链路

| 部分 | 主要职责 | 对应代码 |
| --- | --- | --- |
| RK3568 端 | 视频采集、RGA 预处理、NPU 推理、车牌 OCR、本地显示、结果发布 | `RK3568/final/` |
| FPGA 端 | 在串口协同模式下提供车牌候选 ROI，交给 RK3568 进一步识别 | 接收协议见 `RK3568/final/core/serial_io.py`；硬件工程不提供 |
| 云服务器端 | 网页展示、身份认证、设备控制、事件与截图管理、媒体分发 | `云服务器/RK3568-Upper-Computer/` |
| Windows Agent（云端配套） | 接收板端 JPEG 与元数据，通过 SSH 控制板端，通过 SRT 与 WSS 连接云端 | 同目录下的 `agent.py`、`backend/device_agent.py` |

完整运行过程如下：

1. RK3568 从视频采集设备获取图像；在 FPGA 串口模式下，同时接收 FPGA 发来的候选区域。
2. RK3568 完成检测、识别和结果叠加，通过 DRM 输出本地画面，并按需向 Windows Agent 发布 JPEG 和识别元数据。
3. Windows Agent 将视频编码为 H.264，通过 SRT 上传 MediaMTX；通过 WSS 同步状态、日志和事件。
4. 浏览器访问云端 Flask/Vue 平台，通过 WebRTC 查看视频；云端链路支持 LL-HLS 备用播放。
5. 管理员在网页切换模式后，云端经 Agent 下发操作，Agent 再通过 SSH 启动或停止 RK3568 程序。

视频、识别元数据和控制命令分别传输。板端采集/显示的默认目标为 1080p60；检测、OCR 和公网播放的实际帧率取决于模型、设备与网络条件。

## 源码目录

```text
.
├── README.md
├── LICENSE
├── docs/technical/                # 初赛、分赛区决赛、全国总决赛 PDF
├── RK3568/
│   ├── README.md
│   └── final/
│       ├── fpga_ai.py              # RK3568 独立车牌识别入口
│       ├── fpga_serial.py          # FPGA ROI 串口协同识别入口
│       ├── people_redlight.py      # 行人闯红灯检测入口
│       ├── plate_det/              # 车牌运行时、定位、透视矫正与 OCR
│       ├── people_det/             # 行人/信号灯/违规目标检测
│       ├── core/                   # 采集、显示、串口、推流与并发基础设施
│       │   └── native/             # 原生 C 库与 Makefile
│       ├── rknn_models/            # 模型放置目录，权重未提交
│       ├── tests/                  # 板端逻辑与接口测试
│       ├── tools/                  # 性能测试工具
│       └── docs/                   # 架构、模型、串口与部署说明
└── 云服务器/
    └── RK3568-Upper-Computer/
        ├── app.py                 # 云平台/独立上位机入口
        ├── agent.py               # Windows Device Agent 入口
        ├── config.py              # 环境变量与运行模式配置
        ├── backend/               # 认证、接收、SSH、媒体、事件和 Agent 通道
        ├── frontend/              # Vue 3 + Element Plus + Vite 前端
        ├── deploy/                # Agent、Nginx、MediaMTX、Coturn、systemd 模板
        ├── scripts/               # 配置生成、部署、数据迁移和启动脚本
        ├── tests/                  # 上位机后端测试
        ├── requirements*.txt      # 不同部署角色的 Python 依赖
        └── .env.example            # 配置示例
```

本仓库不包含 FPGA 硬件工程。`fpga_ai.py` 和 `fpga_serial.py` 均为 **RK3568 上执行的 Python 程序**。

## 技术文档

三个阶段的原始 PDF 均已收录在 [`docs/technical/`](docs/technical/README.md)：

| 阶段 | 文档 | 页数 |
| --- | --- | --- |
| 初赛 | [初赛技术文档](docs/technical/preliminary.pdf) | 53 |
| 分赛区决赛 | [分赛区决赛技术文档](docs/technical/regional-final.pdf) | 96 |
| 全国总决赛 | [全国总决赛技术文档](docs/technical/national-final.pdf) | 63 |

文档包含硬件方案、算法设计、模型训练与量化、FPGA 模块、性能分析及仿真测试，展示不同比赛阶段的设计演进。历史方案、参数和性能记录以各文档对应阶段为准；当前程序接口以本仓库源码为准。

## RK3568 端

### 三种运行模式

| 模式 | 启动入口 | 功能 | 发布流 ID |
| --- | --- | --- | --- |
| 独立车牌识别 | `fpga_ai.py` | 全图 Pose26 检测与角点定位、单双层车牌预处理、OCR | `plate_ai` |
| FPGA 串口协同 | `fpga_serial.py` | 接收 0～4 个 ROI，扩展区域并组成四槽输入，执行 Pose26 与 OCR | `plate_serial` |
| 行人闯红灯检测 | `people_redlight.py` | 识别人、信号灯、违规目标等类别，并输出检测结果与统计 | `people_redlight` |

车牌检测与识别实现位于 [`plate_det`](RK3568/final/plate_det/)，行人模式位于 [`people_det`](RK3568/final/people_det/)。行人模型包含 `person`、`red_light`、`green_light`、`run_red_light`、`yellow_light`、`car`、`bicycle` 和 `motorcycle` 类别，违规结果依据模型输出的类别生成。

### 视频与推理实现

- **采集**：V4L2 获取 MJPEG，经 MPP 解码为 NV12 DMA-BUF 帧。
- **预处理**：RGA 执行缩放、格式转换及 ROI 组图。
- **推理**：默认通过项目原生 C 封装调用 RKNN Runtime。
- **车牌识别**：根据角点进行透视矫正，双层车牌拆分拼接后送入 OCR。
- **显示**：DRM Atomic 管理视频与叠加层；本地视频文件模式另有 DRM-RGA 路径。
- **并发**：采集、处理和显示异步运行，队列优先保留最新任务，避免旧帧积压。

串口 ROI 会绑定接收时可用的最新视频帧。现有串口报文不包含视频帧号或时间戳，因此该链路没有基于报文帧号的逐帧同步。

### 环境准备

需要 RK3568 Linux 板卡及其匹配的驱动、SDK 和开发库：

| 类型 | 依赖 |
| --- | --- |
| Python | Python 3、NumPy、OpenCV；串口模式另需 pyserial |
| 构建工具 | GCC、Make、pkg-config |
| 原生库 | Rockchip MPP、libv4l2、libdrm、librga、FreeType、RKNN Runtime |
| RKNN 文件 | `librknnrt.so` 与匹配的 `rknn_api.h` |
| 设备 | 默认视频节点 `/dev/video9`，串口节点 `/dev/ttyUSB0`，以及可用的 DRM 显示设备 |

板端 Python 依赖可按现有系统环境安装；OpenCV 和原生开发库建议使用与板卡镜像兼容的版本。

```bash
python3 -m pip install numpy opencv-python pyserial
```

源码默认使用 `/home/cat/RK3568/final` 作为板端工作目录。AI 模式与串口检测模型使用固定路径；更换目录时需同步检查模型路径常量和上位机的 `RK_WORK_DIR`。

### 准备模型与构建

将自行准备的模型放入 `RK3568/final/rknn_models/`：

| 模型文件 | 用途 | 输入宽 × 高 |
| --- | --- | --- |
| `plate_ai_pose26_512x288_int8.rknn` | 独立车牌检测与角点定位 | 512 × 288 |
| `plate_serial_roi_pose26_320x160_w8a8_fp16.rknn` | FPGA ROI 四槽检测 | 320 × 160 |
| `plate_ocr_168x48_int8.rknn` | 车牌字符识别 | 168 × 48 |
| `people_redlight_yolov5_384x640_int8.rknn` | 行人闯红灯检测 | 384 × 640 |

串口检测模型启动时会校验 SHA-256 和输入输出张量；不能仅通过重命名替换成其他模型。参考哈希见[模型说明](RK3568/final/rknn_models/README.md)。

在 RK3568 上构建原生库：

```bash
cd /home/cat/RK3568/final
make -C core/native all
# rknn_api.h 不在默认目录时，显式指定实际头文件目录：
# make -C core/native all RKNN_INCLUDE=/path/to/include
```

构建目标为 `librkhdmi_capture.so`、`librkdrm_atomic.so`、`librkdrm_display.so`、`librga_preprocess.so` 和 `librknn_runtime.so`。Makefile 默认 RKNN 头文件目录为 `/home/cat/.local/include`。

### 启动程序

以下三种模式选择一种运行：

```bash
cd /home/cat/RK3568/final

# RK3568 独立车牌识别
python3 fpga_ai.py

# FPGA 候选 ROI + RK3568 识别
python3 fpga_serial.py --port /dev/ttyUSB0 --baud 115205

# 行人闯红灯检测，默认从 HDMI 采集
python3 people_redlight.py
```

三个入口默认使用 `1920×1080 / MJPG / 60 fps` 采集参数。设备节点可通过 `--hdmi-device` 调整。行人模式支持视频文件，例如：

```bash
python3 people_redlight.py /path/to/video.mp4
```

完整参数使用对应入口的 `--help` 查看。板端基础说明见 [`RK3568/README.md`](RK3568/README.md)。

## FPGA 端

### 与 RK3568 的分工

串口协同模式下，FPGA 提供全画面的车牌候选矩形；RK3568 对候选区域做后续定位、单双层车牌预处理和 OCR，并输出车牌号、置信度及叠加结果。

接收实现见 [`core/serial_io.py`](RK3568/final/core/serial_io.py)，处理流程见 [`plate_det/serial_runtime.py`](RK3568/final/plate_det/serial_runtime.py)。当前源码无法确认 FPGA 内部如何产生候选区域，也无法提供具体的硬件综合和烧录命令。

### 串口协议

当前帧格式为：

```text
0xAA | count | ROI[0] | ... | ROI[count-1]
ROI = x1 | y1 | x2 | y2
```

| 字段 | 长度 | 说明 |
| --- | --- | --- |
| 帧头 | 1 字节 | 固定为 `0xAA` |
| `count` | 1 字节 | 无符号整数，取值 0～4 |
| 每个 ROI | 8 字节 | `x1, y1, x2, y2`，每个值为小端 `uint16` |

- 坐标基于 **1920×1080 完整画面**。
- 必须满足 `0 ≤ x1 < x2 ≤ 1920`、`0 ≤ y1 < y2 ≤ 1080`。
- `x2` 和 `y2` 为排他边界。
- 单帧总长度为 `2 + 8 × count` 字节，最多 34 字节。
- `AA 00` 表示没有 ROI，RK3568 会清空车牌结果。
- 非法数量会触发重新搜索帧头；非法坐标会拒绝整包并清空旧结果。
- 当前协议没有校验和、帧尾、时间戳或帧号字段。

例如，一个 ROI 为 `(100, 200, 401, 301)` 时，发送的十六进制字节为：

```text
AA 01 64 00 C8 00 91 01 2D 01
```

**代码中的默认波特率是 `115205`**，可通过 `--baud` 修改，联调时必须与 FPGA 实际发送配置一致。默认串口节点为 `/dev/ttyUSB0`。

协议与模型槽位的详细约定见 [FPGA ROI 串口说明](RK3568/final/docs/SERIAL_ROI_POSE26.md)。

## 云服务器端

### 平台功能与技术栈

- **前端**：Vue 3、Element Plus、Vite、Socket.IO Client、hls.js。
- **后端**：Flask、Flask-SocketIO、Flask-JWT-Extended、SQLite。
- **媒体服务**：MediaMTX 接收 SRT 并提供 WebRTC / LL-HLS；Nginx 提供 HTTPS 与反向代理，Coturn 用于受限网络下的中继。
- **权限**：管理员可以控制设备；访客可以查看实时数据与事件。
- **事件归档**：保存识别事件、元数据和截图，支持查询与 CSV 导出。
- **可靠传输**：Agent 将待发送事件持久化至本地 SQLite 队列，云端确认后再移除。

### 部署角色

`config.py` 使用 `DEPLOYMENT_ROLE` 区分角色：

| 角色 | 入口 | 部署位置与作用 |
| --- | --- | --- |
| `cloud` | `app.py` | Linux 云服务器；提供网页/API，并通过 Agent 控制设备 |
| `agent` | `agent.py` | 连接 RK3568 的 Windows 电脑；接收板端数据、编码视频并执行 SSH 控制 |
| `standalone` | `app.py` | 独立上位机模式；直接连接 RK3568，适合本地联调或原有部署方式 |

完整公网方案需要云端平台、媒体服务和 Windows Agent 配合。仓库内的 Docker Compose 提供管理应用构建配置，完整媒体与公网服务的部署参照 `deploy/` 模板及部署文档。

### 安装依赖和构建前端

以下命令在 Windows PowerShell 中、`云服务器/RK3568-Upper-Computer` 目录执行。建议使用 Python 3.12，以及 Node.js 20.19+ 或 22.12+。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-agent.txt
Copy-Item .env.example .env
npm ci --prefix frontend
npm run build --prefix frontend
```

编辑 `.env`，将占位密码、Token 和示例地址替换为自己的配置。主要配置分组如下：

| 配置 | 用途 |
| --- | --- |
| `SSH_HOST`、`SSH_PORT`、`SSH_USER`、`SSH_PASSWORD`、`RK_WORK_DIR` | Agent 连接并启动 RK3568 程序 |
| `PUBLISH_URL`、`PUBLISH_TOKEN` | RK3568 向 Agent 回传数据的地址和共享凭据 |
| `PUBLIC_IP`、`PRIVATE_IP`、`ACME_EMAIL` | 云端地址与证书配置 |
| `AGENT_ID`、`AGENT_TOKEN` | Agent 身份及云端认证 |
| `JWT_SECRET_KEY`、`ADMIN_*`、`VISITOR_*` | 网页认证和初始账号 |
| `MEDIA_MTX_PUBLISH_*`、`MEDIA_MTX_READ_*`、`MEDIA_MTX_ADMIN_*` | 独立的媒体发布、读取和管理账号 |
| `MEDIA_MTX_SRT_PASSPHRASE` | SRT 上行口令 |
| `WEBRTC_TURN_USERNAME`、`WEBRTC_TURN_CREDENTIAL` | TURN 用户和凭据 |

`203.0.113.10`、`10.0.0.10` 和 `example.com` 均为示例值。`PUBLISH_URL` 必须是 **RK3568 实际能够访问的 Agent 地址**。

### Windows Agent

Windows 电脑需要 NVIDIA GPU、可用驱动，以及支持 `h264_nvenc` 和 SRT 的 FFmpeg。先检查编码器和协议：

```powershell
ffmpeg -hide_banner -encoders | Select-String "h264_nvenc"
ffmpeg -hide_banner -protocols | Select-String "srt"
```

填写 `.env` 后生成 Agent 环境配置并启动：

```powershell
.\.venv\Scripts\python.exe scripts\render-deployment.py --role agent
.\scripts\start-agent.ps1
Invoke-RestMethod http://127.0.0.1:5000/health
```

生成的配置保存在 `runtime/rendered/agent/agent.env`。启动脚本会加载配置并检查 FFmpeg。正常情况下，健康接口返回 `ok: true`、`role: agent`。

### Linux 云服务器

云端使用 `requirements-cloud.txt`，生产入口由 `deploy/cloud/rk3568-cloud.service.template` 定义，使用 Gunicorn 启动 Flask 应用。配套服务包括 Nginx、MediaMTX、Coturn 和证书续期任务。

配置完成后可在 Windows 管理端生成两侧部署文件：

```powershell
.\.venv\Scripts\python.exe scripts\render-deployment.py --role all
```

输出写入 `runtime/rendered/`。云端安装、Canary 验证、上传、激活及回滚步骤见 [云端部署文档](云服务器/RK3568-Upper-Computer/docs/CLOUD_AGENT_DEPLOYMENT.md)。部署脚本的 `-CloudHost` 默认值也是示例 IP，实际调用时应传入自己的云服务器地址。

## 端间通信与联调

### RK3568 向 Agent 发布

三个板端入口均支持 `--publish`、`--publish-token`、`--publish-stream`、`--publish-fps` 和 `--publish-quality`。默认关闭发布。以下以独立车牌模式为例，在板端设置与 Agent 一致的 Token 后运行：

```bash
cd /home/cat/RK3568/final
export PUBLISH_TOKEN='替换为与Agent一致的共享Token'
python3 fpga_ai.py \
  --publish http://192.168.137.1:5000 \
  --publish-token "$PUBLISH_TOKEN" \
  --publish-stream plate_ai \
  --publish-fps 30
```

`192.168.137.1` 是直连网络的示例地址，需按实际网络调整。其余两个入口的流 ID 分别使用 `plate_serial` 和 `people_redlight`，必须与模式一致。

发布器向 `POST /api/ingest` 发送二进制请求，认证头为 `X-Publish-Token`。请求体依次包含 **4 字节大端 JSON 长度、UTF-8 JSON 元数据、JPEG 数据**。元数据协议版本为 `1`，携带流 ID、模式、帧信息、识别结果、叠加对象和性能指标等。

发布采用后台线程与最新帧策略；网络错误会触发重连退避，不直接停止本地采集与推理。更多说明见 [上位机发布链路](RK3568/final/docs/RK3568_MANAGER_PUBLISH.md)。

### 建议联调顺序

1. 在 RK3568 上单独确认视频输入、原生库构建和模型加载。
2. 运行 `fpga_ai.py` 或 `people_redlight.py`，确认本地画面与结果。
3. 接入 FPGA，确认串口设备、波特率、0～4 个 ROI 和坐标范围，再运行 `fpga_serial.py`。
4. 启动 Windows Agent，确认 `/health` 和板端 `/api/ingest` 回传。
5. 启动云端服务，确认 Agent 在线、视频可播放、模式控制可返回执行状态。
6. 验证事件与截图归档，并检查网络中断后的重连和待发送事件补传。

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

板端测试依赖 Python 图像处理等组件；完整推理、HDMI、串口、显示与性能验收仍需要实际 RK3568、FPGA、采集设备和对应模型。视频显示帧率、NPU 推理耗时和公网播放帧率应分别测量。

进一步阅读：

- [板端代码架构](RK3568/final/docs/CODE_ARCHITECTURE.md)
- [FPGA ROI 串口协议与模型](RK3568/final/docs/SERIAL_ROI_POSE26.md)
- [Native 多应用运行说明](RK3568/final/docs/RK3568_NATIVE_MULTI_APP.md)
- [上位机 README](云服务器/RK3568-Upper-Computer/README.md)
- [云端与 Agent 部署](云服务器/RK3568-Upper-Computer/docs/CLOUD_AGENT_DEPLOYMENT.md)

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 视频设备打不开 | 实际 `/dev/video*` 节点、设备权限、MJPG 格式和 1920×1080 采集能力 |
| 原生库无法加载 | 是否在 RK3568 上构建，RKNN/MPP/RGA/DRM 库是否匹配，动态库搜索路径是否正确 |
| 模型不存在或校验失败 | 固定模型目录、文件名、SHA-256 和张量规格 |
| 串口有数据但识别不到 ROI | 波特率、`0xAA` 帧头、数量、小端坐标、排他边界和全图尺寸 |
| 本地正常但 Agent 无数据 | `PUBLISH_URL` 的可达性、Token 和流 ID 是否一致 |
| 网页有状态但没有视频 | FFmpeg 的 NVENC/SRT 支持、MediaMTX、WebRTC/TURN 和云安全组 |
| 网页不能切换模式 | 登录账号是否为管理员、Agent 是否在线、RK3568 SSH 配置及设备权限 |

## 开源范围与许可证

本仓库发布现有 RK3568、云平台与 Windows Agent 源码，以及已确认的 FPGA 对接协议。FPGA 硬件工程不提供。

以下资源不纳入 Git，需要自行准备或在运行时生成：

- `CRPD.zip`、`plate_pose.7z`、`plate_pose_roi.7z` 等图片和标注数据集。
- RKNN 模型权重及 `.so` 等编译产物。
- Python/Node.js 依赖环境、构建输出和缓存。
- `.env`、运行时秘密配置、密钥、数据库、事件截图及日志。

项目自有源码按 [MIT License](LICENSE) 开源。第三方库、硬件 SDK、模型和数据集遵循各自许可证。
