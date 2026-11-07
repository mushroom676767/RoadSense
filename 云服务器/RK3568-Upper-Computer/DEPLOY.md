# RK3568 管理平台部署与运维

> 公网生产部署已迁移为“云端平台 + Windows Agent + SRT/WebRTC/LL-HLS”，
> 不再依赖`traffic.example.com`或其他内网穿透。新部署、证书、事件迁移和回退步骤见
> [docs/CLOUD_AGENT_DEPLOYMENT.md](docs/CLOUD_AGENT_DEPLOYMENT.md)。
> 本文后续standalone与RTSP/SSH内容仅用于兼容回退。

## 1. 数据流

```text
HDMI MJPEG 1080p60
        │
        ├─ MPP → NV12 DMA-BUF → DRM Atomic → 本地HDMI约60Hz
        ├─ RGA → RKNN → 检测/识别元数据
        └─ 原始MJPEG tap（30FPS latest-wins）
                    │
                    ▼
             HTTP POST /api/ingest
                    │
          Flask + Socket.IO + SQLite
             │
             ├─ 模式/日志/指标/结果/事件 → HTTP公网隧道
             │
             └─ RTX 4060 NVENC（只编码一次）
                         │ 单路RTSP/H.264约7Mbps
                         │ SSH本地端口转发
                         ▼
                 云端MediaMTX/SFU
                         │
                         └─ WHEP/WebRTC → 12个浏览器Canvas叠框
```

板端网络失败只影响网页视频，不能停止HDMI显示或NPU推理。UDP和本地视频没有原始MJPEG时，Publisher才在后台使用OpenCV编码。视频重处理集中在Windows和云端：Windows始终只有一个NVENC编码器和一条云上行，云端负责逐用户分发；模式状态、日志、指标、结果和事件均来自同一上位机实例。WebRTC协商失败时网页会明确回退到原JPEG/Socket.IO兼容路径。

## 2. 环境变量

复制 `.env.example` 为 `.env`，至少修改：

```text
JWT_SECRET_KEY
ADMIN_USERNAME
ADMIN_PASSWORD
SSH_HOST / SSH_PORT / SSH_USER / SSH_PASSWORD
PUBLISH_URL
PUBLISH_TOKEN
VIDEO_TRANSPORT / WEBRTC_PROVIDER
MEDIA_MTX_HOST / MEDIA_MTX_SSH_KEY / MEDIA_MTX_KNOWN_HOSTS
MEDIA_MTX_PUBLISH_USERNAME / MEDIA_MTX_PUBLISH_PASSWORD
MEDIA_MTX_READ_USERNAME / MEDIA_MTX_READ_PASSWORD
```

规则：

- SSH密码只保存在服务器 `.env`，浏览器API不接受SSH凭据。
- `PUBLISH_TOKEN`由服务器启动板端程序时传入，日志始终显示为 `********`。
- `PUBLISH_URL`必须是RK3568能够访问的地址，不能写成板端视角的`127.0.0.1`。
- 默认关闭公开注册；管理员仅在数据库中不存在时初始化一次。
- 事件默认保留30天或10000条，先达到的限制生效。
- `VIDEO_TRANSPORT=webrtc`及`WEBRTC_PROVIDER=mediamtx`时，电脑只向云端上传一路H.264；HTTP公网隧道只承载页面、信令和共享状态。
- MediaMTX发布和读取必须使用不同的最小权限账号；凭据、SSH私钥和known-hosts文件不得提交Git。
- 公网防火墙只开放MediaMTX的WebRTC UDP媒体端口；RTSP和WHEP监听在云服务器回环地址并通过SSH转发，不直接暴露公网。

## 3. Windows运行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
.\.venv\Scripts\python app.py
```

`app.py`带有单实例保护。重复启动时会直接提示已有运行实例并退出，避免两个进程共享5000端口后把RK视频帧与WebRTC信令分配到不同内存实例。

Windows Internet Connection Sharing通常给本机分配 `192.168.137.1`，对应：

```text
PUBLISH_URL=http://192.168.137.1:5000
SSH_HOST=192.168.137.216
```

如板端无法回传，检查Windows防火墙是否允许TCP 5000入站：

```powershell
Test-NetConnection 192.168.137.1 -Port 5000
```

板端检查：

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://192.168.137.1:5000/api/config', timeout=3).read())"
```

## 4. Linux和Docker

直接运行：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
python app.py
```

Docker Compose会在多阶段镜像中完成前端构建：

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
docker compose logs -f
```

SQLite和事件截图持久化到宿主机 `./db`。

## 5. 模式与板端命令

网页仅保留三个模式：

```text
plate_serial    → fpga_serial.py
plate_ai        → fpga_ai.py
people_redlight → people_redlight.py --source hdmi
```

服务器自动附加：

```text
--result-log-interval 60
--publish <PUBLISH_URL>
--publish-token <PUBLISH_TOKEN>
--publish-stream <mode_id>
--publish-fps 30
--publish-quality 50
```

切换模式时，服务器先向旧程序发送Ctrl+C并等待退出，随后在同一SSH连接启动新程序。失败时网页保持明确的停止状态，不静默启动其他后端。

板端重新构建Native采集库：

```bash
cd /home/cat/RK3568
make -C final/core/native librkhdmi_capture.so
```

## 6. HTTPS和内网板卡

### Windows公网固定地址（RK仍走直连网线）

RK3568继续使用：

```text
PUBLISH_URL=http://192.168.137.1:5000
```

公网浏览器通过Cloudflare Named Tunnel访问电脑，RK3568到电脑的推流仍走
`192.168.137.0/24`直连以太网，不绕行公网，也不需要路由器端口转发。

从Cloudflare官方下载Windows amd64版本：

```text
https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
```

将文件安装到当前用户目录：

```powershell
$target = Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe"
New-Item -ItemType Directory (Split-Path $target) -Force | Out-Null
Copy-Item "$env:USERPROFILE\Downloads\cloudflared-windows-amd64.exe" $target
& $target --version
```

在Cloudflare Zero Trust控制台创建一个Remotely-managed Tunnel，为其添加固定的
Public Hostname，并将Service设置为：

```text
http://127.0.0.1:5000
```

把控制台给出的Token和固定网址写入本地`.env`，Token不得提交Git：

```text
CLOUDFLARE_TUNNEL_TOKEN=<Cloudflare提供的Tunnel Token>
CLOUDFLARE_PUBLIC_URL=https://traffic.example.com
```

先启动管理平台，再启动隧道：

```powershell
python app.py
.\scripts\start-public-tunnel.ps1
```

脚本确认Cloudflare四条连接中的至少一条注册成功后，输出固定公网网址。停止公网访问：

```powershell
.\scripts\stop-public-tunnel.ps1
```

公开部署必须使用强`ADMIN_PASSWORD`、`JWT_SECRET_KEY`和`PUBLISH_TOKEN`，保持`ALLOW_REGISTER=false`及`CORS_ORIGINS=same-origin`。

Cloudflare只代理浏览器的HTTPS和WebSocket连接。不要把板端`.env`或上位机
`PUBLISH_URL`改成公网域名，必须继续保留：

```text
PUBLISH_URL=http://192.168.137.1:5000
```

Newszero、Cloudflare Tunnel或其他HTTP隧道均只负责页面、JWT、Socket.IO共享状态和WHEP信令。H.264媒体由云端MediaMTX直接通过UDP发送给浏览器，因此HTTP隧道不承担1080p媒体流量，Windows也不再为每名观看者重复上传。

默认配置：

```text
VIDEO_TRANSPORT=webrtc
WEBRTC_PROVIDER=mediamtx
WEBRTC_VIDEO_FPS=30
WEBRTC_VIDEO_WIDTH=1920
WEBRTC_VIDEO_HEIGHT=1080
WEBRTC_VIDEO_BITRATE=4000000
WEBRTC_VIDEO_MAX_BITRATE=5000000
WEBRTC_OFFER_TIMEOUT_SECONDS=6
WEBRTC_SOURCE_GRACE_SECONDS=12
WEBRTC_SESSION_TIMEOUT_SECONDS=60
WEBRTC_VIDEO_ENCODER=h264_nvenc
WEBRTC_MAX_VIEWERS=12
WEBRTC_PEER_QUEUE_FRAMES=60
MEDIA_MTX_HOST=<云服务器IP>
MEDIA_MTX_SSH_USER=root
MEDIA_MTX_SSH_KEY=runtime/mediamtx_ed25519
MEDIA_MTX_KNOWN_HOSTS=runtime/mediamtx_known_hosts
MEDIA_MTX_LOCAL_RTSP_PORT=18554
MEDIA_MTX_LOCAL_WHEP_PORT=18889
MEDIA_MTX_SSH_KEEPALIVE_SECONDS=5
MEDIA_MTX_SSH_KEEPALIVE_COUNT=3
MEDIA_MTX_PUBLISH_USERNAME=<仅发布账号>
MEDIA_MTX_PUBLISH_PASSWORD=<发布密码>
MEDIA_MTX_READ_USERNAME=<仅读取账号>
MEDIA_MTX_READ_PASSWORD=<读取密码>
```

### 云端MediaMTX

云服务器安装MediaMTX后，RTSP只监听`127.0.0.1:8554`，WHEP HTTP只监听`127.0.0.1:8889`，WebRTC媒体监听一个固定公网UDP端口。防火墙只需允许SSH和该UDP媒体端口；不要向公网开放8554或8889。MediaMTX必须配置公网IP为WebRTC附加主机，并为发布、读取创建不同权限的账号。

Windows生成专用密钥和known-hosts文件：

```powershell
New-Item -ItemType Directory runtime -Force | Out-Null
ssh-keygen -t ed25519 -f runtime\mediamtx_ed25519
ssh-keyscan <云服务器IP> | Out-File runtime\mediamtx_known_hosts -Encoding ascii
```

把公钥加入云服务器对应用户的`authorized_keys`。上位机在第一个观看者协商时自动创建两个本地转发：

```text
127.0.0.1:18554 → 云端127.0.0.1:8554（RTSP发布）
127.0.0.1:18889 → 云端127.0.0.1:8889（WHEP信令）
```

默认使用RTX 4060的NVENC以4 Mbps统一编码一次并发布到MediaMTX。Windows上行与观看人数无关，通常维持约4至5 Mbps；12个观看者所需的多路媒体出口由云服务器承担。建议Windows稳定上行至少10 Mbps，云服务器出口至少80 Mbps，并保留20%至30%余量。

`WEBRTC_MAX_VIEWERS=12`是当前云服务器带宽和质量目标下的保护上限。每个浏览器仍独立建立WHEP会话，但共用同一编码源。会话关闭时前端调用DELETE释放云端资源；浏览器还会每10秒发送心跳，异常断网页面遗留的会话在超时后自动回收。服务停止时也会统一关闭所有会话、发布线程和SSH转发。

如果页面提示“已回退 JPEG 视频”，先检查云服务器WebRTC UDP端口、MediaMTX公网候选地址、SSH密钥和`/api/webrtc/config`中的`provider/available/encoder_sources`。JPEG回退只用于故障兼容，会重新经过HTTP隧道，不适合作为12人生产路径。

电脑休眠或关机时隧道会离线。长期运行时可使用Windows任务计划程序在登录时执行
`scripts/start-public-tunnel.ps1`，并禁止电脑自动睡眠。

### 其他固定域名方案

公网服务器无法直接SSH进入内网板卡时，可由板端建立反向隧道：

```bash
ssh -fNR 2222:localhost:22 user@server
```

服务器配置：

```text
SSH_HOST=127.0.0.1
SSH_PORT=2222
```

## 7. API和事件

- 板端发布：`POST /api/ingest`，使用 `X-Publish-Token`。
- 登录：`POST /api/login`。
- 模式：`GET /api/modes`、`POST /api/mode/start`、`POST /api/mode/stop`。
- 流状态：`GET /api/streams`。
- 事件：`GET /api/events`、详情、鉴权截图和 `events/export.csv`。

除登录、公开配置和板端Ingest外，所有接口均要求JWT。

车牌连续稳定出现时只保存一次；目标消失后重新出现才产生新事件。闯红灯只在违规数量从0变为非0时保存，避免逐帧重复截图。

## 8. 故障排查

- 网页无画面：检查 `/api/streams`、板端“上位机推流已启动”日志和5000端口。
- 页面显示“H.264 / WebRTC”但网页帧率始终为0：先检查是否曾重复启动上位机；新版会阻止第二个`app.py`实例。结束旧实例后只启动一次，再重新启动板端模式。
- 页面显示“WebRTC连接中”后回退JPEG：检查浏览器`chrome://webrtc-internals`中的ICE候选；无`srflx/relay`候选时检查STUN DNS、防火墙或配置TURN。
- WebRTC已连接但卡顿：确认电脑H.264编码CPU占用和上行带宽；TURN部署时同时检查TURN服务器出口。
- HDMI正常但网页掉帧：降低 `PUBLISH_FPS`，浏览器和服务器均采用latest-wins，不会反压板端。
- SSH启动失败：确认 `.env` 的板端目录和凭据；页面不会提交或覆盖这些值。
- Native库不支持MJPEG tap：重新执行Makefile构建，不要提交生成的`.so`。
- 事件截图增加过快：确认板端 `stream_id/mode`匹配，检查识别稳定性和事件冷却配置。
