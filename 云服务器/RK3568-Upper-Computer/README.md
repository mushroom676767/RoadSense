# RK3568上位机与Windows视频转发Agent

本项目包含部署在云服务器上的Flask/Vue管理平台，以及运行在Windows电脑上的
Device Agent。RK3568把1080p JPEG和识别元数据发送给Agent；Agent使用RTX 4060的
NVENC编码为H.264，通过SRT向云端MediaMTX只上传一路视频，同时通过WSS同步状态、
日志和事件，并通过SSH控制板端三种生产模式。

## 运行链路

```text
RK3568 --JPEG+元数据--> Windows Agent 192.168.137.1:5000
Windows Agent --NVENC H.264/SRT--> 203.0.113.10:8890
Windows Agent --WSS状态与控制--> https://203.0.113.10/agent
浏览器 --HTTPS/WebRTC或LL-HLS--> 203.0.113.10
```

## 三种板端模式

| 模式 | 入口 | 上位机内容 |
| --- | --- | --- |
| 串口车牌 | `fpga_serial.py` | FPGA的1～4个XYXY候选ROI、RK YOLO26 Pose结果、车牌号和置信度 |
| AI车牌 | `fpga_ai.py` | RK YOLO26 Pose检测框、车牌号和置信度 |
| 行人闯红灯 | `people_redlight.py` | 普通目标、违规目标、计数和性能数据 |

串口模式不再接收四角点。当前协议为：

```text
0xAA
count: uint8，0～4
count × [x1,y1,x2,y2]: uint16 little-endian
```

坐标是1920×1080完整画面的XYXY绝对像素，右边界和下边界为排他边界。详细契约见
[docs/SERIAL_ROI_PROTOCOL.md](docs/SERIAL_ROI_PROTOCOL.md)。

## 启动Windows Agent

首次运行需要 Python 3.12、Node.js 20.19+ 或 22.12+，以及支持 NVENC 和 SRT 的 FFmpeg。
在此目录运行以下命令，然后编辑 `.env`，将全部 `replace-with-...` 值和示例地址替换为自己的配置：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-agent.txt
Copy-Item .env.example .env
npm ci --prefix frontend
npm run build --prefix frontend
```

完成配置后，生成 Git 忽略的 Agent 环境文件：

```powershell
.\.venv\Scripts\python.exe scripts\render-deployment.py --role agent
```

启动服务：

```powershell
cd "$env:USERPROFILE\Desktop\RK3568-Upper-Computer"
powershell -ExecutionPolicy Bypass -File .\scripts\start-agent.ps1
```

也可以双击根目录的`start-agent.cmd`。服务在当前窗口前台运行，按`Ctrl+C`停止；
默认不注册开机自启计划任务。健康检查地址：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Agent监听`0.0.0.0:5000`以适应直连网卡重连，但Windows防火墙仅允许
`192.168.137.0/24`访问该端口。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q agent.py app.py backend scripts
npm test --prefix frontend -- --run
npm run build --prefix frontend
```

`203.0.113.10` 是文档示例地址，使用前须替换为自己的云服务器地址。公网部署和回滚流程见
[docs/CLOUD_AGENT_DEPLOYMENT.md](docs/CLOUD_AGENT_DEPLOYMENT.md)。真实Token、SSH密码、
MediaMTX账号和SRT口令只能保存在`runtime/`或服务器系统配置中，不得提交Git。
