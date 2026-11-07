# 无域名公网部署：云端平台 + Windows Agent

以下地址均为示例。部署前替换公网 IP、私网 IP、用户目录和全部秘密值：

```text
https://203.0.113.10
```

`traffic.example.com`及其他HTTP内网穿透不再参与网页、控制、元数据或视频链路。旧的
`standalone + RTSP/SSH`仅保留为回退方案。

## 1. 最终链路

```text
RK3568 ──JPEG+元数据──> Windows Agent
                           ├─ NVENC H.264 ──SRT/UDP──> MediaMTX
                           ├─ WSS心跳/状态/日志 ─────> Flask-SocketIO
                           ├─ SQLite Outbox事件 ─────> 云端SQLite
                           └─ SSH启动/停止RK三种模式

浏览器 ──HTTPS/WSS──> Nginx ──> Flask/Vue
       ├─ WebRTC UDP / ICE-TCP / TURN-TLS
       └─ LL-HLS（WebRTC故障备用）
```

Windows和RK无需公网入站端口；Windows只主动连接云端。浏览器不再自动回退公网
JPEG。上行中断时保留最后一张无旧框画面，WebRTC快速恢复失败后自动使用LL-HLS。

## 2. 端口与云安全组

云安全组和UFW均需开放：

| 端口 | 协议 | 用途 |
|---|---|---|
| 80 | TCP | ACME IP证书验证、HTTPS跳转 |
| 443 | TCP | HTTPS、WSS、LL-HLS |
| 8890 | UDP | Windows加密SRT上行 |
| 49199 | UDP | WebRTC直接媒体 |
| 49200 | TCP | WebRTC ICE-TCP |
| 3478 | UDP/TCP | TURN |
| 5349 | TCP | TURN/TLS |
| 49160–49190 | UDP | TURN中继 |

不要向公网开放MediaMTX的8888、8889、9997和9998；模板仅绑定
`127.0.0.1`。

## 3. 准备秘密配置

真实值只写入仓库根目录的`.env`，不得提交Git。除原有登录、RK SSH和媒体账号外，
补充：

```text
PUBLIC_IP=203.0.113.10
PRIVATE_IP=10.0.0.10
ACME_EMAIL=<证书告警邮箱>
AGENT_ID=rk3568-primary
AGENT_TOKEN=<至少64位随机值>
MEDIA_MTX_ADMIN_USERNAME=<仅本机API管理账号>
MEDIA_MTX_ADMIN_PASSWORD=<随机强密码>
MEDIA_MTX_SRT_PASSPHRASE=<10至79字符随机值>
AGENT_LISTEN_HOST=192.168.137.1
AGENT_SERVER_HOST=0.0.0.0
AGENT_DATABASE_PATH=C:\projects\RK3568-Upper-Computer\db\agent.db
AGENT_EVENT_DIR=C:\projects\RK3568-Upper-Computer\db\agent-events
```

发布、读取、管理三个MediaMTX账号必须不同。渲染配置：

```powershell
.\.venv\Scripts\python.exe scripts\render-deployment.py --role all
```

输出只会写入Git忽略的`runtime/rendered/`，脚本不会打印秘密。检查文件存在即可，
不要把内容复制到日志或聊天。

## 4. Canary与云端安装

先构建前端并运行独立端口Canary：

```powershell
npm ci --prefix frontend
npm run build --prefix frontend
.\scripts\deploy-cloud.ps1 -Action Canary
```

Canary只在临时端口启动6秒，验证always-available H.264、SRT、WebRTC和LL-HLS，
不会重启现有服务。通过后上传当前Git提交：

```powershell
.\scripts\deploy-cloud.ps1 -Action Upload
```

首次部署先申请Let’s Encrypt测试IP证书：

```powershell
.\scripts\deploy-cloud.ps1 `
  -Action Activate `
  -CertificateMode staging `
  -ConfirmProduction
```

确认80端口挑战、Nginx、MediaMTX、Coturn和Flask服务均正常后，立即切换受公众信任的
生产IP证书：

```powershell
.\scripts\deploy-cloud.ps1 `
  -Action Activate `
  -CertificateMode production `
  -ConfirmProduction
```

正式激活前脚本会备份现有MediaMTX、Coturn、Nginx和systemd配置到：

```text
/var/backups/rk3568-manager/<UTC时间>
```

IP证书约160小时有效。`rk3568-ip-cert.timer`每12小时检查并续期，少于48小时仍未
成功续期时写入systemd告警日志。

## 5. 迁移用户、事件和截图

Windows一致性导出：

```powershell
.\.venv\Scripts\python.exe scripts\export-manager-data.py
```

脚本使用SQLite在线backup API，校验每张截图并生成SHA-256清单，同时把数据库中的
Windows截图路径转换为`/var/lib/rk3568-manager/events/`。本仓库不附带用户数据库、事件记录或截图；迁移时以自己的数据为准。

上传并在云端导入：

```powershell
scp -i runtime\mediamtx_ed25519 `
  -o UserKnownHostsFile=runtime\mediamtx_known_hosts `
  runtime\manager-data.zip `
  root@203.0.113.10:/var/tmp/manager-data.zip

ssh -i runtime\mediamtx_ed25519 `
  -o UserKnownHostsFile=runtime\mediamtx_known_hosts `
  root@203.0.113.10 `
  "systemctl stop rk3568-cloud && /opt/rk3568-manager/current/.venv/bin/python /opt/rk3568-manager/current/scripts/import-manager-data.py /var/tmp/manager-data.zip && chown -R rkmanager:rkmanager /var/lib/rk3568-manager && systemctl start rk3568-cloud"
```

导入脚本先验证全部哈希，再把云端原数据库和截图目录移动到带时间戳的备份目录。

## 6. 启动Windows Agent

Windows Agent需要带`libsrt`和`h264_nvenc`的FFmpeg。使用Scoop安装并验证：

```powershell
scoop install ffmpeg
ffmpeg -hide_banner -protocols | Select-String "srt"
ffmpeg -hide_banner -encoders | Select-String "h264_nvenc"
```

PyAV自身包含SRT协议时直接使用PyAV发布；官方PyAV Wheel不包含SRT时，Agent会自动
使用上述FFmpeg进行NVENC编码和SRT上传。两者都不可用时会在云端状态中显示明确错误。

先停止旧`app.py`和公网隧道，确认5000端口空闲，再运行：

```powershell
cd "$env:USERPROFILE\Desktop\RK3568-Upper-Computer"
.\scripts\start-agent.ps1
```

Agent向RK公布`192.168.137.1:5000`作为回传地址，主动连接云端`/agent` WSS，并向
`203.0.113.10:8890/UDP`发布3.2 Mbps SRT。默认SRT延迟为120 ms；网络RTT持续
超过80 ms或丢包明显时可将`MEDIA_MTX_SRT_LATENCY_MS`调回200。事件先写入本地SQLite Outbox；
云端确认后才出队，因此网络中断或Agent重启不会丢档案。

Agent实际绑定`0.0.0.0:5000`以适应RK直连网卡断开后重新出现；Windows防火墙只允许
`192.168.137.0/24`访问5000端口，不能把该端口映射到公网。首次部署时在管理员
PowerShell中运行：

```powershell
.\scripts\configure-agent-firewall.ps1
```

默认采用前台手动启动，不安装计划任务。启动后可验证：

```powershell
Invoke-WebRequest http://127.0.0.1:5000/health -UseBasicParsing
```

## 7. 验证与回退

服务检查：

```bash
systemctl status rk3568-cloud mediamtx coturn nginx
systemctl status rk3568-ip-cert.timer
curl -I https://203.0.113.10/
```

浏览器应优先显示`H.264 / WebRTC`。禁用UDP后应使用ICE-TCP、TURN/TLS；连续两次
WebRTC失败或10秒无RTP后显示`H.264 / LL-HLS`，WebRTC稳定20秒后自动切回。

若上线阶段失败：

1. 停止Windows Agent。
2. 恢复最近的`/var/backups/rk3568-manager/<时间>/`配置。
3. 将`/opt/rk3568-manager/current`指回上一个release。
4. 重启MediaMTX、Coturn和旧standalone上位机。
5. 在确认云端稳定之前不要停止当前旧链路。
