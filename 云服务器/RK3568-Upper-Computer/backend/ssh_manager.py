"""Reliable, server-side-only SSH control for the RK3568 applications."""

import re
import shlex
import threading
import time

import paramiko

import config


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07|\r")
_REMOTE_PROCESS_CLEANUP = r"""
pids="$(ps -eo pid=,comm=,args= | awk '
    $2 ~ /^python3/ &&
    $0 ~ /(^|[ \/])(fpga_ai|fpga_serial|people_redlight)\.py([[:space:]]|$)/ {
        print $1
    }
')"
if [ -n "$pids" ]; then
    kill -INT $pids 2>/dev/null || true
    sleep 1
    alive=""
    for pid in $pids; do
        kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
    done
    if [ -n "$alive" ]; then
        kill -TERM $alive 2>/dev/null || true
        sleep 1
        for pid in $alive; do
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    printf 'cleaned:%s\n' "$pids"
fi
""".strip()


class SSHManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.client = None
        self.channel = None
        self.current_mode = None
        self.running = False
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._log_history = ""
        self._generation = 0

    def _emit_log(self, text):
        clean = _ANSI_RE.sub("", str(text))
        if clean:
            with self._log_lock:
                self._log_history = (
                    self._log_history + clean
                )[-config.SHARED_LOG_MAX_CHARS :]
            self.socketio.emit("log_output", {"text": clean})

    @staticmethod
    def _sanitize_failure(text):
        clean = _ANSI_RE.sub("", str(text or ""))
        for secret in (config.PUBLISH_TOKEN, config.SSH_PASSWORD):
            if secret:
                clean = clean.replace(secret, "********")
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        return " | ".join(lines[-6:])[-1000:]

    def get_log_snapshot(self):
        with self._log_lock:
            return self._log_history

    def _emit_status(self, mode, running, error=None):
        payload = {"mode": mode, "running": running}
        if mode in config.MODES:
            payload["name"] = config.MODES[mode]["name"]
        if error:
            payload["error"] = error
        self.socketio.emit("mode_status", payload)

    def _connect_locked(self):
        transport = self.client.get_transport() if self.client else None
        if transport is not None and transport.is_active():
            return
        if not config.SSH_PASSWORD:
            raise RuntimeError("服务器未配置SSH_PASSWORD")
        self._close_client_locked()
        self._emit_log(
            f"[SSH] 正在连接 {config.SSH_USER}@"
            f"{config.SSH_HOST}:{config.SSH_PORT} ...\n"
        )
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            config.SSH_HOST,
            port=config.SSH_PORT,
            username=config.SSH_USER,
            password=config.SSH_PASSWORD,
            timeout=config.SSH_CONNECT_TIMEOUT,
            auth_timeout=config.SSH_CONNECT_TIMEOUT,
            banner_timeout=config.SSH_CONNECT_TIMEOUT,
            look_for_keys=False,
            allow_agent=False,
        )
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(15)
        self.client = client
        self._emit_log("[SSH] 连接成功\n")

    def _close_client_locked(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    def _build_command(self, mode_id, source):
        mode = config.MODES[mode_id]
        argv = list(mode["argv"])
        source_name = "HDMI采集卡"
        if mode.get("accepts_source"):
            selected = source if source in config.SOURCES else mode.get(
                "default_source", "hdmi"
            )
            source_config = config.SOURCES[selected]
            argv.extend(source_config["argv"])
            source_name = source_config["name"]

        if mode.get("publish") and config.PUBLISH_URL:
            argv.extend(
                [
                    "--publish",
                    config.PUBLISH_URL,
                    "--publish-token",
                    config.PUBLISH_TOKEN,
                    "--publish-stream",
                    mode_id,
                    "--publish-fps",
                    str(config.PUBLISH_FPS),
                    "--publish-quality",
                    str(config.PUBLISH_QUALITY),
                ]
            )
        command = (
            f"cd {shlex.quote(config.RK_WORK_DIR)} && exec "
            + " ".join(shlex.quote(part) for part in argv)
        )

        display_argv = list(argv)
        if "--publish-token" in display_argv:
            token_index = display_argv.index("--publish-token") + 1
            if token_index < len(display_argv):
                display_argv[token_index] = "********"
        display = (
            f"cd {shlex.quote(config.RK_WORK_DIR)} && exec "
            + " ".join(shlex.quote(part) for part in display_argv)
        )
        return command, display, source_name

    def _cleanup_remote_processes_locked(self):
        transport = self.client.get_transport() if self.client else None
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH连接不可用，无法清理板端遗留进程")
        channel = transport.open_session()
        try:
            channel.exec_command(_REMOTE_PROCESS_CLEANUP)
            deadline = time.monotonic() + max(
                config.SSH_STOP_TIMEOUT + 3,
                5,
            )
            output = bytearray()
            error = bytearray()
            while not channel.exit_status_ready():
                while channel.recv_ready():
                    output.extend(channel.recv(4096))
                while channel.recv_stderr_ready():
                    error.extend(channel.recv_stderr(4096))
                if time.monotonic() >= deadline:
                    raise TimeoutError("清理板端遗留进程超时")
                time.sleep(0.05)
            while channel.recv_ready():
                output.extend(channel.recv(4096))
            while channel.recv_stderr_ready():
                error.extend(channel.recv_stderr(4096))
            exit_code = channel.recv_exit_status()
            if exit_code != 0:
                detail = error.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"清理板端遗留进程失败，返回码{exit_code}: {detail}"
                )
            cleaned = output.decode("utf-8", errors="replace").strip()
            if cleaned:
                self._emit_log(
                    f"[SSH] 已清理板端遗留识别进程: "
                    f"{cleaned.removeprefix('cleaned:')}\n"
                )
        finally:
            channel.close()

    def start_mode(self, mode_id, source="hdmi"):
        with self._lock:
            if mode_id not in config.MODES:
                return {"error": f"未知模式: {mode_id}"}
            if self.running:
                stopped = self._stop_locked()
                if "error" in stopped:
                    return stopped
            return self._start_locked(mode_id, source, restart_attempt=0)

    def _start_locked(self, mode_id, source, restart_attempt):
        try:
            self._connect_locked()
            self._cleanup_remote_processes_locked()
            command, display, source_name = self._build_command(mode_id, source)
            self._emit_log(f"[SSH] 执行: {display}\n")
            self._emit_log(f"[SSH] 输入源: {source_name}\n")
            channel = self.client.get_transport().open_session()
            channel.get_pty(term="dumb", width=200, height=50)
            channel.exec_command(command)
        except Exception as exc:
            self._emit_log(f"[SSH] 启动失败: {exc}\n")
            self._close_client_locked()
            return {"error": f"启动失败: {exc}"}

        startup_error = self._confirm_startup(channel)
        if startup_error is not None:
            channel.close()
            self._emit_log(f"[SSH] 启动确认失败: {startup_error}\n")
            return {"error": startup_error}

        self._generation += 1
        generation = self._generation
        self.channel = channel
        self.current_mode = mode_id
        self.running = True
        reader = threading.Thread(
            target=self._read_output,
            args=(
                channel,
                generation,
                mode_id,
                source,
                restart_attempt,
            ),
            daemon=True,
            name=f"ssh-log-{mode_id}",
        )
        reader.start()
        self._emit_status(mode_id, True)
        return {
            "success": True,
            "mode": mode_id,
            "name": config.MODES[mode_id]["name"],
        }

    def _confirm_startup(self, channel):
        """Catch deterministic CLI/import/model failures before acknowledging."""
        deadline = time.monotonic() + config.SSH_START_CONFIRM_SECONDS
        captured = []
        while time.monotonic() < deadline:
            self._drain_channel(channel, capture=captured)
            if channel.exit_status_ready():
                self._drain_channel(channel, capture=captured)
                exit_code = channel.recv_exit_status()
                detail = self._sanitize_failure("".join(captured))
                if not detail:
                    detail = "未提供错误输出"
                return (
                    f"板端程序在启动确认期内退出（返回码{exit_code}）: "
                    f"{detail}"
                )
            time.sleep(0.05)
        return None

    def stop_mode(self):
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self):
        channel = self.channel
        mode = self.current_mode
        if not self.running or channel is None:
            self._generation += 1
            self.channel = None
            self.current_mode = None
            self.running = False
            return {"success": True, "stopped": None}

        self._emit_log(f"[SSH] 正在停止 {mode} ...\n")
        try:
            channel.send("\x03")
            deadline = time.monotonic() + config.SSH_STOP_TIMEOUT
            while not channel.exit_status_ready() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not channel.exit_status_ready():
                channel.close()
                self._emit_log("[SSH] 停止超时，已关闭远程通道\n")
        except Exception as exc:
            try:
                channel.close()
            except Exception:
                pass
            self._emit_log(f"[SSH] 停止时关闭通道: {exc}\n")

        self._generation += 1
        self.channel = None
        self.current_mode = None
        self.running = False
        self._emit_log(f"[SSH] 模式 {mode} 已停止\n")
        self._emit_status(mode, False)
        return {"success": True, "stopped": mode}

    def get_status(self):
        with self._lock:
            transport = self.client.get_transport() if self.client else None
            return {
                "current_mode": self.current_mode,
                "running": self.running,
                "connected": bool(transport and transport.is_active()),
            }

    def close(self):
        with self._lock:
            self._stop_locked()
            self._close_client_locked()

    def _read_output(
        self,
        channel,
        generation,
        mode,
        source,
        restart_attempt,
    ):
        self._emit_log("[SSH] 开始监听板端输出\n")
        exit_code = None
        failure = None
        try:
            while not channel.closed:
                got_data = self._drain_channel(channel)
                if channel.exit_status_ready():
                    self._drain_channel(channel)
                    exit_code = channel.recv_exit_status()
                    break
                if not got_data:
                    time.sleep(0.05)
        except Exception as exc:
            failure = str(exc)
            self._emit_log(f"\n[SSH读取异常: {exc}]\n")
        finally:
            should_restart = False
            with self._lock:
                if generation != self._generation or channel is not self.channel:
                    return
                self.running = False
                self.current_mode = None
                self.channel = None
                should_restart = (
                    restart_attempt < config.SSH_RESTART_ATTEMPTS
                )
            if exit_code is not None:
                self._emit_log(f"\n[板端程序已退出，返回码: {exit_code}]\n")
            if should_restart:
                next_attempt = restart_attempt + 1
                reason = failure or (
                    f"板端程序异常退出（返回码{exit_code}）"
                    if exit_code is not None
                    else "板端程序连接中断"
                )
                self._emit_status(
                    mode,
                    False,
                    f"{reason}，准备自动重启 "
                    f"{next_attempt}/{config.SSH_RESTART_ATTEMPTS}",
                )
                threading.Thread(
                    target=self._restart_after_exit,
                    args=(
                        generation,
                        mode,
                        source,
                        next_attempt,
                    ),
                    daemon=True,
                    name=f"ssh-restart-{mode}",
                ).start()
            else:
                self._emit_status(mode, False, failure)

    def _restart_after_exit(
        self,
        failed_generation,
        mode,
        source,
        restart_attempt,
    ):
        if config.SSH_RESTART_ATTEMPTS <= 0:
            return
        time.sleep(config.SSH_RESTART_DELAY_SECONDS)
        with self._lock:
            if (
                failed_generation != self._generation
                or self.running
                or self.channel is not None
            ):
                return
            self._emit_log(
                f"[SSH] 自动重启 {mode} "
                f"({restart_attempt}/{config.SSH_RESTART_ATTEMPTS})\n"
            )
            result = self._start_locked(
                mode,
                source,
                restart_attempt=restart_attempt,
            )
        if "error" in result:
            if restart_attempt < config.SSH_RESTART_ATTEMPTS:
                next_attempt = restart_attempt + 1
                self._emit_status(
                    mode,
                    False,
                    f"{result['error']}，准备自动重启 "
                    f"{next_attempt}/{config.SSH_RESTART_ATTEMPTS}",
                )
                threading.Thread(
                    target=self._restart_after_exit,
                    args=(
                        failed_generation,
                        mode,
                        source,
                        next_attempt,
                    ),
                    daemon=True,
                    name=f"ssh-restart-{mode}",
                ).start()
            else:
                self._emit_status(mode, False, result["error"])

    def _drain_channel(self, channel, capture=None):
        got_data = False
        while channel.recv_ready():
            data = channel.recv(8192)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            if capture is not None:
                capture.append(text)
            self._emit_log(text)
            got_data = True
        while channel.recv_stderr_ready():
            data = channel.recv_stderr(8192)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            if capture is not None:
                capture.append(text)
            self._emit_log(text)
            got_data = True
        return got_data
