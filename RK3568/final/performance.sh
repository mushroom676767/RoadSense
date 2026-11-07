#!/usr/bin/env bash
#
# RK3568 定频脚本 —— 把 CPU / NPU / 内存(DMC) / 视频硬解(rkvdec) / GPU 全部锁到
# 最高性能档，消除动态调频带来的推理/解码/拷贝抖动。
#
# 默认板载 governor:  CPU=interactive, NPU=rknpu_ondemand —— 空载会降频,
# 首帧/突发负载要等升频, 表现为耗时毛刺。锁 performance 后直接跑满。
#
# 用法:
#   sudo ./performance.sh            # 锁最高性能 (推理/演示前跑一次)
#   sudo ./performance.sh --restore  # 恢复默认动态调频 (省电/日常)
#   sudo ./performance.sh --status   # 只看当前频率与 governor, 不改动
#
# 说明: 写 sysfs 需要 root。改动是运行时的, 重启后失效, 不持久化。
set -u

CPU_POLICY=/sys/devices/system/cpu/cpufreq/policy0
NPU=/sys/class/devfreq/fde40000.npu
DMC=/sys/class/devfreq/dmc
RKVDEC=/sys/class/devfreq/fdf80200.rkvdec
GPU=/sys/class/devfreq/fde60000.gpu

# 恢复时用的默认 governor (对照本板探测到的板载默认值)
CPU_DEFAULT_GOV=interactive
NPU_DEFAULT_GOV=rknpu_ondemand
DMC_DEFAULT_GOV=dmc_ondemand
RKVDEC_DEFAULT_GOV=vdec2_ondemand
GPU_DEFAULT_GOV=simple_ondemand

log()  { printf '%s\n' "$*"; }
head() { printf '\n== %s ==\n' "$*"; }

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        log "需要 root 权限, 请用: sudo $0 $*"
        exit 1
    fi
}

# 把 $2 写入文件 $1, 写失败只告警不中断 (某些节点内核可能不允许)
write_node() {
    local path="$1" value="$2"
    [ -e "$path" ] || { log "  跳过 (不存在): $path"; return; }
    if printf '%s' "$value" > "$path" 2>/dev/null; then
        log "  set $(basename "$(dirname "$path")")/$(basename "$path") = $value"
    else
        log "  写入失败 (忽略): $path <- $value"
    fi
}

# 用 performance governor 把 devfreq 节点钉到最高频; 若不支持 performance,
# 回退到 userspace + max_freq。
lock_devfreq_max() {
    local dev="$1" name="$2"
    [ -d "$dev" ] || { log "  跳过 (无 $name): $dev"; return; }
    local avail top
    if [ -r "$dev/available_governors" ] && \
       grep -qw performance "$dev/available_governors"; then
        write_node "$dev/governor" performance
    else
        write_node "$dev/governor" userspace
        if [ -r "$dev/available_frequencies" ]; then
            top=$(tr ' ' '\n' < "$dev/available_frequencies" | sort -n | tail -1)
            [ -n "$top" ] && write_node "$dev/userspace/set_freq" "$top"
        fi
    fi
}

restore_devfreq() {
    local dev="$1" name="$2" gov="$3"
    [ -d "$dev" ] || return
    if [ -r "$dev/available_governors" ] && grep -qw "$gov" "$dev/available_governors"; then
        write_node "$dev/governor" "$gov"
    else
        write_node "$dev/governor" simple_ondemand
    fi
}

show_status() {
    head "CPU (policy0)"
    log "  governor : $(cat $CPU_POLICY/scaling_governor 2>/dev/null)"
    log "  cur_freq : $(cat $CPU_POLICY/scaling_cur_freq 2>/dev/null || echo '需root读取')"
    log "  max avail: $(tr ' ' '\n' < $CPU_POLICY/scaling_available_frequencies 2>/dev/null | sort -n | tail -1)"
    for d in "$NPU:NPU" "$DMC:DMC" "$RKVDEC:RKVDEC" "$GPU:GPU"; do
        local path="${d%%:*}" nm="${d##*:}"
        [ -d "$path" ] || continue
        head "$nm ($(basename "$path"))"
        log "  governor : $(cat $path/governor 2>/dev/null)"
        log "  cur_freq : $(cat $path/cur_freq 2>/dev/null)"
        log "  max avail: $(tr ' ' '\n' < $path/available_frequencies 2>/dev/null | sort -n | tail -1)"
    done
}

apply_performance() {
    need_root "$@"
    head "CPU -> performance (max $(tr ' ' '\n' < $CPU_POLICY/scaling_available_frequencies | sort -n | tail -1))"
    if grep -qw performance "$CPU_POLICY/scaling_available_governors" 2>/dev/null; then
        write_node "$CPU_POLICY/scaling_governor" performance
    else
        write_node "$CPU_POLICY/scaling_governor" userspace
        local top
        top=$(tr ' ' '\n' < "$CPU_POLICY/scaling_available_frequencies" | sort -n | tail -1)
        write_node "$CPU_POLICY/scaling_setspeed" "$top"
    fi

    head "NPU -> performance (最高 900MHz, 默认 rknpu_ondemand 会降到 600MHz)"
    lock_devfreq_max "$NPU" NPU
    head "DMC 内存 -> performance (1080p60 帧拷贝/RGA 是内存带宽敏感的)"
    lock_devfreq_max "$DMC" DMC
    head "rkvdec 视频硬解 -> performance (MPP MJPEG 解码走它)"
    lock_devfreq_max "$RKVDEC" RKVDEC
    head "GPU -> performance"
    lock_devfreq_max "$GPU" GPU

    log ""
    log "定频完成。当前状态:"
    show_status
}

restore_default() {
    need_root "$@"
    head "恢复 CPU -> $CPU_DEFAULT_GOV"
    if grep -qw "$CPU_DEFAULT_GOV" "$CPU_POLICY/scaling_available_governors" 2>/dev/null; then
        write_node "$CPU_POLICY/scaling_governor" "$CPU_DEFAULT_GOV"
    else
        write_node "$CPU_POLICY/scaling_governor" ondemand
    fi
    head "恢复 NPU -> $NPU_DEFAULT_GOV"
    restore_devfreq "$NPU" NPU "$NPU_DEFAULT_GOV"
    head "恢复 DMC -> $DMC_DEFAULT_GOV"
    restore_devfreq "$DMC" DMC "$DMC_DEFAULT_GOV"
    head "恢复 rkvdec -> $RKVDEC_DEFAULT_GOV"
    restore_devfreq "$RKVDEC" RKVDEC "$RKVDEC_DEFAULT_GOV"
    head "恢复 GPU -> $GPU_DEFAULT_GOV"
    restore_devfreq "$GPU" GPU "$GPU_DEFAULT_GOV"
    log ""
    log "已恢复默认动态调频。"
}

case "${1:-}" in
    --restore|-r) restore_default "$@" ;;
    --status|-s)  show_status ;;
    ""|--apply|-a) apply_performance "$@" ;;
    *) log "用法: sudo $0 [--apply|--restore|--status]"; exit 2 ;;
esac
