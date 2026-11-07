#!/bin/sh
set -eu

RULE_PATH=/etc/udev/rules.d/99-rk3568-hdmi-capture.rules
RULE='ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="1bcf", ATTR{idProduct}=="28c4", TEST=="power/control", ATTR{power/control}="on"'

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用sudo运行此脚本" >&2
    exit 1
fi

printf '%s\n' "$RULE" >"$RULE_PATH"
udevadm control --reload-rules

for device in /sys/bus/usb/devices/*; do
    [ -r "$device/idVendor" ] || continue
    [ -r "$device/idProduct" ] || continue
    [ "$(cat "$device/idVendor")" = "1bcf" ] || continue
    [ "$(cat "$device/idProduct")" = "28c4" ] || continue
    [ -w "$device/power/control" ] || continue
    printf 'on\n' >"$device/power/control"
    echo "已关闭采集卡USB自动休眠: $device"
done

echo "已安装规则: $RULE_PATH"
