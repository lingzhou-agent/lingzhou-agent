#!/bin/bash
# lingzhou systemd wrapper — 启动 daemon 后保持前台运行
# systemd 监控此脚本进程，daemon 崩溃时此脚本退出 → systemd 自动重启

# 清理可能残留的 PID 文件
rm -f "$HOME/.lingzhou/lingzhou.pid"

# 停止旧进程
/usr/local/bin/lingzhou stop 2>/dev/null
sleep 1

# 启动 daemon
/usr/local/bin/lingzhou gateway start -d
START_EXIT=$?

if [ $START_EXIT -ne 0 ]; then
    echo "启动失败，退出码: $START_EXIT"
    exit 1
fi

# 轮询 daemon 进程，挂了就退出让 systemd 重启
while true; do
    sleep 5
    PID=$(cat "$HOME/.lingzhou/lingzhou.pid" 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "$(date -Is) PID 文件丢失，lingzhou 已停止"
        exit 1
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "$(date -Is) PID=$PID 已退出"
        exit 1
    fi
done
