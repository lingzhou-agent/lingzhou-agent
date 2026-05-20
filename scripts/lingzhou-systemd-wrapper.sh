#!/bin/bash
# lingzhou systemd wrapper — 启动 daemon 后保持前台运行
# systemd 监控此脚本进程，daemon 崩溃时此脚本退出 → systemd 自动重启
# 唯一实例保证由 lingzhou Python 代码的 flock 机制处理

# 标记 wrapper 环境，避免 gateway start 重定向到 systemctl 造成无限循环
export LINGZHOU_WRAPPER=1

# 清理可能残留的 PID 文件
rm -f "$HOME/.lingzhou/lingzhou.pid"

# 停止旧进程（通过 PID 文件，如果存在）
/usr/local/bin/lingzhou stop 2>/dev/null
sleep 2

# 启动 daemon（Python 代码内部会获取 flock）
/usr/local/bin/lingzhou gateway start -d
START_EXIT=$?

if [ $START_EXIT -ne 0 ]; then
    echo "启动失败，退出码: $START_EXIT"
    exit 1
fi

# 等待 daemon 写入 PID 文件（最多 10 秒）
for i in $(seq 1 10); do
    sleep 1
    if [ -f "$HOME/.lingzhou/lingzhou.pid" ]; then
        break
    fi
done

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
