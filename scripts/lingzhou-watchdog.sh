#!/bin/bash
# lingzhou watchdog — 每 60 秒检查一次，挂了就自动重启
PID_FILE="$HOME/.lingzhou/lingzhou.pid"
RESTART_LOG="$HOME/.lingzhou/logs/restart.log"
INTERVAL=60
MAX_RESTARTS_PER_HOUR=5
RESTART_COUNT=0
LAST_HOUR=$(date +%Y%m%d%H)

while true; do
    sleep $INTERVAL
    NOW_HOUR=$(date +%Y%m%d%H)
    if [ "$NOW_HOUR" != "$LAST_HOUR" ]; then
        RESTART_COUNT=0
        LAST_HOUR=$NOW_HOUR
    fi

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            continue  # 正常运行
        fi
    fi

    # 挂了 — 自动重启
    if [ $RESTART_COUNT -ge $MAX_RESTARTS_PER_HOUR ]; then
        echo "[$(date -Is)] ⛔ 每小时重启已达上限($MAX_RESTARTS_PER_HOUR)，跳过" >> "$RESTART_LOG"
        continue
    fi

    echo "[$(date -Is)] 🔄 检测到 lingzhou 已停止，自动重启..." >> "$RESTART_LOG"
    /usr/local/bin/lingzhou gateway restart 2>&1 >> "$RESTART_LOG"
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date -Is)]   restart_count=$RESTART_COUNT" >> "$RESTART_LOG"
done
