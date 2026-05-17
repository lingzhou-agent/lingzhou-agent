#!/bin/bash
# lingzhou healthcheck — 三重健康检查
# 1. 进程存活 (PID 文件轮询)
# 2. 关键模块导入验证
# 3. 内存压力检查 (阈值 80% 警告)

PID_FILE="$HOME/.lingzhou/lingzhou.pid"
MAX_MEM_PCT=80

# --- 1. 进程存活检测 ---
if [ ! -f "$PID_FILE" ]; then
    echo "[FAIL] PID 文件不存在: $PID_FILE"
    exit 1
fi
PID=$(cat "$PID_FILE" 2>/dev/null)
if [ -z "$PID" ]; then
    echo "[FAIL] PID 文件为空"
    exit 1
fi
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[FAIL] 进程 PID=$PID 已退出"
    exit 1
fi
echo "[PASS] 进程存活 (PID=$PID)"

# --- 2. 关键模块导入验证 ---
MODULES=("core.loop" "memory.task_store" "memory.semantic" "memory.working")
for mod in "${MODULES[@]}"; do
    if ! timeout 10 python3 -c "import $mod" 2>/dev/null; then
        echo "[FAIL] 模块导入失败: $mod"
        exit 1
    fi
done
echo "[PASS] 关键模块导入正常"

# --- 3. 内存压力检查 ---
MEM_INFO=$(free | awk '/Mem:/ {print $3,$2}')
USED=$(echo "$MEM_INFO" | awk '{print $1}')
TOTAL=$(echo "$MEM_INFO" | awk '{print $2}')
if [ "$TOTAL" -gt 0 ]; then
    PCT=$(( USED * 100 / TOTAL ))
    echo "[INFO] 内存使用: ${PCT}% (used ${USED} / total ${TOTAL})"
    if [ "$PCT" -ge "$MAX_MEM_PCT" ]; then
        echo "[WARN] 内存使用超过阈值 ${MAX_MEM_PCT}%"
        # 警告但不退出，让监控层决定
    fi
else
    echo "[FAIL] 无法获取内存信息"
    exit 1
fi

echo "[OK] 所有检查通过"
exit 0