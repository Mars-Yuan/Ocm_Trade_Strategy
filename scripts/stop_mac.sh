#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 停止脚本
# 功能: 停止后台服务
# ============================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="com.ocm.tradestrategy"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"

echo -e "${BLUE}正在停止 OCM Trade Strategy 服务...${NC}"

# 停止 launchd 服务
if [ -f "$PLIST_PATH" ]; then
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# 杀死任何残留的 streamlit 进程
pkill -f "streamlit.*ocm_streamlit" 2>/dev/null || true

# 确认服务已停止
sleep 1
if ! launchctl list | grep -q "$SERVICE_NAME"; then
    echo -e "${GREEN}✓ 服务已停止${NC}"
else
    echo -e "${YELLOW}⚠ 服务可能仍在运行${NC}"
fi
