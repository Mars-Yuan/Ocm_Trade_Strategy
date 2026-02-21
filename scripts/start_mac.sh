#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 启动脚本
# 功能: 启动后台服务
# ============================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="com.ocm.tradestrategy"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
PORT=8501

echo -e "${BLUE}正在启动 OCM Trade Strategy 服务...${NC}"

# 检查 plist 文件是否存在
if [ ! -f "$PLIST_PATH" ]; then
    echo -e "${YELLOW}错误: 服务配置文件不存在，请先运行安装脚本${NC}"
    exit 1
fi

# 停止现有服务（如果正在运行）
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# 启动服务
launchctl load "$PLIST_PATH"

# 等待启动
sleep 2

# 检查服务状态
if launchctl list | grep -q "$SERVICE_NAME"; then
    echo -e "${GREEN}✓ 服务已启动${NC}"
    echo -e "${BLUE}访问地址: http://localhost:$PORT${NC}"
    
    # 打开浏览器
    open "http://localhost:$PORT"
else
    echo -e "${YELLOW}⚠ 服务启动可能失败，请检查日志${NC}"
    echo "日志位置: ~/.ocm_trade_strategy/logs/"
fi
