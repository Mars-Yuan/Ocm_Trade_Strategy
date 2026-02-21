#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 卸载脚本
# 功能: 完全卸载程序和所有配置
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="com.ocm.tradestrategy"
INSTALL_DIR="$HOME/.ocm_trade_strategy"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         OCM Trade Strategy - macOS 卸载程序                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 确认卸载
read -p "确定要卸载 OCM Trade Strategy 吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}卸载已取消${NC}"
    exit 0
fi

echo -e "${BLUE}正在卸载...${NC}"

# 1. 停止服务
echo -e "${BLUE}停止服务...${NC}"
if [ -f "$PLIST_PATH" ]; then
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
fi

# 杀死任何残留进程
pkill -f "streamlit.*ocm_streamlit" 2>/dev/null || true

echo -e "${GREEN}✓ 服务已停止${NC}"

# 2. 删除安装目录
echo -e "${BLUE}删除安装文件...${NC}"
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
fi
echo -e "${GREEN}✓ 安装文件已删除${NC}"

# 3. 清理别名（提示用户手动处理）
echo -e "${YELLOW}提示: 如果您添加了 shell 别名，请手动从 ~/.zshrc 或 ~/.bash_profile 中删除:${NC}"
echo "    source ~/.ocm_trade_strategy/aliases.sh"

echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    卸载完成！                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
