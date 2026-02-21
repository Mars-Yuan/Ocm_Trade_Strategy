#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 升级脚本
# 功能: 从 GitHub 拉取最新版本并重新安装
# ============================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="com.ocm.tradestrategy"
INSTALL_DIR="$HOME/.ocm_trade_strategy"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
REPO_URL="https://github.com/Mars-Yuan/Ocm_Trade_Strategy.git"
TEMP_DIR="/tmp/ocm_upgrade_$$"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         OCM Trade Strategy - macOS 升级程序                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 检查 git 是否安装
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}错误: 需要安装 Git${NC}"
    exit 1
fi

# 备份当前数据文件
backup_data() {
    echo -e "${BLUE}备份数据文件...${NC}"
    if [ -f "$INSTALL_DIR/Streamlit_data.json" ]; then
        cp "$INSTALL_DIR/Streamlit_data.json" "/tmp/Streamlit_data_backup.json"
        echo -e "${GREEN}✓ 数据已备份${NC}"
    fi
}

# 停止服务
stop_service() {
    echo -e "${BLUE}停止服务...${NC}"
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    pkill -f "streamlit.*ocm_streamlit" 2>/dev/null || true
    echo -e "${GREEN}✓ 服务已停止${NC}"
}

# 下载最新版本
download_latest() {
    echo -e "${BLUE}下载最新版本...${NC}"
    
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    
    git clone --depth 1 "$REPO_URL" "$TEMP_DIR"
    
    echo -e "${GREEN}✓ 最新版本已下载${NC}"
}

# 更新文件
update_files() {
    echo -e "${BLUE}更新文件...${NC}"
    
    # 更新主要文件
    cp "$TEMP_DIR/ocm_streamlit_Streamlit.py" "$INSTALL_DIR/"
    cp "$TEMP_DIR/requirements.txt" "$INSTALL_DIR/"
    
    # 更新脚本
    if [ -d "$TEMP_DIR/scripts" ]; then
        cp "$TEMP_DIR/scripts/"*.sh "$INSTALL_DIR/scripts/" 2>/dev/null || true
        chmod +x "$INSTALL_DIR/scripts/"*.sh
    fi
    
    echo -e "${GREEN}✓ 文件已更新${NC}"
}

# 更新依赖
update_dependencies() {
    echo -e "${BLUE}更新依赖包...${NC}"
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt --upgrade
    deactivate
    
    echo -e "${GREEN}✓ 依赖已更新${NC}"
}

# 恢复数据
restore_data() {
    echo -e "${BLUE}恢复数据文件...${NC}"
    if [ -f "/tmp/Streamlit_data_backup.json" ]; then
        cp "/tmp/Streamlit_data_backup.json" "$INSTALL_DIR/Streamlit_data.json"
        rm "/tmp/Streamlit_data_backup.json"
        echo -e "${GREEN}✓ 数据已恢复${NC}"
    fi
}

# 启动服务
start_service() {
    echo -e "${BLUE}启动服务...${NC}"
    if [ -f "$PLIST_PATH" ]; then
        launchctl load "$PLIST_PATH"
    fi
    sleep 2
    echo -e "${GREEN}✓ 服务已启动${NC}"
}

# 清理临时文件
cleanup() {
    rm -rf "$TEMP_DIR"
}

# 主函数
main() {
    backup_data
    stop_service
    download_latest
    update_files
    update_dependencies
    restore_data
    start_service
    cleanup
    
    echo ""
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    升级完成！                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  ${BLUE}访问地址:${NC} http://localhost:8501"
}

main
