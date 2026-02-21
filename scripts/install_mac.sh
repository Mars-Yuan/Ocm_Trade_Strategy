#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 一键安装脚本
# 功能: 安装依赖、配置开机自启动、启动后台服务
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
APP_NAME="OCM_Trade_Strategy"
SERVICE_NAME="com.ocm.tradestrategy"
INSTALL_DIR="$HOME/.ocm_trade_strategy"
LOG_DIR="$INSTALL_DIR/logs"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
PORT=8501

# 获取脚本所在目录（项目根目录）
get_script_dir() {
    local SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do
        local DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    echo "$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"
}

PROJECT_DIR="$(get_script_dir)"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         OCM Trade Strategy - macOS 一键安装程序            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 检查是否为 macOS
check_os() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo -e "${RED}错误: 此脚本仅支持 macOS 系统${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 操作系统检测通过: macOS${NC}"
}

# 检查 Python 版本
check_python() {
    echo -e "${BLUE}正在检查 Python 环境...${NC}"
    
    # 优先使用 python3
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR_VERSION" -ge 3 ] && [ "$MINOR_VERSION" -ge 9 ]; then
            echo -e "${GREEN}✓ Python $PYTHON_VERSION 已安装${NC}"
            return 0
        fi
    fi
    
    # 尝试使用 Homebrew 安装
    echo -e "${YELLOW}Python 3.9+ 未找到，尝试使用 Homebrew 安装...${NC}"
    
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Homebrew 未安装，正在安装...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install python@3.11
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ Python 已通过 Homebrew 安装${NC}"
}

# 创建目录结构
create_directories() {
    echo -e "${BLUE}正在创建目录结构...${NC}"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$HOME/Library/LaunchAgents"
    
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 复制项目文件
copy_project_files() {
    echo -e "${BLUE}正在复制项目文件...${NC}"
    
    # 复制主要文件
    cp "$PROJECT_DIR/ocm_streamlit_Streamlit.py" "$INSTALL_DIR/"
    cp "$PROJECT_DIR/Streamlit_data.json" "$INSTALL_DIR/"
    cp "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/"
    
    # 复制脚本
    mkdir -p "$INSTALL_DIR/scripts"
    cp "$PROJECT_DIR/scripts/"*.sh "$INSTALL_DIR/scripts/" 2>/dev/null || true
    chmod +x "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true
    
    echo -e "${GREEN}✓ 项目文件复制完成${NC}"
}

# 创建虚拟环境并安装依赖
setup_venv() {
    echo -e "${BLUE}正在创建虚拟环境...${NC}"
    
    cd "$INSTALL_DIR"
    
    # 删除旧的虚拟环境（如果存在）
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    
    # 创建新的虚拟环境
    $PYTHON_CMD -m venv venv
    
    # 激活并安装依赖
    source venv/bin/activate
    
    echo -e "${BLUE}正在安装依赖包...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    
    deactivate
    
    echo -e "${GREEN}✓ 虚拟环境和依赖安装完成${NC}"
}

# 创建 launchd 服务配置
create_launchd_service() {
    echo -e "${BLUE}正在配置开机自启动服务...${NC}"
    
    # 先停止现有服务
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${SERVICE_NAME}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/venv/bin/streamlit</string>
        <string>run</string>
        <string>${INSTALL_DIR}/ocm_streamlit_Streamlit.py</string>
        <string>--server.port</string>
        <string>${PORT}</string>
        <string>--server.headless</string>
        <string>true</string>
        <string>--server.address</string>
        <string>localhost</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/streamlit_stdout.log</string>
    
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/streamlit_stderr.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${INSTALL_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
    
    echo -e "${GREEN}✓ 开机自启动服务配置完成${NC}"
}

# 启动服务
start_service() {
    echo -e "${BLUE}正在启动后台服务...${NC}"
    
    # 加载服务
    launchctl load "$PLIST_PATH"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务状态
    if launchctl list | grep -q "$SERVICE_NAME"; then
        echo -e "${GREEN}✓ 后台服务已启动${NC}"
    else
        echo -e "${YELLOW}⚠ 服务可能未正确启动，请检查日志${NC}"
    fi
}

# 打开浏览器
open_browser() {
    echo -e "${BLUE}正在打开浏览器...${NC}"
    
    # 等待服务完全启动
    sleep 2
    
    # 检查端口是否可用
    for i in {1..10}; do
        if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
            open "http://localhost:$PORT"
            echo -e "${GREEN}✓ 浏览器已打开${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${YELLOW}⚠ 请手动打开浏览器访问: http://localhost:$PORT${NC}"
}

# 创建快捷命令
create_shortcuts() {
    echo -e "${BLUE}正在创建快捷命令...${NC}"
    
    # 创建 shell 别名配置
    ALIAS_FILE="$INSTALL_DIR/aliases.sh"
    cat > "$ALIAS_FILE" << 'EOF'
# OCM Trade Strategy 快捷命令
alias ocm-start='~/.ocm_trade_strategy/scripts/start_mac.sh'
alias ocm-stop='~/.ocm_trade_strategy/scripts/stop_mac.sh'
alias ocm-status='launchctl list | grep com.ocm.tradestrategy'
alias ocm-logs='tail -f ~/.ocm_trade_strategy/logs/streamlit_stdout.log'
EOF
    
    # 提示用户添加到 shell 配置
    echo -e "${YELLOW}提示: 若要使用快捷命令，请将以下行添加到 ~/.zshrc 或 ~/.bash_profile:${NC}"
    echo -e "    source ~/.ocm_trade_strategy/aliases.sh"
}

# 显示安装摘要
show_summary() {
    echo ""
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    安装完成！                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "  ${BLUE}安装目录:${NC} $INSTALL_DIR"
    echo -e "  ${BLUE}日志目录:${NC} $LOG_DIR"
    echo -e "  ${BLUE}访问地址:${NC} http://localhost:$PORT"
    echo ""
    echo -e "  ${BLUE}常用命令:${NC}"
    echo "    启动服务: $INSTALL_DIR/scripts/start_mac.sh"
    echo "    停止服务: $INSTALL_DIR/scripts/stop_mac.sh"
    echo "    查看日志: tail -f $LOG_DIR/streamlit_stdout.log"
    echo "    卸载程序: $INSTALL_DIR/scripts/uninstall_mac.sh"
    echo ""
    echo -e "  ${GREEN}✓ 服务已配置为开机自启动${NC}"
    echo -e "  ${GREEN}✓ 关闭终端窗口不会影响服务运行${NC}"
    echo ""
}

# 主函数
main() {
    check_os
    check_python
    create_directories
    copy_project_files
    setup_venv
    create_launchd_service
    start_service
    create_shortcuts
    open_browser
    show_summary
}

# 运行安装
main
