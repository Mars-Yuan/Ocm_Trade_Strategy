#!/bin/bash
# ============================================================
# OCM Trade Strategy - macOS 一键下载安装脚本
# 
# 使用方法 (在终端中运行):
# curl -fsSL https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/main/scripts/quick_install_mac.sh | bash
# 
# 或者下载后运行:
# curl -O https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/main/scripts/quick_install_mac.sh
# chmod +x quick_install_mac.sh && ./quick_install_mac.sh
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
REPO_URL="https://github.com/Mars-Yuan/Ocm_Trade_Strategy.git"
REPO_NAME="Ocm_Trade_Strategy"
APP_NAME="OCM_Trade_Strategy"
SERVICE_NAME="com.ocm.tradestrategy"
INSTALL_DIR="$HOME/.ocm_trade_strategy"
LOG_DIR="$INSTALL_DIR/logs"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
PORT=8501
TEMP_DIR="/tmp/ocm_install_$$"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     OCM Trade Strategy - macOS 一键下载安装程序            ║"
echo "║                                                            ║"
echo "║  本脚本将自动完成:                                         ║"
echo "║  1. 下载最新版本代码                                       ║"
echo "║  2. 安装 Python 依赖                                       ║"
echo "║  3. 配置开机自启动                                         ║"
echo "║  4. 启动后台服务                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 检查是否为 macOS
check_os() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo -e "${RED}错误: 此脚本仅支持 macOS 系统${NC}"
        echo "Windows 用户请使用: quick_install_windows.ps1"
        exit 1
    fi
    echo -e "${GREEN}✓ 操作系统: macOS${NC}"
}

# 检查必要工具
check_prerequisites() {
    echo -e "${BLUE}检查必要工具...${NC}"
    
    # 检查 git
    if ! command -v git &> /dev/null; then
        echo -e "${YELLOW}Git 未安装，尝试安装...${NC}"
        if command -v brew &> /dev/null; then
            brew install git
        else
            echo -e "${RED}错误: 请先安装 Git${NC}"
            echo "  运行: xcode-select --install"
            exit 1
        fi
    fi
    echo -e "${GREEN}✓ Git 已安装${NC}"
    
    # 检查 curl
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}错误: curl 未安装${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ curl 已安装${NC}"
}

# 检查 Python
check_python() {
    echo -e "${BLUE}检查 Python 环境...${NC}"
    
    PYTHON_CMD=""
    
    # 优先使用 python3
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR_VERSION" -ge 3 ] && [ "$MINOR_VERSION" -ge 9 ]; then
            PYTHON_CMD="python3"
            echo -e "${GREEN}✓ Python $PYTHON_VERSION 已安装${NC}"
            return 0
        fi
    fi
    
    # 尝试安装 Python
    echo -e "${YELLOW}Python 3.9+ 未找到，尝试安装...${NC}"
    
    # 检查/安装 Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}安装 Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # 配置 Homebrew 路径
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f "/usr/local/bin/brew" ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    
    echo -e "${YELLOW}安装 Python 3.11...${NC}"
    brew install python@3.11
    PYTHON_CMD="python3"
    
    echo -e "${GREEN}✓ Python 已安装${NC}"
}

# 下载项目
download_project() {
    echo -e "${BLUE}下载项目代码...${NC}"
    
    # 清理临时目录
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    
    # 克隆仓库
    git clone --depth 1 "$REPO_URL" "$TEMP_DIR/$REPO_NAME"
    
    echo -e "${GREEN}✓ 项目下载完成${NC}"
}

# 创建目录结构
create_directories() {
    echo -e "${BLUE}创建安装目录...${NC}"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$INSTALL_DIR/scripts"
    
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 复制文件
copy_files() {
    echo -e "${BLUE}安装项目文件...${NC}"
    
    local src="$TEMP_DIR/$REPO_NAME"
    
    # 复制主要文件
    cp "$src/ocm_streamlit_Streamlit.py" "$INSTALL_DIR/"
    cp "$src/Streamlit_data.json" "$INSTALL_DIR/"
    cp "$src/requirements.txt" "$INSTALL_DIR/"
    
    # 复制脚本
    cp "$src/scripts/"*.sh "$INSTALL_DIR/scripts/" 2>/dev/null || true
    chmod +x "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true
    
    echo -e "${GREEN}✓ 文件安装完成${NC}"
}

# 创建虚拟环境
setup_venv() {
    echo -e "${BLUE}创建 Python 虚拟环境...${NC}"
    
    cd "$INSTALL_DIR"
    
    # 删除旧的虚拟环境
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    
    # 创建虚拟环境
    $PYTHON_CMD -m venv venv
    
    # 安装依赖
    echo -e "${BLUE}安装依赖包 (这可能需要几分钟)...${NC}"
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate
    
    echo -e "${GREEN}✓ 虚拟环境配置完成${NC}"
}

# 配置 launchd 服务
setup_launchd() {
    echo -e "${BLUE}配置开机自启动服务...${NC}"
    
    # 停止现有服务
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    
    # 创建 plist 文件
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
    
    echo -e "${GREEN}✓ 开机自启动已配置${NC}"
}

# 启动服务
start_service() {
    echo -e "${BLUE}启动服务...${NC}"
    
    launchctl load "$PLIST_PATH"
    
    # 等待服务启动
    sleep 3
    
    if launchctl list | grep -q "$SERVICE_NAME"; then
        echo -e "${GREEN}✓ 服务已启动${NC}"
    else
        echo -e "${YELLOW}⚠ 服务可能未正确启动，请检查日志: $LOG_DIR${NC}"
    fi
}

# 打开浏览器
open_browser() {
    echo -e "${BLUE}打开浏览器...${NC}"
    
    sleep 2
    
    # 等待服务就绪
    for i in {1..15}; do
        if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
            open "http://localhost:$PORT"
            echo -e "${GREEN}✓ 浏览器已打开${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${YELLOW}请手动打开浏览器访问: http://localhost:$PORT${NC}"
}

# 清理临时文件
cleanup() {
    rm -rf "$TEMP_DIR"
}

# 创建快捷命令脚本
create_shortcuts() {
    cat > "$INSTALL_DIR/aliases.sh" << 'EOF'
# OCM Trade Strategy 快捷命令
alias ocm-start='~/.ocm_trade_strategy/scripts/start_mac.sh'
alias ocm-stop='~/.ocm_trade_strategy/scripts/stop_mac.sh'
alias ocm-status='launchctl list | grep com.ocm.tradestrategy'
alias ocm-logs='tail -f ~/.ocm_trade_strategy/logs/streamlit_stdout.log'
alias ocm-open='open http://localhost:8501'
EOF
}

# 显示安装摘要
show_summary() {
    echo ""
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 🎉 安装成功完成！                          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "  ${BLUE}访问地址:${NC} http://localhost:$PORT"
    echo -e "  ${BLUE}安装目录:${NC} $INSTALL_DIR"
    echo -e "  ${BLUE}日志目录:${NC} $LOG_DIR"
    echo ""
    echo -e "  ${BLUE}常用命令:${NC}"
    echo "    启动: $INSTALL_DIR/scripts/start_mac.sh"
    echo "    停止: $INSTALL_DIR/scripts/stop_mac.sh"
    echo "    卸载: $INSTALL_DIR/scripts/uninstall_mac.sh"
    echo "    升级: $INSTALL_DIR/scripts/upgrade_mac.sh"
    echo ""
    echo -e "  ${GREEN}✓ 服务已配置为开机自启动${NC}"
    echo -e "  ${GREEN}✓ 关闭终端窗口不会影响服务运行${NC}"
    echo -e "  ${GREEN}✓ 重启电脑后服务会自动启动${NC}"
    echo ""
    echo -e "  ${YELLOW}提示: 添加快捷命令到 shell:${NC}"
    echo "    echo 'source ~/.ocm_trade_strategy/aliases.sh' >> ~/.zshrc"
    echo ""
}

# 主函数
main() {
    check_os
    check_prerequisites
    check_python
    download_project
    create_directories
    copy_files
    setup_venv
    setup_launchd
    start_service
    create_shortcuts
    cleanup
    open_browser
    show_summary
}

# 捕获错误
trap 'echo -e "${RED}安装过程中出现错误，请检查上方错误信息${NC}"; cleanup; exit 1' ERR

# 运行安装
main
