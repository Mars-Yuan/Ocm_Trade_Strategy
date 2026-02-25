# ============================================================
# OCM Trade Strategy - Windows 启动脚本
# 使用方法: 
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_windows.ps1
# ============================================================

# --- 自动处理 ExecutionPolicy：如果当前策略受限，用 Bypass 重新启动自身 ---
try {
    $currentPolicy = Get-ExecutionPolicy -Scope Process
} catch {
    $currentPolicy = "Restricted"
}
if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "AllSigned") {
    $selfPath = $MyInvocation.MyCommand.Path
    if ($selfPath -and (Test-Path $selfPath)) {
        Start-Process powershell.exe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$selfPath`"")
        exit
    }
}

$ErrorActionPreference = "Stop"

$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"
$Port = 8501

Write-Host "" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "     OCM Trade Strategy - 启动服务                         " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

# 停止已有进程
Get-Process | Where-Object {
    $_.ProcessName -like "*streamlit*" -or $_.ProcessName -eq "python"
} | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {}
}

# 方法1：尝试通过计划任务启动
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "-> 通过计划任务启动服务..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 4

    # 检查服务是否成功启动
    $maxRetries = 10
    $started = $false
    for ($i = 0; $i -lt $maxRetries; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$Port" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            $started = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if ($started) {
        Write-Host "[OK] 服务已成功启动!" -ForegroundColor Green
        Write-Host "  访问地址: http://localhost:$Port" -ForegroundColor Cyan
        Start-Process "http://localhost:$Port"
        exit 0
    } else {
        Write-Host "[!] 计划任务启动未响应，尝试直接启动..." -ForegroundColor Yellow
    }
} else {
    Write-Host "[!] 未找到计划任务，尝试直接启动..." -ForegroundColor Yellow
}

# 方法2：直接启动 Streamlit（兜底方案）
$streamlitExe = "$InstallDir\venv\Scripts\streamlit.exe"
$appFile = "$InstallDir\ocm_streamlit_Streamlit.py"

if (-not (Test-Path $streamlitExe)) {
    Write-Host "[错误] 未找到 streamlit: $streamlitExe" -ForegroundColor Red
    Write-Host "  请重新安装: irm https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/main/scripts/quick_install_windows.ps1 | iex" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "-> 直接启动 Streamlit 服务..." -ForegroundColor Cyan
Set-Location $InstallDir

# 后台启动
$logFile = "$InstallDir\logs\streamlit.log"
if (-not (Test-Path "$InstallDir\logs")) { New-Item -ItemType Directory -Path "$InstallDir\logs" -Force | Out-Null }

Start-Process -FilePath $streamlitExe `
    -ArgumentList "run", $appFile, "--server.port", $Port, "--server.headless", "true", "--server.address", "localhost" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$InstallDir\logs\streamlit_error.log"

# 等待服务就绪
$maxRetries = 15
$started = $false
for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $started = $true
        break
    } catch {}
}

if ($started) {
    Write-Host "[OK] 服务已成功启动!" -ForegroundColor Green
    Write-Host "  访问地址: http://localhost:$Port" -ForegroundColor Cyan
    Start-Process "http://localhost:$Port"
} else {
    Write-Host "[!] 启动超时，请查看日志:" -ForegroundColor Yellow
    Write-Host "  $logFile" -ForegroundColor Cyan
    Write-Host "  $InstallDir\logs\streamlit_error.log" -ForegroundColor Cyan
    Read-Host "按回车键退出"
}
