# ============================================================
# OCM Trade Strategy - Windows 停止脚本
# 使用方法: 
#   powershell -NoProfile -ExecutionPolicy Bypass -File stop_windows.ps1
# ============================================================

# --- 自动处理 ExecutionPolicy ---
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

$ErrorActionPreference = "SilentlyContinue"

$TaskName = "OCM_Trade_Strategy"

Write-Host "-> 正在停止 OCM Trade Strategy 服务..." -ForegroundColor Cyan

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task -and $task.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName
}

$processes = Get-Process | Where-Object { $_.ProcessName -like "*streamlit*" -or $_.CommandLine -like "*ocm_streamlit*" }
$processes | Stop-Process -Force

Get-Process | Where-Object { 
    $_.ProcessName -eq "python" -and 
    $_.MainWindowTitle -like "*streamlit*" 
} | Stop-Process -Force

Start-Sleep -Seconds 1

Write-Host "[OK] 服务已停止" -ForegroundColor Green
