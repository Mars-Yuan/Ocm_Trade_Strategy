# ============================================================
# OCM Trade Strategy - Windows 
# : 
# ============================================================

$ErrorActionPreference = "Stop"

$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"
$Port = 8501

Write-Host "->  OCM Trade Strategy ..." -ForegroundColor Cyan

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "[!] " -ForegroundColor Yellow
    exit 1
}

Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
if ($taskInfo.LastTaskResult -eq 0 -or $task.State -eq "Running") {
    Write-Host "[OK] " -ForegroundColor Green
    Write-Host "  : http://localhost:$Port" -ForegroundColor Cyan
    Start-Process "http://localhost:$Port"
} else {
    Write-Host "[!] " -ForegroundColor Yellow
    Write-Host "  : $InstallDir\logs\streamlit.log"
}
