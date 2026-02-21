# ============================================================
# OCM Trade Strategy - Windows 
# : 
# ============================================================

$ErrorActionPreference = "SilentlyContinue"

$TaskName = "OCM_Trade_Strategy"

Write-Host "->  OCM Trade Strategy ..." -ForegroundColor Cyan

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

Write-Host "[OK] " -ForegroundColor Green
