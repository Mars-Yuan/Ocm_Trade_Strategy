# ============================================================
# OCM Trade Strategy - Windows Stop Script
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File stop_windows.ps1
# ============================================================

# --- Auto-handle ExecutionPolicy ---
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

Write-Host "-> Stopping OCM Trade Strategy service..." -ForegroundColor Cyan

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

Write-Host "[OK] Service stopped." -ForegroundColor Green
