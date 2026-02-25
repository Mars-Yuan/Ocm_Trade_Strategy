# ============================================================
# OCM Trade Strategy - Windows Uninstall Script
# Usage: powershell -ExecutionPolicy Bypass -File uninstall_windows.ps1
# ============================================================

$ErrorActionPreference = "SilentlyContinue"
$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "       OCM Trade Strategy - Windows Uninstall               " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

$response = Read-Host "Confirm uninstall OCM Trade Strategy? (Y/N)"
if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "Uninstall cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host "-> Stopping scheduled task..." -ForegroundColor Cyan
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "   Scheduled task removed." -ForegroundColor Gray
}

Write-Host "-> Stopping processes..." -ForegroundColor Cyan
Get-Process | Where-Object { $_.ProcessName -like "*streamlit*" } | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "   Waiting for processes to release files..." -ForegroundColor Gray
Start-Sleep -Seconds 3
Write-Host "[OK] Processes stopped." -ForegroundColor Green

Write-Host "-> Removing files..." -ForegroundColor Cyan
if (Test-Path $InstallDir) {
    $retryCount = 0
    $maxRetries = 3
    $deleted = $false
    while (-not $deleted -and $retryCount -lt $maxRetries) {
        try {
            Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction Stop
            $deleted = $true
        } catch {
            $retryCount++
            if ($retryCount -lt $maxRetries) {
                Write-Host "   Retry $retryCount of $maxRetries..." -ForegroundColor Yellow
                Start-Sleep -Seconds 2
            }
        }
    }
    if (-not $deleted) {
        Write-Host "   Using alternative method..." -ForegroundColor Yellow
        cmd /c rmdir /s /q "$InstallDir"
        if (Test-Path $InstallDir) {
            Write-Host "[WARNING] Some files could not be deleted." -ForegroundColor Yellow
            Write-Host "   Please restart your computer and delete manually:" -ForegroundColor Yellow
            Write-Host "   $InstallDir" -ForegroundColor White
        } else {
            $deleted = $true
        }
    }
    if ($deleted) {
        Write-Host "[OK] Files removed." -ForegroundColor Green
    }
} else {
    Write-Host "   Install directory not found, skipping." -ForegroundColor Gray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "       Uninstall completed!                                 " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""







