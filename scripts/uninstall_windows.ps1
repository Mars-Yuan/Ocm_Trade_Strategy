# ============================================================
# OCM Trade Strategy - Windows Uninstall Script
# Usage: powershell -ExecutionPolicy Bypass -File uninstall_windows.ps1
# ============================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = "SilentlyContinue"
$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"

function Write-Info($message) {
    Write-Host "-> $message" -ForegroundColor Cyan
}

function Write-Success($message) {
    Write-Host "[OK] $message" -ForegroundColor Green
}

function Write-Warn($message) {
    Write-Host "[WARNING] $message" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "       OCM Trade Strategy - Windows Uninstall               " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

$response = Read-Host "Confirm uninstall OCM Trade Strategy? (Y/N)"
if ($response -notin @("Y", "y")) {
    Write-Host "Uninstall cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Info "Stopping scheduled task..."
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Success "Scheduled task removed."
} else {
    Write-Host "   Scheduled task not found, skipping." -ForegroundColor Gray
}

Write-Info "Stopping related processes..."
Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Success "Processes stopped."

Write-Info "Removing files..."
if (Test-Path $InstallDir) {
    $deleted = $false
    $maxRetries = 3

    for ($retry = 1; $retry -le $maxRetries; $retry++) {
        try {
            Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction Stop
            $deleted = $true
            break
        } catch {
            if ($retry -lt $maxRetries) {
                Write-Host "   Retry $retry of $maxRetries..." -ForegroundColor Yellow
                Start-Sleep -Seconds 2
            }
        }
    }

    if (-not $deleted) {
        Write-Host "   Using fallback removal method..." -ForegroundColor Yellow
        cmd.exe /c "rmdir /s /q `"$InstallDir`""
        if (-not (Test-Path $InstallDir)) {
            $deleted = $true
        }
    }

    if ($deleted) {
        Write-Success "Files removed."
    } else {
        Write-Warn "Some files could not be deleted."
        Write-Host "   Please reboot Windows and remove this path manually:" -ForegroundColor Yellow
        Write-Host "   $InstallDir" -ForegroundColor White
    }
} else {
    Write-Host "   Install directory not found, skipping." -ForegroundColor Gray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "       Uninstall completed!                                 " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""







