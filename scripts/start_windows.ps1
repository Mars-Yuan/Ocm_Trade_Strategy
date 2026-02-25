# ============================================================
# OCM Trade Strategy - Windows Start Script
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File start_windows.ps1
# ============================================================

# --- Auto-handle ExecutionPolicy: relaunch with Bypass if restricted ---
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
Write-Host "     OCM Trade Strategy - Start Service                     " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

# Stop existing processes
Get-Process | Where-Object {
    $_.ProcessName -like "*streamlit*" -or $_.ProcessName -eq "python"
} | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {}
}

# Method 1: Try starting via Scheduled Task
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "-> Starting via Scheduled Task..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 4

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
        Write-Host "[OK] Service started successfully!" -ForegroundColor Green
        Write-Host "  URL: http://localhost:$Port" -ForegroundColor Cyan
        Start-Process "http://localhost:$Port"
        exit 0
    } else {
        Write-Host "[!] Scheduled Task did not respond, trying direct start..." -ForegroundColor Yellow
    }
} else {
    Write-Host "[!] Scheduled Task not found, trying direct start..." -ForegroundColor Yellow
}

# Method 2: Direct Streamlit start (fallback)
$streamlitExe = "$InstallDir\venv\Scripts\streamlit.exe"
$appFile = "$InstallDir\ocm_streamlit_Streamlit.py"

if (-not (Test-Path $streamlitExe)) {
    Write-Host "[ERROR] streamlit not found: $streamlitExe" -ForegroundColor Red
    Write-Host "  Please reinstall: irm https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/main/scripts/quick_install_windows.ps1 | iex" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "-> Starting Streamlit directly..." -ForegroundColor Cyan
Set-Location $InstallDir

$logFile = "$InstallDir\logs\streamlit.log"
if (-not (Test-Path "$InstallDir\logs")) { New-Item -ItemType Directory -Path "$InstallDir\logs" -Force | Out-Null }

Start-Process -FilePath $streamlitExe `
    -ArgumentList "run", $appFile, "--server.port", $Port, "--server.headless", "true", "--server.address", "localhost" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$InstallDir\logs\streamlit_error.log"

# Wait for service ready
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
    Write-Host "[OK] Service started successfully!" -ForegroundColor Green
    Write-Host "  URL: http://localhost:$Port" -ForegroundColor Cyan
    Start-Process "http://localhost:$Port"
} else {
    Write-Host "[!] Start timeout. Check logs:" -ForegroundColor Yellow
    Write-Host "  $logFile" -ForegroundColor Cyan
    Write-Host "  $InstallDir\logs\streamlit_error.log" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
}
