# ============================================================
# OCM Trade Strategy - Windows 
# : 
# :  PowerShell
# ============================================================

#Requires -RunAsAdministrator

param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$AppName = "OCM_Trade_Strategy"
$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"
$LogDir = "$InstallDir\logs"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

function Write-Success($message) {
    Write-Host "[OK] $message" -ForegroundColor Green
}

function Write-Info($message) {
    Write-Host "-> $message" -ForegroundColor Cyan
}

function Write-Warn($message) {
    Write-Host "[!] $message" -ForegroundColor Yellow
}

function Write-ErrorSafe($message) {
    try {
        Write-Host $message -ForegroundColor Red
    } catch {
        [Console]::Error.WriteLine($message)
    }
}

# 
function Fix-ProxySettings {
    Write-Info "..."
    
    $proxyFixed = $false
    
    # 
    $proxyEnvVars = @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    foreach ($proxyVar in $proxyEnvVars) {
        $proxyValue = [Environment]::GetEnvironmentVariable($proxyVar, "Process")
        if (-not $proxyValue -or -not ($proxyValue -match "127\.0\.0\.1|localhost")) {
            continue
        }

        try {
            $proxyUri = [System.Uri]$proxyValue
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect($proxyUri.Host, $proxyUri.Port)
            $tcpClient.Close()
        } catch {
            Write-Warn " $proxyVar=$proxyValue"
            [Environment]::SetEnvironmentVariable($proxyVar, $null, "Process")
            $proxyFixed = $true
        }
    }
    
    #  Git 
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $gitHttpProxy = git config --global --get http.proxy 2>$null
        $gitHttpsProxy = git config --global --get https.proxy 2>$null
        
        if ($gitHttpProxy -match "127\.0\.0\.1|localhost") {
            try {
                $proxyUri = [System.Uri]$gitHttpProxy
                $tcpClient = New-Object System.Net.Sockets.TcpClient
                $tcpClient.Connect($proxyUri.Host, $proxyUri.Port)
                $tcpClient.Close()
            } catch {
                Write-Warn " Git  $gitHttpProxy"
                git config --global --unset http.proxy 2>$null
                $proxyFixed = $true
            }
        }
        
        if ($gitHttpsProxy -match "127\.0\.0\.1|localhost") {
            try {
                $proxyUri = [System.Uri]$gitHttpsProxy
                $tcpClient = New-Object System.Net.Sockets.TcpClient
                $tcpClient.Connect($proxyUri.Host, $proxyUri.Port)
                $tcpClient.Close()
            } catch {
                Write-Warn " Git HTTPS  $gitHttpsProxy"
                git config --global --unset https.proxy 2>$null
                $proxyFixed = $true
            }
        }
    }
    
    #  pip 
    $env:PIP_NO_PROXY = "*"
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
    $env:PIP_CONFIG_FILE = "NUL"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    
    if ($proxyFixed) {
        Write-Success ""
    } else {
        Write-Success ""
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "       OCM Trade Strategy - Windows             " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

function Check-Python {
    Write-Info " Python ..."
    
    $pythonCommands = @("python", "python3", "py")
    $pythonCmd = $null
    
    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>&1
            if ($version -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 9) {
                    $pythonCmd = $cmd
                    Write-Success "Python $($Matches[0]) "
                    break
                }
            }
        } catch {
            continue
        }
    }
    
    if (-not $pythonCmd) {
        Write-Warn "Python 3.9+ "
        Write-Info " https://www.python.org/downloads/  Python 3.9+"
        Write-Info " 'Add Python to PATH'"
        
        $response = Read-Host " winget  Python? (Y/N)"
        if ($response -eq "Y" -or $response -eq "y") {
            winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
            $pythonCmd = "python"
        } else {
            throw " Python 3.9+"
        }
    }
    
    return $pythonCmd
}

function Create-Directories {
    Write-Info "..."
    
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    if (-not (Test-Path "$InstallDir\scripts")) {
        New-Item -ItemType Directory -Path "$InstallDir\scripts" -Force | Out-Null
    }
    
    Write-Success ""
}

function Copy-ProjectFiles {
    Write-Info "..."
    
    Copy-Item "$ProjectDir\ocm_streamlit_Streamlit.py" "$InstallDir\" -Force
    Copy-Item "$ProjectDir\Streamlit_data.json" "$InstallDir\" -Force
    Copy-Item "$ProjectDir\requirements.txt" "$InstallDir\" -Force
    
    Get-ChildItem "$ProjectDir\scripts\*.ps1" | ForEach-Object {
        Copy-Item $_.FullName "$InstallDir\scripts\" -Force
    }
    
    Write-Success ""
}

function Setup-VirtualEnv {
    param($PythonCmd)
    
    Write-Info "..."
    
    Set-Location $InstallDir
    
    if (Test-Path "venv") {
        Remove-Item -Recurse -Force "venv"
    }
    
    & $PythonCmd -m venv venv
    
    Write-Info "..."
    
    #  pip 
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
    $env:PIP_NO_PROXY = "*"
    $env:PIP_CONFIG_FILE = "NUL"
    $proxyEnvVars = @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "PIP_PROXY")
    foreach ($proxyVar in $proxyEnvVars) {
        [Environment]::SetEnvironmentVariable($proxyVar, $null, "Process")
    }
    $pipArgs = @("--isolated", "--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org")
    
    try {
        & "$InstallDir\venv\Scripts\python.exe" -m pip install --upgrade pip -q @pipArgs 2>$null
    } catch {
        Write-Warn "pip "
    }
    
    & "$InstallDir\venv\Scripts\python.exe" -m pip install -r requirements.txt @pipArgs
    
    Write-Success ""
}

function Create-StartupScript {
    Write-Info "..."
    
    $startupScript = @"
`$InstallDir = "$InstallDir"
`$LogDir = "$LogDir"
`$Port = $Port

Set-Location `$InstallDir
& "`$InstallDir\venv\Scripts\streamlit.exe" run ocm_streamlit_Streamlit.py --server.port `$Port --server.headless true --server.address localhost *>> "`$LogDir\streamlit.log"
"@
    
    $startupScript | Out-File -FilePath "$InstallDir\startup.ps1" -Encoding UTF8
    
    $vbsScript = @"
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""$InstallDir\startup.ps1""", 0, False
"@
    
    $vbsScript | Out-File -FilePath "$InstallDir\startup_hidden.vbs" -Encoding ASCII
    
    Write-Success ""
}

function Create-ScheduledTask {
    Write-Info "..."
    
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    
    $action = New-ScheduledTaskAction `
        -Execute "wscript.exe" `
        -Argument "`"$InstallDir\startup_hidden.vbs`"" `
        -WorkingDirectory $InstallDir
    
    $trigger = New-ScheduledTaskTrigger -AtLogon
    
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)
    
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "OCM Trade Strategy Dashboard - " | Out-Null
    
    Write-Success ""
}

function Start-Service {
    Write-Info "..."
    
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    
    Write-Success ""
}

function Open-Browser {
    Write-Info "..."
    
    Start-Sleep -Seconds 2
    
    $maxRetries = 10
    for ($i = 0; $i -lt $maxRetries; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$Port" -UseBasicParsing -TimeoutSec 2
            Start-Process "http://localhost:$Port"
            Write-Success ""
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    
    Write-Warn ": http://localhost:$Port"
}

function Show-Summary {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "                                                   " -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  : $InstallDir" -ForegroundColor Cyan
    Write-Host "  : $LogDir" -ForegroundColor Cyan
    Write-Host "  : http://localhost:$Port" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  :" -ForegroundColor Cyan
    Write-Host "    : $InstallDir\scripts\start_windows.ps1"
    Write-Host "    : $InstallDir\scripts\stop_windows.ps1"
    Write-Host "    : Get-Content $LogDir\streamlit.log -Tail 50"
    Write-Host "    : $InstallDir\scripts\uninstall_windows.ps1"
    Write-Host ""
    Write-Host "  [OK] " -ForegroundColor Green
    Write-Host "  [OK] " -ForegroundColor Green
    Write-Host ""
}

function Main {
    try {
        Fix-ProxySettings
        $pythonCmd = Check-Python
        Create-Directories
        Copy-ProjectFiles
        Setup-VirtualEnv -PythonCmd $pythonCmd
        Create-StartupScript
        Create-ScheduledTask
        Start-Service
        Open-Browser
        Show-Summary
    } catch {
        $errorMessage = if ($_.Exception -and $_.Exception.Message) {
            $_.Exception.Message
        } else {
            ($_ | Out-String).Trim()
        }

        Write-ErrorSafe ":"
        Write-ErrorSafe $errorMessage
        exit 1
    }
}

Main
