# ============================================================
# OCM Trade Strategy - Windows 
# :  GitHub 
# ============================================================

$ErrorActionPreference = "Stop"

# Clear proxy environment variables to avoid connection issues
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:ALL_PROXY = ""
$env:all_proxy = ""
$env:NO_PROXY = "*"
$env:no_proxy = "*"

$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"
$RepoUrl = "https://github.com/Mars-Yuan/Ocm_Trade_Strategy.git"
$TempDir = "$env:TEMP\ocm_upgrade_$(Get-Random)"
$ScriptVersion = "2026-02-18.3"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "       OCM Trade Strategy - Windows                 " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ": $ScriptVersion" -ForegroundColor DarkGray
Write-Host ""

function Write-ErrorSafe($message) {
    try {
        Write-Host $message -ForegroundColor Red
    } catch {
        [Console]::Error.WriteLine($message)
    }
}

function Invoke-PipNoProxy {
    param(
        [string[]]$PipArgs,
        [switch]$IgnoreFailure
    )

    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    $pipCmd = $PipArgs -join " "
    $cmdLine = "set HTTP_PROXY=& set HTTPS_PROXY=& set ALL_PROXY=& set http_proxy=& set https_proxy=& set all_proxy=& set PIP_PROXY=& set PIP_CONFIG_FILE=NUL& set NO_PROXY=*& set no_proxy=*& `"$pythonExe`" -m pip $pipCmd"

    cmd.exe /d /c $cmdLine
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $IgnoreFailure) {
        throw "pip  (exit code: $exitCode)"
    }
}

function Clear-PipProxyConfig {
    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        return
    }

    $unsetTargets = @(
        @("--global", "global.proxy"),
        @("--user", "global.proxy"),
        @("--site", "global.proxy"),
        @("--global", "install.proxy"),
        @("--user", "install.proxy"),
        @("--site", "install.proxy")
    )

    foreach ($target in $unsetTargets) {
        try {
            & $pythonExe -m pip config $target[0] unset $target[1] 1>$null 2>$null
        } catch {
        }
    }
}

# 
function Fix-ProxySettings {
    Write-Host "-> ..." -ForegroundColor Cyan
    
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
            Write-Host "[WARN]  $proxyVar=$proxyValue" -ForegroundColor Yellow
            [Environment]::SetEnvironmentVariable($proxyVar, $null, "Process")
            $proxyFixed = $true
        }
    }
    
    #  Git 
    $gitHttpProxy = git config --global --get http.proxy 2>$null
    $gitHttpsProxy = git config --global --get https.proxy 2>$null
    
    if ($gitHttpProxy -match "127\.0\.0\.1|localhost") {
        try {
            $proxyUri = [System.Uri]$gitHttpProxy
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect($proxyUri.Host, $proxyUri.Port)
            $tcpClient.Close()
        } catch {
            Write-Host "[WARN]  Git  $gitHttpProxy" -ForegroundColor Yellow
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
            Write-Host "[WARN]  Git HTTPS  $gitHttpsProxy" -ForegroundColor Yellow
            git config --global --unset https.proxy 2>$null
            $proxyFixed = $true
        }
    }
    
    #  pip 
    $env:PIP_NO_PROXY = "*"
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
    $env:PIP_CONFIG_FILE = "NUL"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    
    if ($proxyFixed) {
        Write-Host "[OK] " -ForegroundColor Green
    } else {
        Write-Host "[OK] " -ForegroundColor Green
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host ":  Git" -ForegroundColor Yellow
    Write-Host " https://git-scm.com/downloads "
    exit 1
}

function Backup-Data {
    Write-Host "-> ..." -ForegroundColor Cyan
    if (Test-Path "$InstallDir\Streamlit_data.json") {
        Copy-Item "$InstallDir\Streamlit_data.json" "$env:TEMP\Streamlit_data_backup.json" -Force
        Write-Host "[OK] " -ForegroundColor Green
    }
}

function Stop-AppService {
    Write-Host "-> ..." -ForegroundColor Cyan
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task -and $task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName
    }
    
    Get-Process | Where-Object { $_.ProcessName -like "*streamlit*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "[OK] " -ForegroundColor Green
}

function Download-Latest {
    Write-Host "-> ..." -ForegroundColor Cyan
    
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    $downloadOk = $false

    try {
        git clone --depth 1 $RepoUrl $TempDir
        if ($LASTEXITCODE -eq 0 -and (Test-Path "$TempDir\ocm_streamlit_Streamlit.py")) {
            $downloadOk = $true
        }
    } catch {
        $downloadOk = $false
    }

    if (-not $downloadOk) {
        Write-Host "[WARN] Git  ZIP ..." -ForegroundColor Yellow

        $zipPath = Join-Path $env:TEMP ("ocm_upgrade_" + (Get-Random) + ".zip")
        $extractRoot = Join-Path $env:TEMP ("ocm_upgrade_extract_" + (Get-Random))

        try {
            Invoke-WebRequest -Uri "https://github.com/Mars-Yuan/Ocm_Trade_Strategy/archive/refs/heads/main.zip" -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force

            $zipRepoDir = Join-Path $extractRoot "Ocm_Trade_Strategy-main"
            if (Test-Path "$zipRepoDir\ocm_streamlit_Streamlit.py") {
                Copy-Item "$zipRepoDir\*" $TempDir -Recurse -Force
                $downloadOk = $true
            }
        } catch {
            $downloadOk = $false
        } finally {
            if (Test-Path $zipPath) {
                Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
            }
            if (Test-Path $extractRoot) {
                Remove-Item -Path $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    if (-not $downloadOk -or -not (Test-Path "$TempDir\ocm_streamlit_Streamlit.py")) {
        throw " GitHub  ocm_streamlit_Streamlit.py/"
    }

    Write-Host "[OK] " -ForegroundColor Green
}

function Update-Files {
    Write-Host "-> ..." -ForegroundColor Cyan

    if (-not (Test-Path "$InstallDir\scripts")) {
        New-Item -ItemType Directory -Path "$InstallDir\scripts" -Force | Out-Null
    }
    
    Copy-Item "$TempDir\ocm_streamlit_Streamlit.py" "$InstallDir\" -Force
    Copy-Item "$TempDir\requirements.txt" "$InstallDir\" -Force
    
    if (Test-Path "$TempDir\scripts") {
        Get-ChildItem "$TempDir\scripts\*.ps1" | ForEach-Object {
            Copy-Item $_.FullName "$InstallDir\scripts\" -Force
        }
    }
    
    Write-Host "[OK] " -ForegroundColor Green
}

function Update-Dependencies {
    Write-Host "-> ..." -ForegroundColor Cyan
    
    Set-Location $InstallDir
    
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

    Clear-PipProxyConfig
    
    try {
        Invoke-PipNoProxy -PipArgs (@("install", "--upgrade", "pip", "-q") + $pipArgs) -IgnoreFailure
    } catch {
        Write-Host "[WARN] pip " -ForegroundColor Yellow
    }
    
    Invoke-PipNoProxy -PipArgs (@("install", "-r", "requirements.txt", "--upgrade", "-q") + $pipArgs)
    
    Write-Host "[OK] " -ForegroundColor Green
}

function Restore-Data {
    Write-Host "-> ..." -ForegroundColor Cyan
    if (Test-Path "$env:TEMP\Streamlit_data_backup.json") {
        Copy-Item "$env:TEMP\Streamlit_data_backup.json" "$InstallDir\Streamlit_data.json" -Force
        Remove-Item "$env:TEMP\Streamlit_data_backup.json" -Force
        Write-Host "[OK] " -ForegroundColor Green
    }
}

function Start-AppService {
    Write-Host "-> ..." -ForegroundColor Cyan
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Start-ScheduledTask -TaskName $TaskName
    }
    
    Start-Sleep -Seconds 2
    Write-Host "[OK] " -ForegroundColor Green
}

function Cleanup {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

try {
    Fix-ProxySettings
    Backup-Data
    Stop-AppService
    Download-Latest
    Update-Files
    Update-Dependencies
    Restore-Data
    Start-AppService
    Cleanup
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "                                                   " -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  : http://localhost:8501" -ForegroundColor Cyan
    
} catch {
    $errorMessage = if ($_.Exception -and $_.Exception.Message) {
        $_.Exception.Message
    } else {
        ($_ | Out-String).Trim()
    }

    Write-ErrorSafe ":"
    Write-ErrorSafe $errorMessage
    Cleanup
    exit 1
}
