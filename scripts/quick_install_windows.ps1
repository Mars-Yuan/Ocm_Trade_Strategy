# ============================================================
# OCM Trade Strategy - Windows 
# 
#  ( PowerShell):
# 
# 1 -  Commit SHAraw  ZIP:
# powershell -NoProfile -ExecutionPolicy Bypass -Command '$sha="3d92896316347029c7695a20aa10d866b34c2c24"; $raw="https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/$sha/scripts/quick_install_windows.ps1"; try { $code=(Invoke-WebRequest -Uri $raw -UseBasicParsing -Headers @{''Cache-Control''=''no-cache'';''Pragma''=''no-cache''}).Content; if (-not $code -or $code -notmatch "OCM Trade Strategy") { throw "invalid content" }; & ([ScriptBlock]::Create($code)) } catch { $zip=Join-Path $env:TEMP "ocm_quick_install_$sha.zip"; $dst=Join-Path $env:TEMP "ocm_quick_install_$sha"; if(Test-Path $zip){Remove-Item $zip -Force -ErrorAction SilentlyContinue}; if(Test-Path $dst){Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue}; Invoke-WebRequest -Uri ("https://codeload.github.com/Mars-Yuan/Ocm_Trade_Strategy/zip/"+$sha) -OutFile $zip -UseBasicParsing; Expand-Archive -Path $zip -DestinationPath $dst -Force; $script=Join-Path $dst ("Ocm_Trade_Strategy-"+$sha+"\scripts\quick_install_windows.ps1"); powershell -NoProfile -ExecutionPolicy Bypass -File $script }'
#
# 2 - :
# Set-ExecutionPolicy Bypass -Scope Process -Force; <1>
# ============================================================

$PinnedCommit = "3d92896316347029c7695a20aa10d866b34c2c24"
$RepoSlug = "Mars-Yuan/Ocm_Trade_Strategy"
$thisScriptPath = $MyInvocation.MyCommand.Path

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] ..." -ForegroundColor Yellow

    if ($thisScriptPath -and (Test-Path $thisScriptPath)) {
        Start-Process powershell.exe -Verb RunAs -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$thisScriptPath`"")
        exit
    }

    $bootstrapScript = @"
`$sha = "$PinnedCommit"
`$raw = "https://raw.githubusercontent.com/$RepoSlug/`$sha/scripts/quick_install_windows.ps1"
try {
    `$code = (Invoke-WebRequest -Uri `$raw -UseBasicParsing -Headers @{'Cache-Control'='no-cache';'Pragma'='no-cache'}).Content
    if (-not `$code -or `$code -notmatch "OCM Trade Strategy") { throw "invalid content" }
    & ([ScriptBlock]::Create(`$code))
} catch {
    `$zip = Join-Path `$env:TEMP "ocm_quick_install_`$sha.zip"
    `$dst = Join-Path `$env:TEMP "ocm_quick_install_`$sha"
    if (Test-Path `$zip) { Remove-Item `$zip -Force -ErrorAction SilentlyContinue }
    if (Test-Path `$dst) { Remove-Item `$dst -Recurse -Force -ErrorAction SilentlyContinue }
    Invoke-WebRequest -Uri ("https://codeload.github.com/$RepoSlug/zip/" + `$sha) -OutFile `$zip -UseBasicParsing
    Expand-Archive -Path `$zip -DestinationPath `$dst -Force
    `$script = Join-Path `$dst ("Ocm_Trade_Strategy-" + `$sha + "\scripts\quick_install_windows.ps1")
    powershell -NoProfile -ExecutionPolicy Bypass -File `$script
}
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($bootstrapScript))
    Start-Process powershell.exe -Verb RunAs -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded)
    exit
}

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Mars-Yuan/Ocm_Trade_Strategy.git"
$RepoName = "Ocm_Trade_Strategy"
$TaskName = "OCM_Trade_Strategy"
$InstallDir = "$env:USERPROFILE\.ocm_trade_strategy"
$LogDir = "$InstallDir\logs"
$TempDir = "$env:TEMP\ocm_install_$(Get-Random)"
$Port = 8501

# Clear proxy environment variables to avoid connection issues
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:ALL_PROXY = ""
$env:all_proxy = ""
$env:NO_PROXY = "*"
$env:no_proxy = "*"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "     OCM Trade Strategy - Windows           " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

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

function Invoke-PipNoProxy {
    param(
        [string[]]$PipArgs,
        [switch]$IgnoreFailure
    )

    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    $pipCmd = $PipArgs -join " "
    $cmdLine = "set HTTP_PROXY=& set HTTPS_PROXY=& set ALL_PROXY=& set http_proxy=& set https_proxy=& set all_proxy=& set PIP_PROXY=& set PIP_CONFIG_FILE=NUL& set NO_PROXY=*& set no_proxy=*& set PIP_INDEX_URL=& set PIP_EXTRA_INDEX_URL=& `"$pythonExe`" -m pip $pipCmd"

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

function Check-Git {
    Write-Info " Git..."
    
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Success "Git "
        return
    }
    
    Write-Warn "Git ..."
    
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install Git.Git --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Write-Host " https://git-scm.com/downloads  Git" -ForegroundColor Red
        throw " Git"
    }
    
    Write-Success "Git "
}

function Check-Python {
    Write-Info " Python..."
    
    $pythonCommands = @("python", "python3", "py")
    $pythonCmd = $null
    
    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>&1
            if ($version -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 9) {
                    $script:PythonCmd = $cmd
                    Write-Success "Python $($Matches[0]) "
                    return
                }
            }
        } catch {
            continue
        }
    }
    
    Write-Warn "Python 3.9+ ..."
    
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $script:PythonCmd = "python"
        Write-Success "Python "
    } else {
        Write-Host " https://www.python.org/downloads/  Python 3.9+" -ForegroundColor Red
        throw " Python 3.9+"
    }
}

function Download-Project {
    Write-Info "..."
    
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    $downloadSuccess = $false
    
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Info " Git ..."
        
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"

        try {
            $cloneResult = & git -c http.version=HTTP/1.1 clone --depth 1 $RepoUrl "$TempDir\$RepoName" 2>&1
            $cloneExitCode = $LASTEXITCODE
        } catch {
            $cloneResult = $_
            $cloneExitCode = 1
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }

        if ($cloneExitCode -eq 0 -and (Test-Path "$TempDir\$RepoName\ocm_streamlit_Streamlit.py")) {
            $downloadSuccess = $true
            Write-Success "Git "
        } else {
            Write-Warn "Git "
        }
        
        if (-not $downloadSuccess -and (Test-Path "$TempDir\$RepoName")) {
            Remove-Item -Path "$TempDir\$RepoName" -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    if (-not $downloadSuccess) {
        Write-Info " HTTPS  ZIP ..."
        
        $zipPath = "$TempDir\repo.zip"
        $zipUrls = @(
            "https://github.com/Mars-Yuan/Ocm_Trade_Strategy/archive/refs/heads/main.zip",
            "https://codeload.github.com/Mars-Yuan/Ocm_Trade_Strategy/zip/refs/heads/main"
        )
        
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

        foreach ($zipUrl in $zipUrls) {
            if ($downloadSuccess) { break }
            try {
                Write-Info ": $zipUrl"
                Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
                
                if (Test-Path $zipPath) {
                    Write-Info "..."
                    Expand-Archive -Path $zipPath -DestinationPath $TempDir -Force
                    
                    $extractedDir = "$TempDir\Ocm_Trade_Strategy-main"
                    if (Test-Path $extractedDir) {
                        Rename-Item -Path $extractedDir -NewName $RepoName -Force
                        $downloadSuccess = $true
                        Write-Success "ZIP "
                    }
                    
                    Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
                }
            } catch {
                Write-Warn "ZIP : $($_.Exception.Message)"
            }
        }
    }
    
    if (-not $downloadSuccess -or -not (Test-Path "$TempDir\$RepoName\ocm_streamlit_Streamlit.py")) {
        Write-Host ""
        Write-Host "" -ForegroundColor Red
        Write-Host ": https://github.com/Mars-Yuan/Ocm_Trade_Strategy" -ForegroundColor Cyan
        throw ""
    }
    
    Write-Success ""
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

function Copy-Files {
    Write-Info "..."
    
    $src = "$TempDir\$RepoName"
    
    Copy-Item "$src\ocm_streamlit_Streamlit.py" "$InstallDir\" -Force
    Copy-Item "$src\Streamlit_data.json" "$InstallDir\" -Force
    Copy-Item "$src\requirements.txt" "$InstallDir\" -Force
    
    Get-ChildItem "$src\scripts\*.ps1" | ForEach-Object {
        Copy-Item $_.FullName "$InstallDir\scripts\" -Force
    }
    
    # Re-encode all .ps1 files to UTF-8 with BOM (PowerShell 5.1 compatibility)
    Get-ChildItem "$InstallDir\scripts\*.ps1" | ForEach-Object {
        try {
            $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
            $utf8Bom = New-Object System.Text.UTF8Encoding $true
            [System.IO.File]::WriteAllText($_.FullName, $content, $utf8Bom)
        } catch {}
    }
    
    Write-Success ""
}

function Setup-VirtualEnv {
    Write-Info " Python ..."
    
    Set-Location $InstallDir
    
    if (Test-Path "venv") {
        Remove-Item -Recurse -Force "venv"
    }
    
    & $script:PythonCmd -m venv venv
    
    Write-Info " ()..."
    
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
        Invoke-PipNoProxy -PipArgs @("install", "--upgrade", "pip", "-q") -IgnoreFailure
    } catch {
        Write-Warn "pip "
    }
    
    $installArgs = @("install", "-r", "requirements.txt", "-q") + $pipArgs
    Invoke-PipNoProxy -PipArgs $installArgs
    
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
    
    # 创建 .bat 启动器作为备用方案
    $batScript = @"
@echo off
cd /d "$InstallDir"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "$InstallDir\startup.ps1"
"@
    $batScript | Out-File -FilePath "$InstallDir\startup.bat" -Encoding ASCII
    
    Write-Success "启动脚本已创建"
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
    
    $maxRetries = 15
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

function Cleanup {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Show-Summary {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "                                               " -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  : http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "  : $InstallDir" -ForegroundColor Cyan
    Write-Host "  : $LogDir" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  常用命令:" -ForegroundColor Cyan
    Write-Host "    启动: powershell -NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\scripts\start_windows.ps1`""
    Write-Host "    停止: powershell -NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\scripts\stop_windows.ps1`""
    Write-Host "    卸载: powershell -NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\scripts\uninstall_windows.ps1`""
    Write-Host '    : powershell -NoProfile -ExecutionPolicy Bypass -Command "$sha=''3d92896316347029c7695a20aa10d866b34c2c24''; $raw=''https://raw.githubusercontent.com/Mars-Yuan/Ocm_Trade_Strategy/'' + $sha + ''/scripts/upgrade_windows.ps1''; try { $code=(Invoke-WebRequest -Uri $raw -UseBasicParsing -Headers @{''Cache-Control''=''no-cache'';''Pragma''=''no-cache''}).Content; if (-not $code -or $code -notmatch ''OCM Trade Strategy'') { throw ''invalid content'' }; & ([ScriptBlock]::Create($code)) } catch { $zip=Join-Path $env:TEMP (''ocm_upgrade_''+$sha+''.zip''); $dst=Join-Path $env:TEMP (''ocm_upgrade_''+$sha); if(Test-Path $zip){Remove-Item $zip -Force -ErrorAction SilentlyContinue}; if(Test-Path $dst){Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue}; Invoke-WebRequest -Uri (''https://codeload.github.com/Mars-Yuan/Ocm_Trade_Strategy/zip/''+$sha) -OutFile $zip -UseBasicParsing; Expand-Archive -Path $zip -DestinationPath $dst -Force; $script=Join-Path $dst (''Ocm_Trade_Strategy-''+$sha+''\scripts\upgrade_windows.ps1''); powershell -NoProfile -ExecutionPolicy Bypass -File $script }"'
    Write-Host ""
    Write-Host "  [OK] " -ForegroundColor Green
    Write-Host "  [OK] " -ForegroundColor Green
    Write-Host ""
}

function Main {
    try {
        Fix-ProxySettings
        Check-Git
        Check-Python
        Download-Project
        Create-Directories
        Copy-Files
        Setup-VirtualEnv
        Create-StartupScript
        Create-ScheduledTask
        Start-Service
        Cleanup
        Open-Browser
        Show-Summary
        return $true
    } catch {
        $errorMessage = if ($_.Exception -and $_.Exception.Message) {
            $_.Exception.Message
        } else {
            ($_ | Out-String).Trim()
        }

        Write-ErrorSafe ":"
        Write-ErrorSafe $errorMessage
        Cleanup
        return $false
    }
}

$result = Main
if (-not $result) {
    exit 1
}
