param(
    [string]$EnvironmentFile = "runtime/rendered/agent/agent.env"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envPath = (Resolve-Path (Join-Path $repo $EnvironmentFile)).Path
$python = Join-Path $repo ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment is missing: $python"
}

foreach ($line in Get-Content -LiteralPath $envPath -Encoding utf8) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }
    $parts = $trimmed.Split("=", 2)
    if ($parts.Count -ne 2 -or -not $parts[0]) {
        throw "invalid environment line in $envPath"
    }
    [Environment]::SetEnvironmentVariable(
        $parts[0],
        $parts[1],
        "Process"
    )
}

$requiredVariables = @(
    "AGENT_CLOUD_URL",
    "AGENT_ID",
    "AGENT_TOKEN",
    "SSH_HOST",
    "SSH_USER",
    "SSH_PASSWORD",
    "PUBLISH_URL",
    "PUBLISH_TOKEN",
    "MEDIA_MTX_SRT_HOST",
    "MEDIA_MTX_SRT_PASSPHRASE",
    "MEDIA_MTX_PUBLISH_USERNAME",
    "MEDIA_MTX_PUBLISH_PASSWORD"
)
foreach ($name in $requiredVariables) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        throw "Required Agent setting is missing: $name"
    }
}

$ffmpegSetting = [Environment]::GetEnvironmentVariable("FFMPEG_PATH", "Process")
$ffmpegCommand = if ($ffmpegSetting) { $ffmpegSetting } else { "ffmpeg" }
$ffmpeg = Get-Command $ffmpegCommand -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "FFmpeg was not found. Install a build with NVENC and libsrt support."
}
$encoders = (& $ffmpeg.Source -hide_banner -encoders 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $encoders -notmatch "\bh264_nvenc\b") {
    throw "FFmpeg does not provide the h264_nvenc encoder."
}
$protocols = (& $ffmpeg.Source -hide_banner -protocols 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $protocols -notmatch "(?m)^\s+srt\s*$") {
    throw "FFmpeg does not provide the SRT protocol."
}

$health = $null
try {
    $health = Invoke-RestMethod `
        -Uri "http://127.0.0.1:5000/health" `
        -TimeoutSec 2
} catch {
    $health = $null
}
if ($health -and $health.ok -and $health.role -eq "agent") {
    Write-Output "RK3568 Device Agent is already healthy."
    exit 0
}

$listener = Get-NetTCPConnection `
    -LocalPort 5000 `
    -State Listen `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $listenerProcess = Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($listener.OwningProcess)"
    $agentPattern = [regex]::Escape((Join-Path $repo "agent.py"))
    if (
        -not $listenerProcess.CommandLine -or
        $listenerProcess.CommandLine -notmatch $agentPattern
    ) {
        throw "Port 5000 is occupied by a service that is not the device Agent."
    }

    Stop-Process -Id $listener.OwningProcess -Force
    $deadline = (Get-Date).AddSeconds(10)
    while (
        (Get-Date) -lt $deadline -and
        (Get-NetTCPConnection `
            -LocalPort 5000 `
            -State Listen `
            -ErrorAction SilentlyContinue)
    ) {
        Start-Sleep -Milliseconds 250
    }
    if (Get-NetTCPConnection `
        -LocalPort 5000 `
        -State Listen `
        -ErrorAction SilentlyContinue
    ) {
        throw "Unhealthy Agent did not release port 5000."
    }
}

Push-Location $repo
try {
    Write-Output "Starting RK3568 Device Agent in the foreground. Press Ctrl+C to stop."
    & $python (Join-Path $repo "agent.py")
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
