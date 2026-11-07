param(
    [string]$Origin = "http://127.0.0.1:5000",
    [string]$CloudflaredPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"
$runtimeDir = Join-Path $projectRoot "runtime"
$logsDir = Join-Path $projectRoot "logs"
$pidFile = Join-Path $runtimeDir "public-tunnel.pid"
$urlFile = Join-Path $runtimeDir "public-url.txt"
$tokenFile = Join-Path $runtimeDir "cloudflare-tunnel.token"
$stdoutLog = Join-Path $logsDir "cloudflare-tunnel.out.log"
$stderrLog = Join-Path $logsDir "cloudflare-tunnel.err.log"

function Get-DotEnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Test-Path -LiteralPath $envFile)) {
        return ""
    }
    $escapedName = [regex]::Escape($Name)
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding utf8) {
        if ($line -match "^\s*$escapedName\s*=(.*)$") {
            $value = $Matches[1].Trim()
            if (
                $value.Length -ge 2 -and
                (($value[0] -eq '"' -and $value[-1] -eq '"') -or
                 ($value[0] -eq "'" -and $value[-1] -eq "'"))
            ) {
                return $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return ""
}

function Resolve-Cloudflared {
    $candidates = @(
        $CloudflaredPath,
        $env:CLOUDFLARED_BIN,
        (Get-DotEnvValue "CLOUDFLARED_BIN"),
        (Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe"),
        (Join-Path $env:USERPROFILE "Downloads\cloudflared-windows-amd64.exe")
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw @"
cloudflared.exe was not found. Download the Windows amd64 binary from:
https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
Then place it at:
$env:LOCALAPPDATA\cloudflared\cloudflared.exe
"@
}

$tunnelToken = $env:CLOUDFLARE_TUNNEL_TOKEN
if (-not $tunnelToken) {
    $tunnelToken = Get-DotEnvValue "CLOUDFLARE_TUNNEL_TOKEN"
}
if (-not $tunnelToken) {
    throw @"
CLOUDFLARE_TUNNEL_TOKEN is missing from .env.
Create a remotely managed tunnel in Cloudflare Zero Trust, add a public
hostname whose service is $Origin, and copy its token into .env.
"@
}

$publicUrl = $env:CLOUDFLARE_PUBLIC_URL
if (-not $publicUrl) {
    $publicUrl = Get-DotEnvValue "CLOUDFLARE_PUBLIC_URL"
}
$cloudflared = Resolve-Cloudflared

try {
    Invoke-WebRequest -Uri "$Origin/api/config" -UseBasicParsing -TimeoutSec 3 |
        Out-Null
} catch {
    throw "Manager is not reachable at $Origin. Start app.py first."
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing -and $existing.ProcessName -eq "cloudflared") {
        if ($publicUrl) {
            Write-Output $publicUrl
        } else {
            Write-Output "Cloudflare tunnel is already running (PID $existingPid)."
        }
        exit 0
    }
}

Remove-Item -LiteralPath $pidFile, $urlFile, $tokenFile, $stdoutLog, $stderrLog `
    -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $tokenFile -Value $tunnelToken -Encoding ascii -NoNewline

$process = Start-Process `
    -FilePath $cloudflared `
    -ArgumentList @(
        "tunnel",
        "--no-autoupdate",
        "--url", $Origin,
        "--loglevel", "info",
        "run",
        "--token-file", $tokenFile
    ) `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

try {
    $connected = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($process.HasExited) {
            $details = Get-Content -LiteralPath $stderrLog -Raw `
                -ErrorAction SilentlyContinue
            throw "Cloudflare tunnel exited before connecting. $details"
        }
        $output = (
            (Get-Content -LiteralPath $stdoutLog -Raw -ErrorAction SilentlyContinue) +
            "`n" +
            (Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue)
        )
        if ($output -match "Registered tunnel connection") {
            $connected = $true
            break
        }
    }
    if (-not $connected) {
        throw "Timed out waiting for Cloudflare tunnel registration."
    }
} catch {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile, $urlFile, $tokenFile `
        -Force -ErrorAction SilentlyContinue
    throw
}

if ($publicUrl) {
    Set-Content -LiteralPath $urlFile -Value $publicUrl -Encoding utf8
    Write-Output $publicUrl
} else {
    Write-Output "Cloudflare tunnel connected (PID $($process.Id))."
    Write-Output "Set CLOUDFLARE_PUBLIC_URL in .env to show its fixed URL here."
}
