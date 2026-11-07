$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $projectRoot "runtime"
$pidFile = Join-Path $runtimeDir "public-tunnel.pid"
$urlFile = Join-Path $runtimeDir "public-url.txt"
$tokenFile = Join-Path $runtimeDir "cloudflare-tunnel.token"

if (Test-Path -LiteralPath $pidFile) {
    $tunnelPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "cloudflared") {
        Stop-Process -Id $tunnelPid
        Wait-Process -Id $tunnelPid -Timeout 10 -ErrorAction SilentlyContinue
    }
}

Remove-Item -LiteralPath $pidFile, $urlFile, $tokenFile `
    -Force -ErrorAction SilentlyContinue
Write-Output "Cloudflare tunnel stopped."
