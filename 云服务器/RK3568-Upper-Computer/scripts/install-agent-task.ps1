param(
    [string]$TaskName = "RK3568 Device Agent",
    [string]$EnvironmentFile = "runtime/rendered/agent/agent.env"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$startScript = Join-Path $repo "scripts/start-agent.ps1"
$environmentPath = Join-Path $repo $EnvironmentFile
$python = Join-Path $repo ".venv/Scripts/python.exe"

foreach ($requiredPath in @($startScript, $environmentPath, $python)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Agent file is missing: $requiredPath"
    }
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$powerShell = Join-Path $PSHOME "powershell.exe"
$escapedStartScript = $startScript.Replace('"', '""')
$escapedEnvironmentFile = $EnvironmentFile.Replace('"', '""')
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden " +
    "-File `"$escapedStartScript`" -EnvironmentFile `"$escapedEnvironmentFile`""

$action = New-ScheduledTaskAction `
    -Execute $powerShell `
    -Argument $arguments `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$watchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -Hidden `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

$task = New-ScheduledTask `
    -Action $action `
    -Trigger @($trigger, $watchdogTrigger) `
    -Principal $principal `
    -Settings $settings `
    -Description "Persistent RK3568 Windows device Agent"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Output "Installed and started scheduled task '$TaskName' for $currentUser."
Write-Output "Agent health endpoint: http://127.0.0.1:5000/health"
