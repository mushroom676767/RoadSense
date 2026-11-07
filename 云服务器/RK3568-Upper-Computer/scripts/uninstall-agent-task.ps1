param(
    [string]$TaskName = "RK3568 Device Agent"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$agentScriptPattern = [regex]::Escape((Join-Path $repo "agent.py"))
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$agentProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "python*.exe" -and
    $_.CommandLine -and
    $_.CommandLine -match $agentScriptPattern
}
foreach ($agentProcess in $agentProcesses) {
    Stop-Process -Id $agentProcess.ProcessId -Force -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddSeconds(10)
while (
    (Get-Date) -lt $deadline -and
    (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue)
) {
    Start-Sleep -Milliseconds 250
}
if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) {
    throw "Port 5000 is still listening after removing the Agent task."
}

Write-Output (
    "Stopped Agent processes and removed scheduled task '$TaskName'."
)
