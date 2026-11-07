param(
    [string]$RuleName = "RK3568 Device Agent (Ethernet)"
)

$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script once from an elevated PowerShell window."
}

Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule
New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 5000 `
    -RemoteAddress 192.168.137.0/24 `
    -Profile Any `
    -Description "Allow only the RK3568 direct Ethernet subnet to publish JPEG and metadata." |
    Out-Null

Write-Output "Configured '$RuleName' for TCP 5000 from 192.168.137.0/24 only."
