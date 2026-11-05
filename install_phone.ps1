[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Serial,
    [string]$WorkDir = 'C:\gemma-aw'
)
$ErrorActionPreference = 'Stop'
Get-Command adb -ErrorAction Stop | Out-Null
$root = [IO.Path]::GetFullPath($WorkDir)
$downloads = Join-Path $root 'downloads'
foreach ($name in @('termux-arm64.apk', 'gemma-4-E4B_q4_0-it.gguf')) {
    if (-not (Test-Path -LiteralPath (Join-Path $downloads $name))) { throw "Missing download: $name" }
}
& adb -s $Serial get-state
if ($LASTEXITCODE -ne 0) { throw 'Phone unavailable. Check USB debugging and authorization.' }
& adb -s $Serial install -r (Join-Path $downloads 'termux-arm64.apk')
if ($LASTEXITCODE -ne 0) { throw 'Termux install failed. Check whether a differently signed Termux is already installed.' }
& adb -s $Serial shell 'mkdir -p /sdcard/Download/gemma-aw'
if ($LASTEXITCODE -ne 0) { throw 'Cannot create phone staging directory.' }
foreach ($file in @(
    (Join-Path $downloads 'gemma-4-E4B_q4_0-it.gguf'),
    (Join-Path $PSScriptRoot 'setup_phone.sh'),
    (Join-Path $PSScriptRoot 'start_phone.sh')
)) {
    & adb -s $Serial push $file /sdcard/Download/gemma-aw/
    if ($LASTEXITCODE -ne 0) { throw "Phone transfer failed: $file" }
}
Write-Host 'INSTALL COMPLETE. Open Termux on the phone and continue with the numbered guide.'
