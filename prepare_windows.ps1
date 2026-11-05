[CmdletBinding()]
param([string]$WorkDir = 'C:\gemma-aw')
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($WorkDir)
$downloads = Join-Path $root 'downloads'
New-Item -ItemType Directory -Force -Path $downloads | Out-Null
Get-Command curl.exe -ErrorAction Stop | Out-Null
$files = @(
    @{
        Name = 'termux-arm64.apk'
        Url = 'https://github.com/termux/termux-app/releases/download/v0.118.3/termux-app_v0.118.3%2Bgithub-debug_arm64-v8a.apk'
    },
    @{
        Name = 'gemma-4-E4B_q4_0-it.gguf'
        Url = 'https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/gemma-4-E4B_q4_0-it.gguf?download=true'
    }
)
foreach ($item in $files) {
    $destination = Join-Path $downloads $item.Name
    Write-Host "Downloading $($item.Name)"
    & curl.exe --fail --location --retry 3 --continue-at - --output $destination $item.Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($item.Name). Run the same command again to resume." }
}
Get-FileHash -LiteralPath (Join-Path $downloads 'gemma-4-E4B_q4_0-it.gguf') -Algorithm SHA256 |
    Format-List | Out-File -LiteralPath (Join-Path $downloads 'model_sha256.txt') -Encoding utf8
Write-Host "DOWNLOADS READY: $downloads"
