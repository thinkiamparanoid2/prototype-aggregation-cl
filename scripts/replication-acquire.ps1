param(
    [ValidateSet('PlantDoc', 'PlantWild', 'PlantVillage', 'all')][string]$Dataset = 'all',
    [string]$SourceUrl,
    [switch]$Execute
)

# This launcher deliberately has no default side effect.  A source URL is supplied by the
# approved replication plan; it is never guessed here because licence/access status is a gate.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataRoot = Join-Path $ProjectRoot 'ssl_hdc_pilot/replication/data'
$Names = @('PlantDoc', 'PlantWild', 'PlantVillage')
if ($Dataset -ne 'all') { $Names = @($Dataset) }

foreach ($name in $Names) {
    $target = Join-Path $DataRoot $name
    "PREPARE dataset=$name target=$target manifest=$target/sha256_manifest.csv source_url=$(if ($SourceUrl) {$SourceUrl} else {'REQUIRES_APPROVED_PLAN'})"
}
if (-not $Execute) {
    Write-Output 'Prepare-only: no download, extraction, adaptation, capture, or evaluation was launched. Re-run with -Execute after plan approval.'
    return
}
if ($Names.Count -ne 1 -or [string]::IsNullOrWhiteSpace($SourceUrl)) {
    throw '-Execute requires exactly one -Dataset and its approved -SourceUrl; access/licence must be recorded before download.'
}
if ((Read-Host 'Type ACQUIRE REPLICATION DATA') -cne 'ACQUIRE REPLICATION DATA') { throw 'Cancelled before acquisition.' }
New-Item -ItemType Directory -Force -Path (Join-Path $DataRoot $Names[0]) | Out-Null
$archive = Join-Path (Join-Path $DataRoot $Names[0]) 'source_download.zip'
Invoke-WebRequest -Uri $SourceUrl -OutFile $archive
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
[pscustomobject]@{ dataset=$Names[0]; source_url=$SourceUrl; downloaded_utc=(Get-Date).ToUniversalTime().ToString('o'); archive='source_download.zip'; sha256=$hash } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Join-Path $DataRoot $Names[0]) 'acquisition_receipt.json') -Encoding utf8
Write-Output "ACQUIRED dataset=$($Names[0]) archive_sha256=$hash; extraction and split construction remain separate gated steps."
