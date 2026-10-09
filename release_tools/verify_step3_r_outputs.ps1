# Run the original and modified Step 3 scripts with identical native arguments,
# paths, inputs, R runtime and libraries. Every exported file is compared directly.
# Example:
# powershell -File release_tools/verify_step3_r_outputs.ps1 `
#   -BaselineScript RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R `
#   -CandidateScript RAW_OCT_PROCESSING_2023_09SEP-05_WSU_optimized.R `
#   -FixtureDirectory "Examples/R_TEST/flip human"
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BaselineScript,
    [Parameter(Mandatory = $true)][string]$CandidateScript,
    [Parameter(Mandatory = $true)][string]$FixtureDirectory,
    [string]$Rscript = 'C:\Program Files\R\R-3.3.1\bin\x64\Rscript.exe',
    [string]$RLibrary = '',
    [string]$WorkDirectory = '',
    [string]$ImageIndex = '1,2',
    [string]$PixelWidth = '3.89',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$CandidateName = 'candidate',
    [ValidateRange(1, 32)][int]$Workers = 1,
    [ValidateSet('all', 'baseline', 'repeat', 'candidate', 'compare')]
    [string]$Phase = 'all',
    [switch]$Diagnostics,
    [switch]$RepeatBaseline
)

$ErrorActionPreference = 'Stop'
if ($CandidateName -in @('baseline', 'repeat')) {
    throw 'CandidateName must not overwrite a baseline snapshot.'
}
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$baselinePath = (Resolve-Path -LiteralPath $BaselineScript).Path
$candidatePath = (Resolve-Path -LiteralPath $CandidateScript).Path
$fixturePath = (Resolve-Path -LiteralPath $FixtureDirectory).Path
$rscriptPath = (Resolve-Path -LiteralPath $Rscript).Path
if (!$RLibrary) { $RLibrary = Join-Path $projectRoot 'R-packages' }
$libraryPath = (Resolve-Path -LiteralPath $RLibrary).Path
if (!$WorkDirectory) {
    $WorkDirectory = Join-Path $projectRoot 'debug_outputs\r_optimization_validation\harness'
}
$scratchPath = [System.IO.Path]::GetFullPath($WorkDirectory)
if (!$scratchPath.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The validation work directory must be inside this checkout.'
}
$inputPath = Join-Path $scratchPath 'work\input'
$outputPath = Join-Path $scratchPath 'work\output'
$runScript = Join-Path $scratchPath 'work\run.R'
$snapshotPath = Join-Path $scratchPath 'snapshots'
$inputNames = @('Light.hdr', 'Light.img', 'Light_MARKED.hdr', 'Light_MARKED.img')
New-Item -ItemType Directory -Force -Path $inputPath, $outputPath, $snapshotPath | Out-Null

function Assert-ScratchFile([string]$Path) {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    if (!$resolvedPath.StartsWith($scratchPath + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a file outside the validation directory: $resolvedPath"
    }
}

function Get-Artifacts {
    foreach ($root in @($outputPath, $inputPath)) {
        foreach ($item in Get-ChildItem -LiteralPath $root -Recurse -File) {
            if ($root -eq $inputPath -and $inputNames -icontains $item.Name) { continue }
            $scope = if ($root -eq $inputPath) { 'input' } else { 'output' }
            $relativePath = $item.FullName.Substring($root.Length + 1)
            [pscustomobject]@{ File = $item.FullName; Relative = (Join-Path $scope $relativePath) }
        }
    }
}

if (!('Step3Validation.FileBytes' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
namespace Step3Validation {
    public static class FileBytes {
        public static long FirstDifference(string left, string right) {
            using (FileStream a = File.OpenRead(left))
            using (FileStream b = File.OpenRead(right)) {
                byte[] x = new byte[1048576], y = new byte[1048576];
                long offset = 0;
                int n;
                while ((n = a.Read(x, 0, x.Length)) > 0) {
                    int m = b.Read(y, 0, n);
                    for (int i = 0; i < Math.Min(n, m); i++) {
                        if (x[i] != y[i]) return offset + i;
                    }
                    if (n != m) return offset + m;
                    offset += n;
                }
                return b.ReadByte() == -1 ? -1 : offset;
            }
        }
    }
}
'@
}

function Compare-FileBytes([string]$Left, [string]$Right) {
    return [Step3Validation.FileBytes]::FirstDifference($Left, $Right)
}

function Run-AndSnapshot([string]$RunLabel, [string]$SourceScript) {
    foreach ($artifact in Get-Artifacts) {
        Assert-ScratchFile $artifact.File
        Remove-Item -LiteralPath $artifact.File -Force
    }
    foreach ($name in $inputNames) {
        Copy-Item -LiteralPath (Join-Path $fixturePath $name) -Destination $inputPath -Force
    }
    Copy-Item -LiteralPath $SourceScript -Destination $runScript -Force
    $env:R_LIBS_USER = $libraryPath
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    $env:AIDAS_STEP3_DIAGNOSTICS = if ($Diagnostics) { '1' } else { '0' }
    $env:AIDAS_STEP3_WORKERS = [string]$Workers
    $logPath = Join-Path $scratchPath ($RunLabel + '.log')
    Write-Host "Running $RunLabel with Rscript $rscriptPath"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    # Nonempty indexes retain positions under Windows PowerShell 5.1, which
    # omits empty-string arguments to native executables.
    $savedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $rscriptPath --vanilla $runScript $inputPath $outputPath `
        DARK_MARKED LIGHT_MARKED DARK LIGHT $ImageIndex $ImageIndex $PixelWidth *> $logPath
    $runExit = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorAction
    $watch.Stop()
    if ($runExit -ne 0) {
        Get-Content -LiteralPath $logPath -Tail 30 | Write-Host
        throw "$RunLabel failed with exit code $runExit. See $logPath"
    }
    $destination = Join-Path $snapshotPath $RunLabel
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($old in Get-ChildItem -LiteralPath $destination -Recurse -File) {
        Assert-ScratchFile $old.FullName
        Remove-Item -LiteralPath $old.FullName -Force
    }
    $manifest = foreach ($artifact in Get-Artifacts) {
        $target = Join-Path $destination $artifact.Relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $artifact.File -Destination $target -Force
        [pscustomobject]@{
            File = $artifact.Relative
            Bytes = (Get-Item -LiteralPath $target).Length
            SHA256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        }
    }
    if (!($manifest | Where-Object { $_.File -like '*_done_*.RData' })) {
        throw "$RunLabel completed without a final .RData workspace."
    }
    $metadata = [pscustomobject]@{
        Script = $SourceScript
        ScriptSHA256 = (Get-FileHash -LiteralPath $runScript -Algorithm SHA256).Hash
        Rscript = $rscriptPath
        RLibrary = $libraryPath
        Fixture = $fixturePath
        ImageIndex = $ImageIndex
        PixelWidth = $PixelWidth
        Diagnostics = [bool]$Diagnostics
        Workers = $Workers
        Seconds = $watch.Elapsed.TotalSeconds
        Artifacts = @($manifest)
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $scratchPath ($RunLabel + '.json')) -Encoding UTF8
    Write-Host ("Saved {0}: {1} artifacts in {2:N2} seconds" -f $RunLabel, @($manifest).Count, $watch.Elapsed.TotalSeconds)
}

function Compare-Snapshots([string]$LeftName, [string]$RightName) {
    $leftRoot = Join-Path $snapshotPath $LeftName
    $rightRoot = Join-Path $snapshotPath $RightName
    if (!(Test-Path -LiteralPath $leftRoot) -or !(Test-Path -LiteralPath $rightRoot)) {
        throw "Missing snapshots for comparison: $LeftName, $RightName"
    }
    $leftFiles = @{}
    $rightFiles = @{}
    foreach ($file in Get-ChildItem -LiteralPath $leftRoot -File -Recurse) {
        $leftFiles[$file.FullName.Substring($leftRoot.Length + 1)] = $file.FullName
    }
    foreach ($file in Get-ChildItem -LiteralPath $rightRoot -File -Recurse) {
        $rightFiles[$file.FullName.Substring($rightRoot.Length + 1)] = $file.FullName
    }
    $results = foreach ($name in @(@($leftFiles.Keys) + @($rightFiles.Keys) | Sort-Object -Unique)) {
        $status = 'identical'
        $offset = -1L
        $pdfTimestampOnly = $false
        if (!$leftFiles.ContainsKey($name)) { $status = 'added' }
        elseif (!$rightFiles.ContainsKey($name)) { $status = 'missing' }
        else {
            $offset = Compare-FileBytes $leftFiles[$name] $rightFiles[$name]
            if ($offset -ge 0) {
                $status = 'different'
                if ([System.IO.Path]::GetExtension($name) -ieq '.pdf') {
                    # This secondary diagnosis never changes the strict byte
                    # result. Latin-1 preserves every byte in binary PDF streams.
                    $encoding = [System.Text.Encoding]::GetEncoding(28591)
                    $leftText = $encoding.GetString([System.IO.File]::ReadAllBytes($leftFiles[$name]))
                    $rightText = $encoding.GetString([System.IO.File]::ReadAllBytes($rightFiles[$name]))
                    $datePattern = '/(CreationDate|ModDate) \(D:\d{14}\)'
                    $leftDatesRemoved = [regex]::Replace($leftText, $datePattern, '/$1 (D:00000000000000)')
                    $rightDatesRemoved = [regex]::Replace($rightText, $datePattern, '/$1 (D:00000000000000)')
                    $pdfTimestampOnly = [string]::Equals($leftDatesRemoved, $rightDatesRemoved,
                        [System.StringComparison]::Ordinal)
                }
            }
        }
        [pscustomobject]@{
            File = $name
            Status = $status
            FirstDifferentByte = $offset
            PDFTimestampOnly = $pdfTimestampOnly
        }
    }
    $results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (
        Join-Path $scratchPath ($LeftName + '-vs-' + $RightName + '.json')) -Encoding UTF8
    $different = @($results | Where-Object { $_.Status -ne 'identical' })
    Write-Host ("{0} vs {1}: {2}/{3} artifacts byte-identical" -f $LeftName, $RightName,
        (@($results).Count - $different.Count), @($results).Count)
    $outputResults = @($results | Where-Object { $_.File.StartsWith('output\') })
    $outputIdentical = @($outputResults | Where-Object { $_.Status -eq 'identical' })
    Write-Host ("OUTPUTDIR: {0}/{1} files byte-identical" -f $outputIdentical.Count, $outputResults.Count)
    $different | Format-Table -AutoSize | Out-Host
    return [pscustomobject]@{ Results = @($results); Different = $different }
}

if ($Phase -in @('all', 'baseline')) { Run-AndSnapshot 'baseline' $baselinePath }
if ($Phase -eq 'repeat' -or ($Phase -eq 'all' -and $RepeatBaseline)) {
    Run-AndSnapshot 'repeat' $baselinePath
}
if ($Phase -in @('all', 'candidate')) { Run-AndSnapshot $CandidateName $candidatePath }
if ($Phase -in @('all', 'compare')) {
    if (Test-Path -LiteralPath (Join-Path $snapshotPath 'repeat')) {
        $repeatComparison = Compare-Snapshots 'baseline' 'repeat'
        if ($repeatComparison.Different.Count) {
            Write-Host 'The original script is nondeterministic for the listed artifacts; this is an observed baseline limitation.'
        }
    }
    $candidateComparison = Compare-Snapshots 'baseline' $CandidateName
    if ($candidateComparison.Different.Count) {
        throw 'Byte-for-byte output parity failed. No artifacts were excluded or normalized.'
    }
}
