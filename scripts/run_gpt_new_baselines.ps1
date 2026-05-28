param(
    [string[]]$Datasets = @("hotpotqa", "musique", "nq", "triviaqa"),
    [int]$Size = 1000,
    [string]$Model = "gpt-4.1-mini-2025-04-14",
    [string]$OutputDir = "results/gpt_new_baselines",
    [string]$RunSuffix = "newbaselines"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $name, $value = $line.Split("=", 2)
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set. Add it to .env or the current environment."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$retrievalCacheDir = Join-Path $OutputDir "cache_shared"
$openaiCachePath = Join-Path $OutputDir "openai_cache_shared.jsonl"
New-Item -ItemType Directory -Force -Path $retrievalCacheDir | Out-Null

foreach ($dataset in $Datasets) {
    $expandedK = if ($dataset -in @("hotpotqa", "musique")) { 8 } else { 5 }
    $runName = "gpt-newbaselines-$dataset-s$Size-$RunSuffix"
    $logPath = Join-Path $OutputDir "$dataset`_s$Size`_$RunSuffix.log"

    Write-Host "[RUN] dataset=$dataset size=$Size model=$Model expanded_k=$expandedK run=$runName"
    & $python "src\run_experiments.py" `
        --mode $dataset `
        --sizes $Size `
        --doc-limit 20000 `
        --corpus-split validation `
        --query-split validation `
        --initial-k 3 `
        --expanded-k $expandedK `
        --candidate-pool-k 8 `
        --tail-level 0.5 `
        --sufficiency-tolerance 0.02 `
        --utility-rho 0.1 `
        --utility-alpha 0.0 `
        --utility-beta 0.0 `
        --stability-rho-grid 0.1 `
        --stability-alpha-grid 0.0 `
        --stability-beta-grid 0.0 `
        --stability-tail-grid 0.5 `
        --label-strategy evidence `
        --use-openai `
        --openai-model $Model `
        --run-stability-selection `
        --use-run-subdir `
        --run-name $runName `
        --output-dir $OutputDir `
        --retrieval-cache-dir $retrievalCacheDir `
        --openai-cache-path $openaiCachePath 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for dataset=$dataset with exit code $LASTEXITCODE"
    }
}

Write-Host "[DONE] GPT new-baseline runs complete."
