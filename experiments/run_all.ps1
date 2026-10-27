<#
.SYNOPSIS
  Re-run every experiment of the paper into a SEPARATE results folder (default: ecd\results_rerun).
  ecd\results - the folder the paper was built from - is never written to.

.DESCRIPTION
  Sets ECD_RESULTS_DIR (and optionally ECD_FRESH_CACHE) and runs the experiment scripts in the
  order the paper used them.  Every step is logged to <results>\logs\<step>.log and summarised
  in <results>\RUN_INFO.md (start time, duration, exit code).  A failing step stops the run; re-run
  with -Steps to continue from it (trained models are reused from <results>\models).

.PARAMETER ResultsDir   Target folder, absolute or relative to ecd\ (default results_rerun).
.PARAMETER FreshCache   Also recompute the Z3 ground truth and the overlap labels (slower, fully
                        independent of the copies cached next to the datasets).
.PARAMETER Steps        Subset / order of steps.  Known steps:
                        rq1 train_complex rq2 rq2_complex rq3 rq5 real rq4 summarize export figures plots

.EXAMPLE
  .\experiments\run_all.ps1                               # everything -> ecd\results_rerun
  .\experiments\run_all.ps1 -FreshCache                   # also recompute truth + labels
  .\experiments\run_all.ps1 -Steps rq2,rq2_complex        # just RQ2 (needs models from rq1/train_complex)
  .\experiments\run_all.ps1 -ResultsDir D:\runs\ecd_2026-10-28
#>
param(
    [string]$ResultsDir = "results_rerun",
    [switch]$FreshCache,
    [string[]]$Steps = @("rq1", "train_complex", "rq2", "rq2_complex", "rq3", "rq5", "real", "rq4",
                         "summarize", "export", "tables", "figures", "plots")
)

$Steps = @($Steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # "-Steps rq2,rq3" also works via -File / run_all.cmd
$Root = Split-Path -Parent $PSScriptRoot          # ecd\
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
if ([System.IO.Path]::IsPathRooted($ResultsDir)) { $Res = $ResultsDir } else { $Res = Join-Path $Root $ResultsDir }
$Logs = Join-Path $Res "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$env:ECD_RESULTS_DIR = $Res
if ($FreshCache) { $env:ECD_FRESH_CACHE = "1" } else { $env:ECD_FRESH_CACHE = "" }
# stdout goes through a pipe into the log: force UTF-8 so non-ASCII progress output cannot crash a step
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# step -> python script + arguments (paths relative to ecd\)
$Plan = [ordered]@{
    rq1           = @("experiments\run_rq1_embedding.py")                      # labels + GNN for 6 datasets, SBERT/hash baselines
    train_complex = @("experiments\train_model.py", "data/processed/complex_1k") # GNN for complex_1k (used by rq2_complex)
    rq2           = @("experiments\run_rq2_pruning.py")                        # synthetic_1k
    rq2_complex   = @("experiments\run_rq2_pruning.py", "rq2_complex")         # complex_1k
    rq3           = @("experiments\run_rq3_incremental.py")                    # Event-B contexts -> <results>\eventb\rq3
    rq5           = @("experiments\run_rq5_hetero.py")                         # hetero_1k_h00/h25/h50
    real          = @("experiments\run_real.py")                               # 11 real policies
    rq4           = @("experiments\run_rq4_scalability.py")                    # 1k..100k rules x 30 seeds (longest step)
    summarize     = @("experiments\summarize.py")                              # <results>\SUMMARY.md
    export        = @("experiments\export_numbers.py", (Join-Path $Res "numbers.tex"))
    tables        = @("experiments\export_tables.py", (Join-Path $Res "paper_tables"))
    figures       = @("experiments\paper_figures.py", (Join-Path $Res "paper_figures"))
    plots         = @("experiments\plot_results.py")                           # <results>\figures\*.png
}
foreach ($s in $Steps) {
    if (-not $Plan.Contains($s)) { throw "unknown step '$s'. Known steps: $($Plan.Keys -join ', ')" }
}

$Info = Join-Path $Res "RUN_INFO.md"
$started = Get-Date
$pyinfo = & $Py -c "import sys, platform; print('python', sys.version.split()[0], platform.platform())"
# package versions from metadata (no heavy imports)
$pkg = & $Py -c "from importlib.metadata import version, PackageNotFoundError
out = []
for p in ['numpy', 'scipy', 'torch', 'z3-solver', 'sentence-transformers', 'faiss-cpu']:
    try: out.append(p + ' ' + version(p))
    except PackageNotFoundError: pass
print(', '.join(out))"
$cores = [Environment]::ProcessorCount
Add-Content -Path $Info -Encoding UTF8 -Value @(
    "# ECD experiment re-run",
    "",
    "- started: $($started.ToString('yyyy-MM-dd HH:mm:ss'))",
    "- results dir: $Res",
    "- fresh cache (truth + labels recomputed): $($FreshCache.IsPresent)",
    "- machine: $env:COMPUTERNAME, $cores logical cores, $env:PROCESSOR_IDENTIFIER",
    "- $pyinfo",
    "- packages: $pkg",
    "- steps: $($Steps -join ', ')",
    "",
    "| step | started | duration | exit code |",
    "|---|---|---|---|")

foreach ($s in $Steps) {
    $argv = $Plan[$s]
    $log = Join-Path $Logs "$s.log"
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=== [$($t0.ToString('HH:mm:ss'))] $s : python $($argv -join ' ')" -ForegroundColor Cyan
    Write-Host "    log: $log"
    # cmd merges stderr into stdout (tracebacks land in the log); -u keeps the log live
    cmd /c $Py -u @argv "2>&1" | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $dur = (Get-Date) - $t0
    Add-Content -Path $Info -Encoding UTF8 -Value ("| {0} | {1} | {2} | {3} |" -f $s, $t0.ToString("HH:mm:ss"), $dur.ToString("hh\:mm\:ss"), $code)
    if ($code -ne 0) {
        Write-Host "step '$s' failed with exit code $code - see $log" -ForegroundColor Red
        Add-Content -Path $Info -Encoding UTF8 -Value "", "- aborted at step '$s' ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
        exit $code
    }
}
$total = (Get-Date) - $started
Add-Content -Path $Info -Encoding UTF8 -Value "", "- finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (total $($total.ToString('hh\:mm\:ss')))"
Write-Host ""
Write-Host "done in $($total.ToString('hh\:mm\:ss')) -> $Res" -ForegroundColor Green
Write-Host "compare with the paper's results:  python experiments\compare_results.py results $ResultsDir"
