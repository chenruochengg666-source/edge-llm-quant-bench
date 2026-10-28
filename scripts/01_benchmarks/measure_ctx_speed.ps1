# measure_ctx_speed.ps1 (v2) —— 测不同上下文长度下的预填充/生成速度（4K~256K）
# v2 修正：兼容 llama-bench 新版 CSV 列名（n_prompt / n_gen / avg_ts / stddev_ts），
#          同时兼容旧版（test / t/s / stddev）；并支持 -ReuseExisting 复用已跑过的 csv。
# 用法：
#   .\measure_ctx_speed.ps1 -Model "C:\...\qwen1.5b-Q4_K_M.gguf" -Out "C:\...\ctx_speed_qwen1.5b.csv"
#   .\measure_ctx_speed.ps1 -Model "..." -Out "..." -Contexts 32768,131072,262144
#   .\measure_ctx_speed.ps1 -Model "..." -Out "..." -ReuseExisting     # 已有 bench_ctx*.csv 就直接解析，不重跑
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$LlamaBench = "C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-bench.exe",
    [int[]]$Contexts = @(4096, 8192, 16384, 32768, 65536, 131072, 262144),
    [int]$Threads = 4,
    [int]$GenTokens = 128,
    [int]$Rep = 1,
    [string]$ExtraArgs = "-b 2048 -ub 512 -fa on -ngl 0",
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Continue"
$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$tmpDir = Join-Path $outDir "ctx_speed_logs"
if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null }

function Parse-BenchCsv([string]$path) {
    $res = @{ prefill = $null; prefill_sd = $null; gen = $null; gen_sd = $null }
    try { $rows = Import-Csv -Path $path } catch { return $res }
    if (-not $rows -or $rows.Count -eq 0) { return $res }
    $cols = $rows[0].PSObject.Properties.Name
    foreach ($r in $rows) {
        if ($cols -contains 'n_prompt') {
            if ([double]$r.n_prompt -gt 0 -and $null -eq $res.prefill) {
                $res.prefill = [math]::Round([double]$r.avg_ts, 2)
                $res.prefill_sd = [math]::Round([double]$r.stddev_ts, 2)
            }
            if ([double]$r.n_gen -gt 0 -and $null -eq $res.gen) {
                $res.gen = [math]::Round([double]$r.avg_ts, 2)
                $res.gen_sd = [math]::Round([double]$r.stddev_ts, 2)
            }
        }
        elseif ($cols -contains 'test') {
            if ($r.test -match '^pp(\d+)$' -and $null -eq $res.prefill) {
                $res.prefill = [math]::Round([double]$r.'t/s', 2)
                $res.prefill_sd = [math]::Round([double]$r.stddev, 2)
            }
            if ($r.test -match '^tg(\d+)$' -and $null -eq $res.gen) {
                $res.gen = [math]::Round([double]$r.'t/s', 2)
                $res.gen_sd = [math]::Round([double]$r.stddev, 2)
            }
        }
    }
    return $res
}

$results = @()
$leaf = Split-Path -Leaf $Model

foreach ($ctx in $Contexts) {
    Write-Host "===== 测 $ctx 的预填充/生成速度 =====" -ForegroundColor Cyan
    $csv = Join-Path $tmpDir ("bench_ctx{0}.csv" -f $ctx)
    $err = Join-Path $tmpDir ("bench_ctx{0}.err.log" -f $ctx)

    $status = "OK"
    $note = ""
    $elapsed = 0

    if ($ReuseExisting -and (Test-Path $csv)) {
        $status = "CACHED"
        Write-Host "  复用已有结果：$csv" -ForegroundColor DarkGray
    } else {
        if (Test-Path $csv) { Remove-Item $csv -Force }
        $argList = "-m `"$Model`" -p $ctx -n $GenTokens -r $Rep -t $Threads $ExtraArgs -o csv"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $LlamaBench -ArgumentList $argList `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $csv -RedirectStandardError $err
        $sw.Stop()
        $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        if ($proc.ExitCode -ne 0 -or -not (Test-Path $csv)) {
            $status = "FAIL"
            if (Test-Path $err) { $note = ((Get-Content $err -Tail 4 -ErrorAction SilentlyContinue) -join " | ") }
        }
    }

    $parsed = @{ prefill = $null; prefill_sd = $null; gen = $null; gen_sd = $null }
    if ($status -ne "FAIL" -and (Test-Path $csv)) {
        $parsed = Parse-BenchCsv $csv
        if ($null -eq $parsed.prefill -and $status -eq "OK") { $status = "NO_PREFILL" }
        if ($null -eq $parsed.gen -and $status -eq "OK") { $status = "NO_GEN" }
    }

    $ob = [PSCustomObject]@{
        Model       = $leaf
        Ctx         = $ctx
        Prefill_tps = $parsed.prefill
        Prefill_sd  = $parsed.prefill_sd
        Gen_tps     = $parsed.gen
        Gen_sd      = $parsed.gen_sd
        Status      = $status
        Elapsed_s   = $elapsed
        Note        = if ($note.Length -gt 180) { $note.Substring(0,180) } else { $note }
    }
    $results += $ob

    $color = "Green"; if ($status -notin @("OK", "CACHED")) { $color = "Red" }
    Write-Host ("  -> {0} | prefill={1} t/s | gen={2} t/s | 耗时 {3}s" -f $status, $parsed.prefill, $parsed.gen, $elapsed) -ForegroundColor $color
}

$results | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
Write-Host "`n完成：$Out" -ForegroundColor Green
Write-Host "明细日志：$tmpDir" -ForegroundColor Gray
