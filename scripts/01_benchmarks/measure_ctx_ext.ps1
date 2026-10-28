# measure_ctx_ext.ps1  —— 扩展版：支持 16K / 32K 上下文，含 OOM 判定
# 用法示例：
#   .\measure_ctx_ext.ps1 -Model "C:\...\qwen1.5b-Q4_K_M.gguf" -Out "C:\...\ctx_qwen1.5b-Q4_K_M_ext.csv"
#   .\measure_ctx_ext.ps1 -Model "..." -Out "..." -Contexts 16384,32768      # 只测大上下文
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$LlamaServer = "C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe",
    [int[]]$Contexts = @(512, 1024, 2048, 4096, 8192, 16384, 32768,65536,131072,262144),
    [string]$ExtraArgs = "-fa on -ngl 0 -t 4",   # 与主实验保持一致
    [int]$Port = 8080,
    [string]$LogDir = ""
)

$ErrorActionPreference = "Continue"
if (-not $LogDir) { $LogDir = Join-Path (Split-Path -Parent $Out) "ctx_logs" }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

function Get-AvailMemMiB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1024, 0)
}

function Get-CommitLimitMiB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1024, 0)
}

$results = @()
$modelLeaf = (Split-Path -Leaf $Model)

foreach ($ctx in $Contexts) {
    Write-Host "===== 测试上下文 $ctx ($modelLeaf) =====" -ForegroundColor Cyan
    taskkill /F /IM llama-server.exe 2>$null | Out-Null
    Start-Sleep -Seconds 3

    $availBefore = Get-AvailMemMiB
    $stdoutLog = Join-Path $LogDir ("ctx{0}_stdout.log" -f $ctx)
    $stderrLog = Join-Path $LogDir ("ctx{0}_stderr.log" -f $ctx)

    $argList = "-m `"$Model`" -c $ctx --port $Port $ExtraArgs"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $LlamaServer -ArgumentList $argList `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    # 自适应等待：16K/32K 加载更慢
    $maxWait = 30
    if ($ctx -ge 32768) { $maxWait = 180 }
    elseif ($ctx -ge 16384) { $maxWait = 120 }
    elseif ($ctx -ge 8192) { $maxWait = 60 }

    $ready = $false
    $status = ""
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 2
        $waited += 2
        if ($proc.HasExited) { $status = "LOAD_FAIL"; break }
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 | Out-Null
            $ready = $true; break
        } catch {
            # 老版本 llama-server 无 /health 时，用 /completion 探测
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
                    -Body '{"prompt":"A,B,C","n_predict":8}' -ContentType "application/json" -TimeoutSec 5 | Out-Null
                $ready = $true; break
            } catch {}
        }
    }
    $sw.Stop()

    $peak = $null
    $working = $null
    $oomNote = ""
    if ($ready -and -not $proc.HasExited) {
        # 触发一次推理，使激活内存到位（不填满上下文，避免额外变量）
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
                -Body '{"prompt":"A,B,C","n_predict":32}' -ContentType "application/json" -TimeoutSec 60 | Out-Null
        } catch { $oomNote = "inference_failed" }
        Start-Sleep -Seconds 3
        try {
            $pobj = Get-Process -Id $proc.Id
            $peak = [math]::Round($pobj.PeakWorkingSet64 / 1MB, 2)
            $working = [math]::Round($pobj.WorkingSet64 / 1MB, 2)
            $status = "OK"
        } catch {
            $status = "PROC_GONE"
        }
    } elseif ($status -ne "LOAD_FAIL") {
        $status = "TIMEOUT"
    }

    # 读取日志尾部用于定位失败原因
    $tail = ""
    if ($status -ne "OK") {
        if (Test-Path $stderrLog) { $tail = ((Get-Content $stderrLog -Tail 8 -ErrorAction SilentlyContinue) -join " | ") }
        if (-not $tail -and (Test-Path $stdoutLog)) { $tail = ((Get-Content $stdoutLog -Tail 8 -ErrorAction SilentlyContinue) -join " | ") }
        $oomNote = "$oomNote $tail".Trim()
    }

    # 关闭
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    taskkill /F /IM llama-server.exe 2>$null | Out-Null
    Start-Sleep -Seconds 3

    $ob = [PSCustomObject]@{
        Model      = $modelLeaf
        Ctx        = $ctx
        Status     = $status
        Peak_RSS_MiB = $peak
        WS_MiB     = $working
        Load_s     = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        AvailBefore_MiB = $availBefore
        Note       = if ($oomNote.Length -gt 200) { $oomNote.Substring(0,200) } else { $oomNote }
    }
    $results += $ob
    $color = "Green"; if ($status -ne "OK") { $color = "Red" }
    Write-Host ("  -> {0} | Peak={1} MiB | Load={2}s" -f $status, $peak, $ob.Load_s) -ForegroundColor $color
}

$results | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
Write-Host "完成，结果: $Out" -ForegroundColor Green
Write-Host "日志目录: $LogDir" -ForegroundColor Gray
