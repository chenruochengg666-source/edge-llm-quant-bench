# run_supplement_tests.ps1 —— 补测：HWiNFO 功耗（2×2 电源模式）与权重重排内存
# 必须在计划 B 结束后运行（与评测同时跑会抢 CPU/内存，功耗读数失真）。
#
# 用法（在仓库根目录下，PowerShell；先设置 $env:LLAMA_CPP_BIN 与 $env:EDGE_LLM_MODEL_DIR）：
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test Power -Label AC_HighPerf
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test Power -Label AC_BestEff
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test Power -Label BAT_HighPerf
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test Power -Label BAT_BestEff
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test Repack      （约 8~10 分钟）
#   powershell -ExecutionPolicy Bypass -File .\scripts\01_benchmarks\run_supplement_tests.ps1 -Test RepackSpeed （约 30 分钟）
#
# 功耗每组流程（约 3.5 分钟）：待机 60 s → 仅解码（-p 0 -n 128，重复 20 次）→ 间隔 15 s → 仅预填充（-p 512 -n 0，重复 20 次）→ 间隔 15 s。
# 每个阶段的起止时间（精确到毫秒）写入 *_markers.csv，analyze_power_hwinfo.py 据此在 HWiNFO 日志中截取对应时段。
param(
    [ValidateSet("Power", "Repack", "RepackSpeed")][string]$Test = "Power",
    [string]$Label = "AC_HighPerf"
)
$ErrorActionPreference = "Continue"   # llama-bench 会往 stderr 打印加载信息，不能当作错误
$REPO  = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BIN   = if ($env:LLAMA_CPP_BIN) { $env:LLAMA_CPP_BIN } else { Join-Path (Split-Path $REPO -Parent) "llama.cpp\build\bin\Release" }
$MODELDIR = if ($env:EDGE_LLM_MODEL_DIR) { $env:EDGE_LLM_MODEL_DIR } else { Join-Path $REPO "models" }
$BENCH = Join-Path $BIN "llama-bench.exe"
$SERVER = Join-Path $BIN "llama-server.exe"
$MODEL = Join-Path $MODELDIR "qwen1.5b-Q4_K_M.gguf"   # 注意：llama-bench 无法读取含中文字符的路径
$CTRL  = Join-Path $MODELDIR "qwen1.5b-Q5_K_M.gguf"   # 对照：不含 Q4_K 张量

function Stamp { (Get-Date).ToString("yyyy-MM-dd HH:mm:ss.fff") }

if ($Test -eq "Power") {
    $OUT = Join-Path $REPO "results\raw\power_hwinfo"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null
    $markers = Join-Path $OUT "$($Label)_markers.csv"
    $err_decode = Join-Path $OUT "$($Label)_decode.err.log"
    $err_prefill = Join-Path $OUT "$($Label)_prefill.err.log"
    Write-Host "===== 功耗补测：$Label =====" -ForegroundColor Cyan
    Write-Host "开始前请确认："
    Write-Host "  1) 电源模式已切换为本组设置（平衡计划 + 最佳性能 / 最佳能效），未开节电模式"
    Write-Host "  2) 插电组：电池已充满；电池组：已拔掉电源"
    Write-Host "  3) 屏幕亮度与上一组相同，已关闭其他占用 CPU 的程序"
    Write-Host "  4) HWiNFO64 已在「仅传感器」模式下开始记录（Logging Start，间隔 1000 ms），文件名：$($Label)_hwinfo.csv"
    Read-Host "全部确认后按回车开始"
    "phase,start,end" | Out-File -Encoding utf8 $markers

    $t0 = Stamp; Write-Host "[$t0] 待机 60 秒（请勿操作电脑）..."
    Start-Sleep -Seconds 60
    "idle,$t0,$(Stamp)" | Out-File -Append -Encoding utf8 $markers

    $t0 = Stamp; Write-Host "[$t0] 仅解码：-p 0 -n 128 -r 20 ..."
    & $BENCH -m $MODEL -p 0 -n 128 -r 20 -t 4 -fa 1 -b 2048 -ub 512 -o csv 2>$err_decode | Out-File -Encoding utf8 (Join-Path $OUT "$($Label)_decode.csv")
    if ($LASTEXITCODE -ne 0) { Write-Host "llama-bench 解码阶段异常退出（代码 $LASTEXITCODE），详见同目录 *.err.log" -ForegroundColor Red; exit 1 }
    "decode,$t0,$(Stamp)" | Out-File -Append -Encoding utf8 $markers
    Start-Sleep -Seconds 15

    $t0 = Stamp; Write-Host "[$t0] 仅预填充：-p 512 -n 0 -r 20 ..."
    & $BENCH -m $MODEL -p 512 -n 0 -r 20 -t 4 -fa 1 -b 2048 -ub 512 -o csv 2>$err_prefill | Out-File -Encoding utf8 (Join-Path $OUT "$($Label)_prefill.csv")
    if ($LASTEXITCODE -ne 0) { Write-Host "llama-bench 预填充阶段异常退出（代码 $LASTEXITCODE），详见同目录 *.err.log" -ForegroundColor Red; exit 1 }
    "prefill,$t0,$(Stamp)" | Out-File -Append -Encoding utf8 $markers
    Start-Sleep -Seconds 15

    Write-Host "完成。请在 HWiNFO 中停止记录，把日志保存为：" -ForegroundColor Green
    Write-Host "  $OUT\$($Label)_hwinfo.csv"
    Write-Host "四组都做完后运行：python scripts/03_analysis/analyze_power_hwinfo.py"
}

if ($Test -eq "Repack") {
    # 内存复测：同一模型分别以默认（CPU 在线重排 Q4_K 权重）与 --no-repack 启动，比较峰值 RSS
    # RSS 协议与 measure_rss.ps1 一致：-c 4096，默认线程，触发一次 32 token 推理，3 s 后读 PeakWorkingSet64
    # 速度部分：qwen1.5b-Q4_K_M 以 -t 4 启动，约 512 token 提示 + 生成 128 token，读取服务端 timings
    $OUT = Join-Path (Join-Path $REPO "results\raw") "rss_repack"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null
    $help = (& $SERVER --help 2>&1 | Out-String)
    if ($help -notmatch "no-repack") {
        Write-Host "当前 llama-server 不支持 --no-repack 参数，无法做该补测。" -ForegroundColor Yellow
        exit 0
    }
    $names = @("qwen0.5b-Q4_K_M", "qwen1.5b-Q3_K_M", "qwen1.5b-Q4_K_M", "qwen1.5b-Q5_K_M", "qwen3b-Q4_K_M")
    $REPS = 3
    Write-Host "===== 内存复测（--no-repack）：$($names.Count) 个模型 × 2 种设置 × $REPS 次，约 8~10 分钟 =====" -ForegroundColor Cyan
    Write-Host "请先关闭浏览器等占用内存的程序；运行期间请勿操作电脑。"
    Read-Host "确认后按回车开始"

    function Start-Llama([string]$modelPath, [bool]$noRepack, [string]$extra) {
        Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 3
        $port = Get-Random -Minimum 8100 -Maximum 9000
        $argList = "-m `"$modelPath`" -c 4096 --port $port $extra"
        if ($noRepack) { $argList += " --no-repack" }
        $proc = Start-Process -FilePath $SERVER -ArgumentList $argList -PassThru -WindowStyle Hidden
        $ready = $false
        for ($i = 0; $i -lt 180; $i++) {
            if ($proc.HasExited) { break }
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -ContentType "application/json" `
                    -Body '{"prompt":"A,B,C","n_predict":32}' -TimeoutSec 60 | Out-Null
                $ready = $true; break
            } catch { Start-Sleep -Seconds 1 }
        }
        return @{ Proc = $proc; Port = $port; Ready = $ready }
    }

    $rows = @()
    foreach ($n in $names) {
        $m = Join-Path $MODELDIR "$n.gguf"
        if (-not (Test-Path $m)) { Write-Host "跳过：找不到 $m" -ForegroundColor Yellow; continue }
        $sizeMiB = [math]::Round((Get-Item $m).Length / 1MB, 1)
        for ($r = 1; $r -le $REPS; $r++) {
            foreach ($nr in @($false, $true)) {
                $s = Start-Llama $m $nr ""
                Start-Sleep -Seconds 3
                $peak = if ($s.Proc.HasExited) { $null } else { [math]::Round((Get-Process -Id $s.Proc.Id).PeakWorkingSet64 / 1MB, 2) }
                Write-Host ("[{0}] {1,-17} 第{2}次 no-repack={3,-5} 峰值 RSS = {4} MiB（文件 {5} MiB）" -f (Stamp), $n, $r, $nr, $peak, $sizeMiB)
                $rows += [pscustomobject]@{ Model = $n; File_MiB = $sizeMiB; NoRepack = $nr; Rep = $r; Ctx = 4096; Peak_RSS_MiB = $peak; Ready = $s.Ready }
                if (-not $s.Proc.HasExited) { Stop-Process -Id $s.Proc.Id -Force }
            }
        }
        $rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OUT "rss_repack.csv")
    }

    # 速度：重排的代价 / 收益
    $prompt = ("The quick brown fox jumps over the lazy dog while the farmer counts his sheep. " * 32)
    $speed = @()
    $m = Join-Path $MODELDIR "qwen1.5b-Q4_K_M.gguf"
    foreach ($nr in @($false, $true)) {
        $s = Start-Llama $m $nr "-t 4"
        for ($r = 1; $r -le 4; $r++) {
            $body = @{ prompt = $prompt; n_predict = 128; temperature = 0; ignore_eos = $true; cache_prompt = $false } | ConvertTo-Json
            try {
                $res = Invoke-RestMethod -Uri "http://127.0.0.1:$($s.Port)/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
                $t = $res.timings
                if ($r -gt 1) {   # 第 1 次作为预热，不计入
                    Write-Host ("[{0}] 速度 no-repack={1,-5} 第{2}次  预填充 {3:N1} t/s（{4} token）  生成 {5:N2} t/s（{6} token）" -f (Stamp), $nr, ($r - 1), $t.prompt_per_second, $t.prompt_n, $t.predicted_per_second, $t.predicted_n)
                    $speed += [pscustomobject]@{ NoRepack = $nr; Rep = $r - 1; Prompt_n = $t.prompt_n; PP_tps = $t.prompt_per_second; Gen_n = $t.predicted_n; TG_tps = $t.predicted_per_second }
                }
            } catch { Write-Host "速度测试请求失败：$($_.Exception.Message)" -ForegroundColor Red }
        }
        if (-not $s.Proc.HasExited) { Stop-Process -Id $s.Proc.Id -Force }
    }
    $speed | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OUT "speed_repack.csv")
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "已保存 $OUT\rss_repack.csv 与 speed_repack.csv" -ForegroundColor Green
}

if ($Test -eq "RepackSpeed") {
    # 重排速度补测：默认 vs --no-repack 交替运行（ABBA 顺序抵消升温/时间漂移），每轮之间冷却
    # A 部分：llama-bench（与论文表 1 同口径：-p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0），若支持 -nr 参数
    # B 部分：llama-server（与第一次复测同口径：513 token 提示 + 128 token 生成，每次启动先预热 1 次再测 3 次）
    $OUT = Join-Path (Join-Path $REPO "results\raw") "rss_repack"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null
    $m = Join-Path $MODELDIR "qwen1.5b-Q4_K_M.gguf"
    $ROUNDS = 6; $COOL = 30
    Write-Host "===== 重排速度交替补测：qwen1.5b-Q4_K_M，$ROUNDS 轮 × 2 种设置，约 30 分钟 =====" -ForegroundColor Cyan
    Write-Host "请保持插电 + 最佳性能模式，关闭浏览器等程序；运行期间请勿操作电脑。"
    Read-Host "确认后按回车开始"

    # ---------- A：llama-bench ----------
    $bhelp = (& $BENCH --help 2>&1 | Out-String)
    if ($bhelp -match "no-repack") {
        $benchCsv = Join-Path $OUT "speed_repack_bench.csv"
        $benchErr = Join-Path $OUT "speed_repack_bench.err.log"
        if (Test-Path $benchCsv) { Remove-Item $benchCsv }
        for ($r = 1; $r -le $ROUNDS; $r++) {
            $order = if ($r % 2 -eq 1) { @(0, 1) } else { @(1, 0) }
            foreach ($nr in $order) {
                Write-Host ("[{0}] llama-bench 第{1}轮 no-repack={2} ..." -f (Stamp), $r, $nr)
                $o = & $BENCH -m $m -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0 -nr $nr -o csv 2>> $benchErr
                if ($LASTEXITCODE -ne 0) { Write-Host "llama-bench 异常退出（代码 $LASTEXITCODE），详见 $benchErr" -ForegroundColor Red; break }
                $o | ForEach-Object { "$r,$_" } | Add-Content -Encoding UTF8 $benchCsv
                Start-Sleep -Seconds $COOL
            }
        }
        Write-Host "llama-bench 部分完成：$benchCsv" -ForegroundColor Green
    } else {
        Write-Host "当前 llama-bench 不支持 -nr/--no-repack，跳过 A 部分，只做 llama-server 部分。" -ForegroundColor Yellow
    }

    # ---------- B：llama-server ----------
    $prompt = ("The quick brown fox jumps over the lazy dog while the farmer counts his sheep. " * 32)
    $speed = @()
    for ($r = 1; $r -le $ROUNDS; $r++) {
        $order = if ($r % 2 -eq 1) { @($false, $true) } else { @($true, $false) }
        foreach ($nr in $order) {
            Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
            Start-Sleep -Seconds 3
            $port = Get-Random -Minimum 8100 -Maximum 9000
            $argList = "-m `"$m`" -c 4096 -t 4 --port $port"
            if ($nr) { $argList += " --no-repack" }
            $proc = Start-Process -FilePath $SERVER -ArgumentList $argList -PassThru -WindowStyle Hidden
            $ready = $false
            for ($i = 0; $i -lt 180; $i++) {
                if ($proc.HasExited) { break }
                try {
                    Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -ContentType "application/json" `
                        -Body '{"prompt":"A,B,C","n_predict":8}' -TimeoutSec 60 | Out-Null
                    $ready = $true; break
                } catch { Start-Sleep -Seconds 1 }
            }
            if (-not $ready) { Write-Host "服务未就绪，跳过本次" -ForegroundColor Red; continue }
            for ($k = 0; $k -le 3; $k++) {
                $body = @{ prompt = $prompt; n_predict = 128; temperature = 0; ignore_eos = $true; cache_prompt = $false } | ConvertTo-Json
                try {
                    $res = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
                    $t = $res.timings
                    if ($k -gt 0) {   # 第 0 次为预热，不计入
                        Write-Host ("[{0}] 第{1}轮 no-repack={2,-5} 第{3}次  预填充 {4:N1} t/s  生成 {5:N2} t/s" -f (Stamp), $r, $nr, $k, $t.prompt_per_second, $t.predicted_per_second)
                        $speed += [pscustomobject]@{ Round = $r; NoRepack = $nr; Rep = $k; Time = (Stamp); Prompt_n = $t.prompt_n; PP_tps = $t.prompt_per_second; Gen_n = $t.predicted_n; TG_tps = $t.predicted_per_second }
                    }
                } catch { Write-Host "请求失败：$($_.Exception.Message)" -ForegroundColor Red }
            }
            if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
            $speed | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OUT "speed_repack_interleaved.csv")
            Start-Sleep -Seconds $COOL
        }
    }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "已保存 $OUT\speed_repack_interleaved.csv" -ForegroundColor Green
}
