# C:\Users\cjy13\quant_eval\04_scripts\measure_rss.ps1
param(
    [string]$Model,
    [int]$Ctx = 4096,
    [string]$Out
)

$llama_server = "C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
$port = 8080

# 1. 确保输出目录存在（防止路径不存在导致静默失败）
$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

# 清理残留进程
taskkill /F /IM llama-server.exe 2>$null
Start-Sleep -Seconds 3

Write-Host "正在启动模型: $(Split-Path -Leaf $Model)" -ForegroundColor Cyan
# 启动服务
$proc = Start-Process -FilePath $llama_server -ArgumentList "-m $Model -c $Ctx --port $port" -PassThru -WindowStyle Hidden

# 2. 增加等待时间（8B模型加载慢，给足 20 秒）
Start-Sleep -Seconds 20

if ($proc.HasExited) {
    Write-Host "❌ 错误: 模型加载失败或端口被占用，进程已退出。" -ForegroundColor Red
    exit
}

# 触发一次推理，确保 KV Cache 和激活内存达到峰值
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body '{"prompt":"A,B,C","n_predict":32}' -ContentType "application/json" | Out-Null
    Write-Host "  [触发推理成功]" -ForegroundColor Gray
} catch {
    Write-Host "  ⚠️ 触发推理失败，但继续读取内存..." -ForegroundColor Yellow
}
Start-Sleep -Seconds 3

# 读取峰值内存 (PeakWorkingSet64)
$peak_rss_bytes = (Get-Process -Id $proc.Id).PeakWorkingSet64
$peak_rss_mib = [math]::Round($peak_rss_bytes / 1MB, 2)

# 停止服务
Stop-Process -Id $proc.Id -Force
taskkill /F /IM llama-server.exe 2>$null

# 导出 CSV
$result = [PSCustomObject]@{
    Model = (Get-Item $Model).BaseName
    Ctx = $Ctx
    Peak_RSS_MiB = $peak_rss_mib
}
$result | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
Write-Host "✅ 导出成功: $Out (峰值 RSS: $peak_rss_mib MiB)" -ForegroundColor Green