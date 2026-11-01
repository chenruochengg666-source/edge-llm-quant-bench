# C:\Users\cjy13\quant_eval\04_scripts\measure_ctx.ps1
param(
    [string]$Model,
    [string]$Out,
    [string]$LlamaServer = "C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
)

$contexts = @(512, 1024, 2048, 4096, 8192)
$results = @()

foreach ($ctx in $contexts) {
    Write-Host "正在测上下文 $ctx ..." -ForegroundColor Cyan
    
    # 清理残留进程
    taskkill /F /IM llama-server.exe 2>$null
    Start-Sleep -Seconds 3
    
    # 启动服务
    $proc = Start-Process -FilePath $LlamaServer `
        -ArgumentList "-m $Model -c $ctx --port 8080" `
        -PassThru -WindowStyle Hidden
    
    Start-Sleep -Seconds 15  # 等待模型和 KV Cache 分配

    # 触发一次推理，确保 KV Cache 真正建立
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:8080/completion" `
            -Method Post `
            -Body '{"prompt":"A,B,C","n_predict":32}' `
            -ContentType "application/json" | Out-Null
    } catch {}
    
    Start-Sleep -Seconds 3
    
    # 读取峰值 RSS
    $peak_rss_mib = [math]::Round((Get-Process -Id $proc.Id).PeakWorkingSet64 / 1MB, 2)
    
    # 关闭服务
    Stop-Process -Id $proc.Id -Force
    taskkill /F /IM llama-server.exe 2>$null
    Start-Sleep -Seconds 3
    
    $results += [PSCustomObject]@{
        Ctx = $ctx
        Peak_RSS_MiB = $peak_rss_mib
    }
    Write-Host "✅ 上下文 $ctx 峰值 RSS: $peak_rss_miB MiB" -ForegroundColor Green
}

# 导出
$results | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
Write-Host "✅ 全部完成，结果已保存至: $Out" -ForegroundColor Green
