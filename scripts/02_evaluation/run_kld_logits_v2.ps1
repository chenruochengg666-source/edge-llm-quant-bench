# C:\Users\cjy13\quant_eval\04_scripts\run_kld_logits_v2.ps1
$LLAMA = "C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-perplexity.exe"
$MODEL_DIR = "C:\Users\cjy13\quant_eval\02_gguf\quantized"
$WIKI = "C:\Users\cjy13\quant_eval\03_data\wikitext2\wiki.test.raw"
$KLD_BASE_DIR = "C:\Users\cjy13\quant_eval\05_raw\kld_base"
$KLD_LOG_DIR = "C:\Users\cjy13\quant_eval\05_raw\kld_logs"

$ErrorActionPreference = "Continue"

# 创建必要的目录
if (-not (Test-Path $KLD_BASE_DIR)) { New-Item -ItemType Directory -Force -Path $KLD_BASE_DIR | Out-Null }
if (-not (Test-Path $KLD_LOG_DIR)) { New-Item -ItemType Directory -Force -Path $KLD_LOG_DIR | Out-Null }

# 定义所有模型及其对应的量化档位
$models = @{
    "qwen1.5b" = @("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
    "qwen3b"   = @("Q4_K_M", "Q3_K_M", "Q2_K")
    "qwen0.5b" = @("Q4_K_M", "Q2_K")
    "phi3"     = @("Q4_K_M", "Q2_K")
    "ll32"     = @("Q4_K_M", "Q2_K")
}

foreach ($model in $models.Keys) {
    # ============ 1. 生成 F16 基线 KLD 文件 ============
    $base_model = Join-Path $MODEL_DIR "$model-base.gguf"
    $base_kld = Join-Path $KLD_BASE_DIR "$model-base.kld"

    if (-not (Test-Path $base_model)) {
        Write-Host "[WARN] Skip model $model : Base model not found at $base_model" -ForegroundColor Yellow
        continue
    }

    if (-not (Test-Path $base_kld)) {
        Write-Host "[INFO] Generating F16 baseline KLD for $model ..." -ForegroundColor Yellow
        & $LLAMA -m $base_model -f $WIKI -c 512 --chunks 20 --kl-divergence-base $base_kld
    } else {
        Write-Host "[INFO] Baseline KLD already exists: $base_kld" -ForegroundColor Green
    }

    # ============ 2. 计算量化模型相对基线的 KLD ============
    foreach ($quant in $models[$model]) {
        $quant_model = Join-Path $MODEL_DIR "$model-$quant.gguf"
        $log_file = Join-Path $KLD_LOG_DIR "$model-$quant-kld.txt"

        if (-not (Test-Path $quant_model)) {
            # 使用 -f 格式化字符串，避免 $model-$quant 引起的解析错误
            $msg = "[WARN] Skip {0}-{1} : File not found" -f $model, $quant
            Write-Host $msg -ForegroundColor Yellow
            continue
        }

        $msg = "[INFO] Calculating KL divergence for {0}-{1} ..." -f $model, $quant
        Write-Host $msg -ForegroundColor Cyan
        
        # 执行计算，并保存日志
        # 使用 cmd /c 确保 stdout 和 stderr 全部写入日志文件
cmd /c "`"$LLAMA`" -m `"$quant_model`" -f `"$WIKI`" -c 512 --chunks 20 --kl-divergence --kl-divergence-base `"$base_kld`" > `"$log_file`" 2>&1"
    }
}

Write-Host "`n[DONE] All KLD calculations finished! Logs saved to: $KLD_LOG_DIR" -ForegroundColor Green