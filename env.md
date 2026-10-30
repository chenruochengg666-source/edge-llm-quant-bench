# 实验环境与可复现参数

## 1. 硬件与系统环境
- **设备型号**: ASUS Vivobook 14
- **CPU**: Intel Core Ultra 5 225H（14 线程，异构架构，支持 AVX2、AVX-VNNI，不支持 AVX-512）
- **内存**: 16 GB DDR5
- **显卡**: 无独立显卡（纯 CPU 推理）
- **操作系统**: Windows 11
- **Python 版本**: 3.10（pandas / numpy / scipy / statsmodels / matplotlib）
- **实验前准备**: 页面文件（虚拟内存）扩容至 16~32 GiB；测量前关闭全部非实验程序，确保 CPU 与内存资源专用。

## 2. 推理框架版本
- **llama.cpp commit**: `4da633776`（build 11223，AVX2 编译版）
- *注：跨架构数据由本轮重测生成；早期版本（0.4.1-dev）数据已废弃。*

## 3. 受测模型与词表规模
- **Qwen2.5 系列**: 0.5B、1.5B、3B、7B（词表 151936）
- **Phi-3-mini**: 4k-instruct（词表 32064）
- **Llama 系列**: Llama-3.2-1B-Instruct（词表 128256）、Llama-3.1-8B-Instruct
- *注意：跨词表 KLD 不可比，跨架构比较主要使用 ARC-Easy 准确率。*

## 4. 控制变量与参数设定
- **线程数**: `-t 4`（解码阶段较优；预填充随线程数持续上升）；批量下游推理与 PPL/KLD 用 `-t 8`
- **批大小**: `-b 2048 -ub 512`
- **FlashAttention**: `-fa on`
- **GPU 卸载层数**: `-ngl 0`
- **KV Cache 类型**: `-ctk f16 -ctv f16`
- **下游评测**: `temperature=0`、`top_p=1`、`seed=42`（推理 `max_tokens=1024`，下游批量 `256`）
- **电源模式**: 插电 + 高性能（电源对照实验除外）
- **上下文伸缩**: `-c` 取 512、1024、2048、4096、8192、16384、32768、65536、131072、262144；每次以 `-c` 启动后发送 32 token 触发 KV 分配，`n=1`
- **上下文速度**: `llama-bench -p 4096~131072 -n 128 -r 1`（单次测量）

## 5. 统计判定规则
- **ARC-Easy（比例指标）**：两比例 z 检验，效应量为 Cohen's h。
- **KLD / PPL（连续指标）**：Welch t 检验（`chunks=20`，即 n=20），效应量为 Cohen's d。
- **多重比较**：p 值经 Benjamini-Hochberg（BH-FDR）校正，报告 q 值。
- **实质差异判定**：**q < 0.05 且 |效应量| ≥ 0.2** 方视为实质差异；仅统计显著而效应量低于 0.2 者记为「统计显著但效应量小」，不计为实质差异。
- **保守校核**：对独立测量量按 σ_pool = √(σ₁² + σ₂²) 合成，│x̄₁ − x̄₂│ ≥ 2σ_pool 作为 2σ 保守校核（非主判据）。
- 主结果表：`results/summary/table5_stat_final.csv`；脚本：`scripts/03_analysis/stat_tests_v2.py`

## 6. 核心实验命令

### 量化命令（FP16 GGUF 生成各档位，未使用 imatrix）

```bash
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q8_0.gguf  Q8_0
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q6_K.gguf  Q6_K
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q5_K_M.gguf Q5_K_M
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q4_K_M.gguf Q4_K_M
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q2_K.gguf  Q2_K
```

### PPL / KLD 命令

```bash
# FP16 生成基准 logits
llama-perplexity.exe -m qwen1.5b-f16.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base qwen1.5b-f16.kld

# 各量化档位
llama-perplexity.exe -m qwen1.5b-Q4_K_M.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base qwen1.5b-f16.kld
```

### 性能评测命令

```bash
llama-bench -m model.gguf -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0
```

### 上下文伸缩与速度命令

```powershell
# 峰值 RSS（512~256K，含 OOM 判定）
.\scripts\01_benchmarks\measure_ctx_ext.ps1 -Model .\model-Q4_K_M.gguf -Out ..\results\raw\ctx\ctx_ext.csv

# 预填充/生成速度（4K~128K）
.\scripts\01_benchmarks\measure_ctx_speed.ps1 -Model .\model-Q4_K_M.gguf -Out ..\results\raw\ctx\ctx_speed.csv
```

### 统计检验

```bash
python scripts/03_analysis/stat_tests_v2.py   # 两比例 z + Welch t + BH-FDR + Cohen's h/d
```
