# 实验环境与可复现参数

## 1. 硬件与系统环境
- **设备型号**: ASUS Vivobook 14
- **CPU**: Intel Core Ultra 5 225H (14线程，异构架构，支持 AVX2、AVX-VNNI，不支持 AVX-512)
- **内存**: 16 GB DDR5
- **显卡**: 无独立显卡（纯 CPU 推理）
- **操作系统**: Windows 11
- **Python 版本**: 3.10
- **实验前准备**: 页面文件（虚拟内存）扩容至 16~32 GiB；测量前关闭全部非实验程序，确保 CPU 与内存资源专用。

## 2. 推理框架版本
- **llama.cpp commit**: `4da633776` (build 11223, AVX2 编译版)
- *注：跨架构数据由本轮重测生成；早期版本 (0.4.1-dev) 数据已废弃。*

## 3. 受测模型与词表规模
- **Qwen2.5 系列**: 0.5B, 1.5B, 3B, 7B (词表: 151936)
- **Phi-3-mini**: 4k-instruct (词表: 32064)
- **Llama 系列**: Llama-3.2-1B-Instruct (词表: 128256), Llama-3.1-8B-Instruct
- *注意：跨词表 KLD 不可比，跨架构比较主要使用 ARC-Easy 准确率。*

## 4. 控制变量与参数设定
- **线程数**: `-t 4` (解码阶段较优；预填充随线程数持续上升)
- **批大小**: `-b 2048 -ub 512`
- **FlashAttention**: `-fa on`
- **GPU 卸载层数**: `-ngl 0`
- **KV Cache 类型**: `-ctk f16 -ctv f16`
- **下游评测**: `temperature=0`, `top_p=1`, `seed=42`
- **电源模式**: 插电+高性能（除电源对照实验外）

## 5. 统计判定规则
- 两个独立测量量按 \(\sigma_{pool} = \sqrt{\sigma_1^2 + \sigma_2^2}\) 合成标准差。当 \(|\bar{x}_1 - \bar{x}_2| \ge 2\sigma_{pool}\) 时判定为显著差异。
- ARC-Easy 准确率采用两比例 z 检验，KLD 采用 Welch t 检验。
- p 值经 BH-FDR 校正，q < 0.05 视为显著。

## 6. 核心实验命令

### 量化命令 (FP16 GGUF 生成各档位，未使用 imatrix)

llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q8_0.gguf Q8_0
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q6_K.gguf Q6_K
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q5_K_M.gguf Q5_K_M
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q4KM.gguf Q4_K_M
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q2K.gguf Q2_K

PPL/KLD 命令

llama-perplexity.exe -m <model>.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
-kl-divergence-base <f16-base>.kld -kl-divergence

性能评测命令

llama-bench -m model.gguf -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0

