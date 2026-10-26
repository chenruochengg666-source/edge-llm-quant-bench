
---
# 数据集说明 (Data Directory)

本目录（`03_data/`）用于存放评测所需的语料、题库及样本。本仓库不包含大体积原始数据集，请根据以下说明前往官方渠道下载或通过脚本生成。

## 1. 公开评测数据集 (Public Datasets)

- **WikiText-2**: 用于计算困惑度 (PPL) 与 KL 散度 (KLD)。
  - **文件路径**: `03_data/wikitext2/wiki.test.raw`
  - **来源**: [HuggingFace - Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (配置：`wikitext-2-raw-v1`，test split)。
- **ARC-Easy**: 用于常识推理能力测试（全量 2376 题）。
  - **文件路径**: `03_data/arc/ARC-Easy-test-full.jsonl`
  - **来源**: [HuggingFace - allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)。
  - *评测说明*：通过本地 `llama-server` 的 API 调用，`temperature=0`，`max_tokens=16`，模型输出首 token 与标准答案匹配记 1 分。参考 `lm-evaluation-harness` 逻辑。

## 2. 混合多步推理题库 (Mixed Reasoning Dataset)

为提升评估的统计效力并避免单一数据分布偏差，本文构建了 **400题** 的混合多步推理题库。

- **文件路径**: `03_data/reasoning/reasoning_mixed_400.jsonl`
- **构建脚本**: `scripts/02_evaluation/build_mixed_reasoning_set.py`
- **题库构成**:
  1. **自建逻辑/数学题 (120题)**：涵盖 2步~5+步的比较关系、传递推理及代数计算。
  2. **GSM8K 测试集子集 (130题)**：来源于 `openai/gsm8k` (main split)，主要是小学数学应用题。
  3. **BBH 子集 (150题)**：来源于 `lukaemon/bbh`，包含 `multistep_arithmetic_two` (75题) 与 `word_sorting` (75题)。
- **Prompt 模板**: `"{题目内容}\n请逐步推理，最后一行必须以'答案：'开头给出最终结果。"`
- **评分机制**:
  - 采用**分层抽样**策略：主实验模型 (Qwen1.5B) 每档抽取 50 题，其余模型档位每档抽取 30 题。
  - 引入**双人独立盲审**与第三人仲裁机制，计算加权 Cohen's Kappa 保证评分信度。对应的原始输出与打分记录位于 `06_results/manual_scoring/`。

## 3. 外部大文件（模型权重、KLD基准）

由于体积限制，模型文件 (`*.gguf`) 和基准文件 (`*.kld`) 未直接上传至本仓库。

### 3.1 基础模型权重 (Base Models)
请从 HuggingFace 获取原始权重并转换为 FP16 GGUF 格式：
- **Qwen2.5 系列**:
  - [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  - [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Llama 系列**:
  - [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  - [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Phi 系列**:
  - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

### 3.2 KLD 基准文件 (KLD Base Logits)
- KLD 计算必须依赖各模型 FP16 版本对 WikiText-2 推理产生的 logits 基准文件（`.kld`），单文件通常在数百MB量级。
- **生成方式**: 请运行 `scripts/02_evaluation/run_kld_logits_v2.ps1`，脚本会自动生成本地基准文件并保存至 `05_raw/kld_base/`。
- **Zenodo 存档**: 若需直接获取本文实验使用的基准文件，可访问 Zenodo 仓库（DOI: `[待补充]`，SHA256: `[待补充]`）。