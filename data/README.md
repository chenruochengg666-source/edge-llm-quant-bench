# 数据集说明（data/）

本目录存放评测所需的语料、题库与样本。**大体积原始数据不纳入 Git**，请按下列说明下载或由脚本生成。

## 1. 公开评测数据集

### WikiText-2（PPL / KLD 语料）
- **存放位置**：`data/wikitext2/wiki.test.raw`（**仓库不含，需自行下载**）
- **来源**：[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)（`wikitext-2-raw-v1`，test split）

### ARC-Easy（常识推理，全量 2376）
- **存放位置**：`data/arc/ARC-Easy-test-full.jsonl`（**仓库不含，需自行下载**）
- **来源**：[allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)（ARC-Easy test split）
- **评测口径**：本地 `llama-server` OpenAI 兼容接口，`temperature=0`、`top_p=1`、`seed=42`、`n_predict=32`；模型输出首 token 与标准答案选项匹配记 1 分，准确率 = 正确数 / 2376。
- *注：7B/8B 规模使用固定种子抽取的 500 题子集，逐题结果见 `results/raw/arc_7b_subset/`、`results/raw/arc_llama8b_subset/`。*

## 2. 混合多步推理题库（400 题）

- **存放位置**（运行构建脚本后生成，**仓库不含**）：
  - `data/reasoning/reasoning_mixed_400.jsonl`（400 题混合）
  - `data/reasoning/reasoning120.jsonl`（120 题自建）
  - `data/reasoning/reasoning_mixed_220.jsonl`（早期版本，保留供追溯）
- **构建脚本**：`scripts/03_analysis/build_mixed_reasoning_set.py`
- **构成**：
  1. 自建逻辑/数学题 **120 题**（2 步~5+ 步比较关系、传递推理、代数计算）
  2. GSM8K 子集 **130 题**（`openai/gsm8k`，main split）
  3. BBH 子集 **150 题**（`lukaemon/bbh`：`multistep_arithmetic_two` 75 + `word_sorting` 75）
- **Prompt 模板**：`"{题目内容}\n请逐步推理，最后一行必须以'答案：'开头给出最终结果。"`
- **评分**：1.0 完全正确 / 0.5 过程正确但格式或数值有误、或漏 1 个条件 / 0.0 错误、乱码、复读、截断
- **抽样与信度**：Qwen1.5B 主要档位每档 50 题、其余档位每档 30 题；**双人独立盲审 + 第三人仲裁**，加权 Cohen's κ = **0.9989**（95% CI 0.9966–1.0），Krippendorff's α = 0.9972。评分明细见 `results/summary/manual_scoring/`。

## 3. 外部大文件（未纳入 Git）

### 3.1 基础模型权重
请从 HuggingFace 获取并转为 FP16 GGUF：
- Qwen2.5：[0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) / [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) / [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) / [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Llama：[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) / [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Phi-3：[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

### 3.2 KLD 基准文件（`.kld`）
- **生成方式**：`scripts/02_evaluation/run_kld_logits_v2.ps1`，输出至 `results/raw/kld/`（FP16 基准 `.kld` 同目录）
- **体积**：单文件数百 MB 量级
- **归档**：如需直接获取本文使用的基准文件，见 Zenodo 归档 <https://doi.org/10.5281/zenodo.23656926>（文件校验值以 Zenodo 页面公布为准）

## 4. 复现提示
- 题库与语料固定后，**所有档位必须使用同一题集**，否则跨档准确率不可比（本文早期分层抽样即因此修正为统一 400 题）。
- 文件校验：`results/summary/chunks_check.csv` 记录 PPL/KLD 的 chunk 完整性检查结果。

> **仓库不含题库与语料本体**（体积与许可原因）。上述路径为下载/生成后的目标位置；`data/` 目录在克隆后仅含本说明文件。
