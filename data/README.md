
---

### 2. 完善 `data/README.md`（数据说明与题库）

```markdown
# 数据集说明

本仓库不包含大体积原始数据集，请前往官方渠道下载。针对自建任务，本文提供了生成脚本与原始题库。

## 1. 公开数据集
- **WikiText-2**: 用于 PPL 与 KLD 测试（文件：`wiki.test.raw`），下载链接：[HuggingFace - wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext)
- **ARC-Easy**: 用于常识推理测试（全量 2376 题），数据源：[allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)
  - *计分方式*：模型输出首 token 与标准答案匹配记 1 分。参考 `lm-evaluation-harness` 逻辑。

## 2. 自建多步逻辑推理题库 (120题)
- **文件路径**: `03_data/reasoning/reasoning120.jsonl`
- **生成脚本**: `scripts/02_evaluation/gen_reasoning.py`
- **构成**: 2步(25题)、3步(25题)、4步(35题)、5+步(35题)，满分120分。
- **Prompt模板**: "请逐步推理，最后一行必须以‘答案：’开头给出最终结果。"
- **完整题库**: 请参见论文附录 A.3 或仓库内的 jsonl 文件。

## 3. 外部大文件（模型权重、KLD基准）
由于体积限制，模型文件 (*.gguf) 和超大基准文件（如 `*.kld`）未上传至本仓库。
- **模型托管地址**: 
Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
Llama-3.2-1B-Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
Llama-3.2-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Phi-3-mini-4k-instruct: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **KLD 基准文件 Zenodo DOI 及 SHA256**: [待补充]