# 多步推理重试前的原始输出（legacy）

本目录保存 v1.6.0 之前 `results/raw/reasoning/` 中的 26 个文件。这一版评测脚本在服务偶发失败时未重试，
各档有大量空输出（如 Qwen2.5-1.5B FP16 为 105/400、Q4_K_M 为 176/400），与论文表 S1 及
`results/summary/table4_reasoning_full400.csv`（基于经 3 次重试的新版输出）不一致。

v1.6.0 起，`results/raw/reasoning/` 已替换为新版输出（空输出降至每档约 5~25 题，Phi-3 FP16 为 242 题），
`scripts/03_analysis/extract_reasoning.py` 与 `mcnemar_reasoning400.py` 均以新版为输入。
`results/summary/table5_reasoning_detail.csv` 由本目录的旧版生成，仅供追溯。

`*-f16.gguf.jsonl`（120 行）为更早期 120 题版本的输出，仍保留在 `results/raw/reasoning/` 原位。
