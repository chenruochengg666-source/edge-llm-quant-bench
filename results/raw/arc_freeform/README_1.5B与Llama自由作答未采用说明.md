# 1.5B / Llama 自由作答数据：未纳入正文论证的说明

本目录下 `qwen1.5b-f16.gguf.jsonl`、`qwen1.5b-Q3_K_M.gguf.jsonl`、`ll32-f16.gguf.jsonl`、`ll32-Q4_K_M.gguf.jsonl`
是按 0.5B 自由作答对照同一协议（`freeform_v1`：不加约束、`max_tokens=64`）跑出的结果，**未用于论文正文的格式 × 量化 2×2 论证**。

## 原因

0.5B 的自由作答答案通常在 64 token 内给出，但 1.5B 与 Llama-3.2-1B 在不加约束时倾向于输出
"Step 1 / 第一步……"式的分步推理，64 token 上限会在给出最终字母前就截断。这是**模型自身的回答风格**造成的
提取失败，而不是量化带来的效应，因此不能像 0.5B 那样用于验证"选项塌缩"机理：

| 模型 | 答对 | 未提取（字母被截断前输出已结束） |
|---|---|---|
| Qwen2.5-1.5B FP16 | 76.0% | 17.8% |
| Qwen2.5-1.5B Q3_K_M | 9.7% | **89.4%** |
| Llama-3.2-1B FP16 | 7.0% | **79.5%** |
| Llama-3.2-1B Q4_K_M | 14.0% | 56.7% |

## 处理方式

论文正文改用两条已有数据作为旁证，不依赖这批结果：Qwen2.5-0.5B 的 Q4_K_M 在 ARC-Challenge 上选项 A 占比
78%（`results/summary/arc_challenge_judgement.csv`）；Llama-3.2-1B 的 FP16 在约束作答下选项 A 占比 79%
（`results/summary/arc_easy_judgement.csv`）。

若后续要重新尝试在 1.5B / Llama 上做格式 × 量化对照，建议将 `max_tokens` 提高到至少 300，并先对 FP16
做抽样检查确认分步推理能在限额内给出最终答案。
