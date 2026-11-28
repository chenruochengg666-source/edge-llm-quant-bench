# ARC 统一评测协议重测（v1.6.0）

## 为什么要重测

早期 ARC-Easy 主实验（`scripts/02_evaluation/run_arc_full.py`）对 FP16 与量化档使用了不同的评测协议：

| 设置 | 早期 FP16 | 早期量化档 | 本目录（统一协议） |
|---|---|---|---|
| 提示词末尾 | 追加“只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容” | 不追加 | 所有档位均追加 |
| `max_tokens` | 64 | 8 | 64 |
| 答案提取 | 全文搜索 `\b[A-E]\b`（会把冠词 a 误判为 A） | 同左 | `extract_letter()`：整段为单个字母 / 开头为字母 / 明确的“答案是 X”句式 |
| 数字标签（1~4） | 未转换，一律判错 | 未转换 | 映射为 A~D |
| 服务预热 | 无（开头一批题因模型未加载完而空输出） | 无 | 预热至可正常响应再开始，空输出重试 3 次 |

协议差异与量化效应相互混淆：统一协议后，Qwen2.5-0.5B 的 Q4_K_M 由“+14.9pp（能力反升）”变为“−11.7pp（沉默区）”，Llama-3.2-1B 的 Q4_K_M 由“−25.0pp”变为“+2.3pp”。作答稳定的 Qwen2.5-1.5B/3B 与 Phi-3-mini 受影响在 1pp 以内。

## 做了什么

| 实验 | 规模 | 输出 |
|---|---|---|
| ARC-Easy 统一协议重跑 | 15 个量化档 × 2376 题 | `results/raw/arc_unified/<模型>.gguf.jsonl` |
| FP16 缺失题补跑 | 5 个 FP16，共 846 题（原 FP16 协议本已与统一协议一致） | 同上，`source` 字段区分 `original` / `rerun` |
| FP16 一致性抽检 | 每个 FP16 固定抽 100 道原有输出的题重跑 | `<模型>.audit.jsonl`、`_fp16_audit_summary.csv`（500/500 逐字一致） |
| ARC-Challenge 补跑 | Qwen2.5-0.5B、Phi-3-mini 各 3 档 × 1172 题 | `results/raw/arc_challenge_unified/` |
| 答题格式 × 量化 2×2 对照 | Qwen2.5-0.5B 的 FP16 与 Q4_K_M，不加约束的自由作答 × 2376 题 | `results/raw/arc_freeform/` |
| 7B 子集补跑 | Qwen2.5-7B Q4_K_M 开头 32 题（原协议），另抽检 30 题（30/30 一致） | `results/raw/arc_7b_subset_fill/` |

每个输出目录下的 `_run_log.csv` 记录每次运行的起止时间、题数、答对数、未提取数与空输出数。

## 复现

```powershell
# 1. 环境变量（模型目录请使用纯英文路径；可用分号分隔多个目录）
$env:LLAMA_CPP_BIN      = "D:\llama.cpp\build\bin\Release"
$env:EDGE_LLM_MODEL_DIR = "D:\gguf"
$env:EDGE_LLM_DATA_DIR  = "D:\edge-llm-quant-bench\data"   # 可省略，默认为仓库 data/

# 2. 检查文件与预计耗时（不运行模型）
python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --check

# 3. 分两批运行（可随时中断，重新执行同一命令即续跑）
python -u scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --plan A   # 约 4 小时：0.5B/Llama-1B/Phi-3 共 6 档 + FP16 补跑与抽检 + 0.5B Challenge
python -u scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --plan B   # 约 9 小时：自由作答对照 + 7B 补跑 + Qwen 其余 9 档 + Phi-3 Challenge

# 4. 重算判定与对照
python scripts/03_analysis/analyze_arc_unified.py                      # → results/summary/arc_easy_judgement.csv
python scripts/03_analysis/analyze_arc_unified.py --task challenge     # → results/summary/arc_challenge_judgement.csv
python scripts/03_analysis/analyze_freeform_2x2.py                     # → results/summary/freeform_2x2_summary.csv
```

ARC 题库需按 `data/README.md` 自行下载到 `data/arc/`（`ARC-Easy-test-full.jsonl`、`ARC-Challenge-test-full.jsonl`）。

## 判定规则（`analyze_arc_unified.py`）

- 保真度报警：|ΔPPL| ≥ 2σ_Δ，σ_Δ = √(σ_FP16² + σ_q²)，PPL 取 KLD 运行中的 Mean PPL(base)/Mean PPL(Q)（ctx=512，chunks=20）
- 能力变化：两比例 z 检验，模型内 BH-FDR 校正，q < 0.05 且 |Cohen’s h| ≥ 0.2
- 判定前自检：C1 FP16 准确率显著高于 25% 随机基线；C2 FP16 单一选项预测占比 ≤ 50%（不满足 → 判定不适用）
- 附加标注（不改变区间）：结构偏移（统计等价区内 McNemar 显著）、选项塌缩（量化档单一选项占比 > 50%）、格式失控（量化档 > 5% 输出无法提取答案）

## 主要结果

| 模型 | 档位 | FP16 | 量化 | Δ (pp) | h | 区间 | 标注 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Q4_K_M | 56.02 | 44.32 | −11.70 | 0.235 | **沉默区** | 选项塌缩（A 73%） |
| Qwen2.5-0.5B | Q2_K | 56.02 | 48.57 | −7.45 | 0.149 | 保真先行区 | |
| Qwen2.5-1.5B | Q4_K_M | 90.28 | 89.18 | −1.09 | 0.036 | 统计等价区 | 结构偏移 |
| Qwen2.5-1.5B | Q3_K_M | 90.28 | 85.48 | −4.80 | 0.148 | 保真先行区 | |
| Qwen2.5-3B | Q4_K_M | 93.22 | 93.18 | −0.04 | 0.002 | 保真先行区 | |
| Qwen2.5-3B | Q3_K_M | 93.22 | 78.62 | −14.60 | 0.435 | 同步退化区 | |
| Phi-3-mini | Q4_K_M | 96.13 | 95.50 | −0.63 | 0.032 | 统计等价区 | 结构偏移 |
| Phi-3-mini | Q2_K | 96.13 | 68.00 | −28.13 | 0.806 | 同步退化区 | 格式失控（572 题） |
| Llama-3.2-1B | Q4_K_M | 39.39 | 41.71 | +2.32 | 0.047 | 判定不适用 | FP16 A 选项 79% |

全部 15 档见 `results/summary/arc_easy_judgement.csv`。

答题格式 × 量化 2×2（Qwen2.5-0.5B，四格均可提取的 1512 题）：约束作答 FP16 58.13% / Q4_K_M 45.30%（−12.8pp）；自由作答 72.82% / 71.16%（−1.7pp，p=0.08）；交互 −11.2pp，95% CI [−13.9, −8.5]。
