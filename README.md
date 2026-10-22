# edge-llm-quant-bench

边缘端大语言模型量化基准评测复现包。

## 已完成实验
- [x] Qwen2.5 系列 (0.5B/1.5B/3B) 八级量化评测 (FP16 至 Q2_K)
- [x] 跨架构对照 (Phi-3-mini, Llama-3.2-1B)
- [x] 规模外推极限对照 (Qwen2.5-7B, Llama-3.1-8B 的 Q4_K_M / Q2_K)
- [x] 统计显著性检验 (2σ 规则, BH-FDR 校正)
- [x] 辅助实验 (KV Cache 消融、长上下文、电源模式、并发吞吐)

## 待完成实验
- [ ] 扩充自建推理题库（当前120题偏少，存在地板效应）
- [ ] 跨平台内存带宽验证 (Intel vs AMD vs Apple Silicon)
- [ ] imatrix 量化对比实验
- [ ] 完善多步推理的人工/AI双人标注流程

## 映射表（持续更新）
⚠️ 注意：复现时请修改 scripts 中的 BASE_DIR 路径”。
| 论文图表 | 对应脚本 (Scripts) | 运行说明 / 输出文件 |
| --- | --- | --- |
| **Table 2** (工程性能) | `scripts/01_benchmarks/extract_performance.py`<br>`scripts/01_benchmarks/measure_rss.ps1` | 依赖 `results/raw/perf/` 与 `rss/`，输出 `table2_engineering.csv` |
| **Table 3** (KLD 保真度) | `scripts/02_evaluation/run_kld_logits_v2.ps1`<br>`scripts/03_analysis/extract_kld.py` | 依赖 `results/raw/kld_logs/`，输出 `table3_fidelity.csv` |
| **Table 4** (ARC 下游任务) | `scripts/02_evaluation/run_arc_full.py`<br>`scripts/03_analysis/extract_arc.py` | 依赖 `results/raw/arc/`，输出 `table4_downstream_arc.csv` |
| **Table 5** (多步推理) | `scripts/02_evaluation/run_reasoning_full.py`<br>`scripts/03_analysis/ai_grader_final.py`<br>`scripts/03_analysis/merge_manual_final.py` | AI 自动评分与人工评分（Kappa 校验）合并 |
| **Table 7** (一致性与Kappa) | `scripts/03_analysis/calc_ca_kappa_final_v2.py` | 计算 FP16 vs Quant 的正确性一致性 (Correctness Agreement) |
| **Fig 01 - Fig 14** (全套图表) | `scripts/04_plotting/plot_all_final_v3.py` | 统一读取 `06_results/` 下所有 CSV，一键生成所有 PDF 图片至 `figures/` |
| **Table A2/A3** (附加实验) | `scripts/01_benchmarks/extract_kv_power.py` | 提取 KV Cache 消融与电源模式数据 |