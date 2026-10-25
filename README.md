# edge-llm-quant-bench

纯 CPU 边缘设备大语言模型量化评测基准复现包。

## ✅ 已完成实验
- [x] **Qwen2.5 系列八级量化评测**：涵盖 0.5B / 1.5B / 3B 模型，从 FP16 至 Q2_K 的全档位工程与保真度测量。
- [x] **跨架构对照**：完成 Phi-3-mini 与 Llama-3.2-1B 的多档位实测。
- [x] **极限规模外推 (7B/8B)**：补充 Qwen2.5-7B 与 Llama-3.1-8B 的最小对照实验，**实测发现 16GB 内存下 FP16 长文本推理 OOM 现象**，验证了量化在边缘设备大模型部署中的先决性。
- [x] **多步推理扩样与混合题库**：构建包含自建逻辑题、GSM8K、BBH 子集的 400 题混合题库，完成全量分层抽样测试。
- [x] **双人盲审与信度检验**：引入双人独立盲审与第三人仲裁机制，计算加权 Cohen's Kappa (0.9989) 与 Krippendorff's Alpha，确保主观评分高度可靠。
- [x] **统计显著性检验**：基于 2σ 规则与 BH-FDR 校正，为崩塌区间提供严格统计依据。
- [x] **工程辅助实验**：KV Cache 消融、长上下文 RSS 测量、电源模式对照、并发吞吐测试。

## ⏳ 待完成实验 (Future Work)
- [ ] 跨平台内存带宽验证 (Intel vs AMD vs Apple Silicon)
- [ ] imatrix 量化对比实验
- [ ] 13B 及以上更大参数模型的 OOM 边界测试

---

## 📂 脚本与论文图表映射表

⚠️ **注意**：复现时请务必修改各脚本开头的 `BASE_DIR` 或绝对路径配置。

| 论文对应内容 | 核心脚本 (Scripts) | 运行说明 / 输出文件 |
| :--- | :--- | :--- |
| **表1：体积与性能** | `extract_performance.py`<br>`measure_rss.ps1` | 依赖 `05_raw/perf` 与 `rss`，输出 `table2_engineering.csv` |
| **表2：PPL/KLD保真度** | `run_kld_logits_v2.ps1`<br>`extract_kld.py`<br>`summarize_kld.py` | 依赖 `05_raw/kld_logs`，输出 `table3_fidelity.csv` 与 `table_kld_summary.csv` |
| **表3：ARC-Easy** | `run_arc_full.py`<br>`extract_arc.py` | 依赖 `05_raw/arc`，输出 `table4_downstream_arc.csv` |
| **表4：多步推理得分** | `build_mixed_reasoning_set.py`<br>`run_reasoning_full.py`<br>`make_manual_tasks.py`<br>`merge_manual_final.py` | 构建 400 题混合题库，跑推理，经双人盲审后合并输出 `table5_reasoning_final_summary.csv` |
| **表8：模型间一致性** | `calc_ca_kappa_final_v2.py` | 计算 FP16 与量化档位逐题的正确性一致率 (CA) 与 Cohen's Kappa |
| **附录 A.5 (表A5)**<br>标注者间信度 | `calc_inter_rater_reliability.py` | 读取 `rater1_scored.csv` 与 `rater2_scored.csv`，计算加权 Kappa、Bootstrap CI 及 Alpha |
| **图1 - 图17**<br>(基础工程与能力图) | `plot_all_final_v3.py` | 统一读取 `06_results/` 下的所有 CSV，一键生成基础 PDF 图表至 `figures/` 目录 |
| **图5：归一化热力图** | `fix_fig04.py` | 融合工程轨与能力轨（含 Reasoning Acc），生成全新版热力图 `fig04_heatmap.pdf` |
| **图10：推理准确率图** | `plot_new_reasoning_summary.py` | 生成带 **准确率Y轴** 及 **OOM 异常标记**的 `fig_reasoning_accuracy_new.pdf` 及保真度背离散点图 |
| **附录 A.2 / A.3**<br>(附加实验) | `extract_kv_power.py` | 提取 KV Cache 消融与电源模式数据 |

---

## 🛠️ 环境依赖
- **Python**: 3.10+
- **推理框架**: llama.cpp (AVX2 编译版, commit 4da633776)
- **数据分析**: `pandas`, `numpy`, `scipy`
- **绘图**: `matplotlib`, `seaborn`
- **统计检验**: `scikit-learn`, `krippendorff`