# 基于纯 CPU 边缘设备的大模型量化评估

> 在 16 GB 无独显轻薄本上，用 llama.cpp 对 Qwen2.5（0.5B/1.5B/3B/7B）、Phi-3-mini、Llama-3.2-1B、Llama-3.1-8B 做从 FP16 到 Q2_K 的系统量化评测，检验 **PPL/KLD 这类保真度指标能否代表下游能力**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b11223-blue)](https://github.com/ggml-org/llama.cpp)
[![Version](https://img.shields.io/badge/version-v1.4.0--paper-green)](CITATION.cff)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23656925-blue)](https://doi.org/10.5281/zenodo.23656925)

- 主仓库（Gitee）：<https://gitee.com/yfddyqcxy/edge-llm-quant-bench>
- 镜像/归档（GitHub）：<https://github.com/chenruocheng666-source/edge-llm-quant-bench>
- 论文归档（Zenodo DOI）：<https://doi.org/10.5281/zenodo.23656925>
- 对应提交：`72d13b9ea2116767eae34ce1ebcdb6ed0f9fc491`

## 版本记录

| 版本 | 日期 | Zenodo 版本说明 | 主要变更 |
|---|---|---|---|
| **v1.4.0-paper** | 2026-11-08 | 见下方「v1.4.0 版本说明」 | 补齐多步推理全量（26 档 × 400 题，含 7B/8B）；新增电源 2×2 析因与 J/token 能耗；新增表5A 双轨判定与表8 全量 CA/Kappa；修正 Extra 口径；文献按首引顺序重排至 41 条；修复 6 个脚本 BOM 与续跑缺陷 |
| v1.3.0 | 2026-10-31 | 文档与图表规范化修订（Documentation and figure normalization） | 论文表格统一为三线表；补充附录A（复现记录、实验环境与补充实验数据）；图01–图19 统一命名与归档；核心实验数据未变动 |
| v1.2.1 | 2026-10-26 | feat(p04): structured benchmark, figures, and docs sync | 80 题结构化评测（四级指标 + Extra）、图18、统计判定规则统一、README/env/docs 同步 |
| v1.1.1 | 2026-10-20 | Complete P0-1 statistical validation | 完成统计规范化：Bootstrap CI、BH-FDR、Cohen's h/d |

> 说明：v1.1.1 与 v1.2.1 的版本说明沿用早期提交信息（英文）；v1.3.0 起改为内容说明；v1.4.0 起采用中英文对照的规范化描述。


### v1.4.0 版本说明（Zenodo）

**中文：**

> 补齐多步推理全量数据（26 个「模型×档位」组合 × 400 题混合题库，含 2/3/4/5+ 步分层与有效响应数；7B/8B 的 FP16 因 OOM 未纳入）；新增电源 2×2 析因实验（插电/电池 × 最佳性能/最佳能效，每组 n=5）与 J/token 净能耗；新增双轨交叉判定（表5A，四类区间）与基于全量交集的实例级一致性（表8，N=2040~2332）；修正 Extra 指标口径并重算结构化基准；参考文献按首引顺序重排至 41 条并新增 2.4 节；修复 6 个脚本的 UTF-8 BOM 与续跑判定缺陷。

**English:**

> Full-scale multi-step reasoning benchmark (26 model–quantization combinations × 400 mixed tasks, with per-level breakdown and valid-response counts; 7B/8B FP16 excluded due to OOM); new 2×2 factorial power experiment (AC/battery × best-performance/best-efficiency, n = 5 per cell) with J/token net-energy accounting; new dual-track interval judgement (Table 5A, four-range taxonomy) and instance-level agreement recomputed on full ARC-Easy intersections (Table 8, N = 2040–2332); corrected the Extra metric definition and recomputed the structured benchmark; references reordered by first citation (41 entries) with a new related-work section 2.4; fixed UTF-8 BOM in 6 scripts and repaired the resume logic that previously treated empty predictions as completed.

## 实验进度

- [x] **PO-1 统计检验规范化**：两比例 z（Cohen's h）+ Welch t（Cohen's d）+ BH-FDR，判定为「q<0.05 且 |效应量|≥0.2」
- [ ] PO-2 imatrix 敏感性
- [x] **P0-3 多种子复测**：500 题子集 × 6 档位 × 种子 42–46（15 000 次）；FP16/Q4_K_M 种子间 SD ≤ 0.18pp、逐题一致率 ≥ 99.2%，见 `results/summary/table14_seed_stability.csv`、图19
- [x] **P0-4 结构化生成任务**：80 题（JSON 提取 30 / 表格 25 / 函数调用 25），四级指标（Parseable / Fields / Schema / Content）+ 洁净度（Extra）；见 `results/summary/table_struct_summary_v2.csv`、图18
- [ ] P1-5 Needle 深度扫描
- [ ] P1-6 安全对齐
- [ ] P1-7 决策树 + 能力矩阵

---

## 核心发现

1. **效率红利**：Qwen2.5-1.5B 量化至 Q4_K_M，体积 2949→940 MiB（-68.1%），生成速度 11.84→34.35 t/s（约 2.9×）。
2. **保真度与下游能力存在系统性错配**：PPL/KLD 与 ARC-Easy 呈方向背离（部分档位准确率反升）与幅度背离（KLD 微变而下游无显著差异）；跨词表 KLD 不可比。
3. **量化正则化效应**（限定 ARC-Easy 全量）：Qwen2.5-0.5B 的 Q4_K_M 档 **+15.2pp**（q<0.001，Cohen's h=**0.311**，显著）；Qwen2.5-3B **+7.2pp**（q<0.001 但 h=**0.188**，未达 0.2 实质差异阈值，仅边缘证据）；Qwen2.5-1.5B 不显著。**该效应未在多步推理任务上复现**（统一 400 题呈单调下降）。
4. **崩塌区间统计显著**：Qwen2.5-1.5B/3B 的 Q2_K 相对 FP16，ARC 差异 q<0.001 且 h=**0.867**（1.5B）/ **2.165**（3B）；KLD 经 Welch t 检验 q<0.001、Cohen's d≥21，2σ 保守校核为 29.2σ / 101.2σ。
5. **量化安全边界具模型特异性**：Phi-3-mini 的 Q2_K 未完全崩塌（ARC 0.670，h=0.269）；Llama-3.2-1B 的 Q4_K_M 即显著退化（0.361→0.120，h=0.581），Q2_K 回升至 0.272 但 h=0.191 未达阈值；7B/8B 规模上 Q2_K 仍保有较强韧性（Qwen7B 85.00%）。
6. **长上下文瓶颈在 KV Cache**：1.5B Q4_K_M 可稳定运行至 256K，峰值 RSS 512→262144 严格线性（**28.05 KB/token**，与理论 28 KB/token 吻合），KV 占用约为权重的 8 倍；同一区间内预填充速度 155.81→19.91 t/s（**-87.2%**）而生成速度 34.08→32.80 t/s（**-3.8%**），说明长上下文瓶颈在内存带宽而非算力。

---

> **统计口径说明**：ARC-Easy 准确率按「排除推理失败后的有效样本」计算（服务崩溃/超时不计为答错），
> 失败率单列于 `table4_downstream_arc.csv` 的 `Failure_Rate(%)` 列。

## 仓库结构

```text
edge-llm-quant-bench/
├── scripts/
│   ├── 01_benchmarks/        # 性能、内存、上下文伸缩与速度
│   │   ├── measure_ctx_ext.ps1     # -c 512~262144，峰值 RSS + OOM 判定
│   │   ├── measure_ctx_speed.ps1   # llama-bench -p 4096~131072，预填充/生成速度
│   │   ├── measure_rss.ps1         # 单档峰值 RSS
│   │   └── extract_performance.py  # llama-bench 结果解析
│   ├── 02_evaluation/        # ARC-Easy / 多步推理 / 结构化生成 / 种子复测
│   │   ├── run_arc_full.py
│   │   ├── run_arc_seed.py         # 种子 42–46 重复推理（断点续跑 + 原子写盘）
│   │   ├── run_reasoning_full.py
│   │   └── run_structured_v2.py    # 80 题结构化任务（支持断点续跑）
│   ├── 03_analysis/          # 统计检验、信度、汇总
│   │   ├── stat_tests_v2.py        # 两比例 z + Welch t + BH-FDR + Cohen's h/d
│   │   ├── calc_inter_rater_reliability.py
│   │   ├── calc_ca_kappa_final_v2.py
│   │   ├── build_structured_set_v2.py   # 结构化题库（含 expected）
│   │   ├── eval_structured_v2.py        # 四级指标 + 洁净度评分
│   │   ├── prepare_arc_subset.py        # 固定 seed(0) 抽 500 题子集
│   │   └── analyze_arc_seed.py          # 种子间 SD / 逐题一致率统计
│   └── 04_plotting/          # 论文图表
├── data/                     # 题库与语料（不入库，由脚本下载/生成；见 data/README.md）
│   ├── arc/  reasoning/  wikitext2/  structured/
├── results/
│   ├── summary/              # 汇总表（table*.csv，含 table_struct_*_v2）
│   ├── raw/                  # 原始输出（arc / reasoning / kld / perf / rss / ctx / structured_v2）
│   └── legacy/               # 历史版本结果（如 table_stat_tests_legacy.csv）
├── figures/                  # 论文图表 PNG（fig01–fig18，_legacy 为历史图）
├── docs/                     # 说明文档（如 PO4_structured_v2_notes.md）
├── configs/                  # 配置
└── README.md / env.md / CITATION.cff / LICENSE
```

> 说明：模型权重与 KLD 基准 `.kld` 大文件受原始许可与体积限制，未纳入 Git。请按下文「快速开始」自行生成。

---

## 环境

| 项 | 配置 |
|---|---|
| 设备 | ASUS Vivobook 14 |
| CPU | Intel Core Ultra 5 225H（14 线程，异构） |
| 指令集 | AVX2、AVX-VNNI（**不支持 AVX-512**） |
| 内存 | 16 GB DDR5，无独显（纯 CPU） |
| 系统 | Windows 11（实验在原生环境执行，非容器） |
| 框架 | llama.cpp `4da633776`（build 11223，AVX2） |
| Python | 3.10（pandas / numpy / scipy / statsmodels / matplotlib） |

完整参数见 [env.md](env.md)。**性能测试前请关闭浏览器、微信等常驻程序**，并将页面文件扩至 16–32 GiB（3B FP16 转换与 256K 上下文需要）。

---

## 快速开始

### 1. 准备模型与量化

```bash
python convert_hf_to_gguf.py <hf_dir> --outfile model-f16.gguf --outtype f16

llama-quantize.exe model-f16.gguf model-Q8_0.gguf  Q8_0
llama-quantize.exe model-f16.gguf model-Q6_K.gguf  Q6_K
llama-quantize.exe model-f16.gguf model-Q5_K_M.gguf Q5_K_M
llama-quantize.exe model-f16.gguf model-Q4_K_M.gguf Q4_K_M
llama-quantize.exe model-f16.gguf model-Q2_K.gguf  Q2_K
```

### 2. 保真度（PPL / KLD）

```bash
llama-perplexity.exe -m model-f16.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base model-f16.kld

llama-perplexity.exe -m model-Q4_K_M.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base model-f16.kld
```

### 3. 性能与内存

```bash
llama-bench -m model-Q4_K_M.gguf -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0
```

```powershell
.\scripts\01_benchmarks\measure_rss.ps1       -Model .\model-Q4_K_M.gguf -Ctx 4096 -Out .\results\raw\rss\q4km.csv
.\scripts\01_benchmarks\measure_ctx_ext.ps1   -Model .\model-Q4_K_M.gguf -Out .\results\raw\ctx\ctx_ext.csv
.\scripts\01_benchmarks\measure_ctx_speed.ps1 -Model .\model-Q4_K_M.gguf -Out .\results\raw\ctx\ctx_speed.csv
```

### 4. 下游能力与统计

```bash
# 注：评测脚本无命令行参数，运行前请编辑脚本顶部的模型路径 / 端口 / 输出目录
python scripts/02_evaluation/run_arc_full.py         # ARC-Easy 全量 2376
python scripts/02_evaluation/run_reasoning_full.py   # 多步推理 120 / 400 题
python scripts/03_analysis/stat_tests_v2.py          # z / Welch t / BH-FDR / 效应量
python scripts/04_plotting/plot_all_final_v3.py      # 重绘论文图表
```

---

## 结果文件对照

| 论文表/图 | 数据文件 |
|---|---|
| 表1 体积与推理性能 | `results/summary/table2_engineering.csv` |
| 表2 保真度（PPL/KLD/Same top-p） | `results/summary/table_kld_summary.csv`、`table3_fidelity.csv` |
| 表3 ARC-Easy（2376，含 q 值、效应量与失败率） | `results/summary/table3_ARC_final.csv`、`table4_downstream_arc.csv`、`table5_stat_final.csv` |
| 表4 多步推理得分（400 题全量有效响应） | `results/summary/table4_reasoning_full400.csv`；人工评分子集见 `table5_reasoning_final_summary.csv`、`table5_reasoning_detail_human_final.csv` |
| 表5 关键统计检验 | `results/summary/table5_stat_final.csv` |
| 表6 跨架构失败模式 | `results/summary/table4_downstream_arc.csv` |
| 表7 7B/8B 对照 | `results/summary/table7b_kld_summary.csv`、`7b_bench_*.csv`、`llama_8b_*.csv` |
| 表5A 双轨交叉判定矩阵（四类区间） | `results/summary/table5A_zone_judgement.csv` |
| 表8 实例级一致性（ARC 全量交集） | `results/summary/table8_ca_kappa_full.csv` |
| 表9 长上下文 | `results/raw/ctx/ctx_qwen1.5b_ext.csv`、`ctx_qwen1.5b_speed.csv` |
| 表10 KV Cache 消融 | `results/summary/tableA3_kv_cache.csv` |
| 表11 电源 2×2 析因与能耗（J/token） | `results/summary/power_2x2.csv`、`table11_power_energy.csv`；原始读数见 `results/raw/power_log/` |
| 附录 A.7 表A2/A3（完整统计量） | `results/summary/tableA2_power.csv`、`tableA3_kv_cache.csv` |
| 双人评分信度 | `results/summary/table_inter_rater_reliability.csv` |
| 表12/表13、图18 结构化生成 | `results/summary/table_struct_summary_v2.csv`、`table_struct_detail_v2.csv` |
| 表14、图19 种子稳健性 | `results/summary/table14_seed_stability.csv`、`table_p03_seed_anova.csv` |

---

## 图 ↔ 脚本对照

| 论文图 | 内容 | 生成脚本 | 状态 |
|---|---|---|---|
| 图1 | pp512 / tg128 双轴对比折线图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图2 | 体积–解码速度散点关系图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图3 | 五个模型的峰值内存占用 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图4 | PPL / KLD / Same top-p 三联对比 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图5 | Qwen2.5-1.5B 归一化热力图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图6 | KLD 与 ARC-Easy 双轴折线图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图7 | 跨架构 ARC-Easy 对比柱状图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| **图8** | Qwen2.5-1.5B 多步推理得分矩阵热力图 | **`plot_reasoning_v2.py`** | ✅ 可复现 |
| **图9** | Qwen2.5-3B 多步推理得分矩阵热力图 | **`plot_reasoning_v2.py`** | ✅ 可复现 |
| **图10** | 多步推理总准确率柱状图 | **`plot_reasoning_v2.py`** | ✅ 可复现 |
| **图11** | 7B/8B 保真度与下游能力折线图 | **`plot_7b8b_results.py`** | ✅ 已对齐输出名 |
| **图12** | 跨架构 7B/8B ARC-Easy 对比图 | **`plot_7b8b_results.py`** | ✅ 已对齐输出名 |
| **图13** | 实例级一致率（CA）热力图 | **`plot_reasoning_v2.py`** | ✅ 可复现 |
| 图14 | 长上下文峰值 RSS 折线图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图15 | KV Cache 消融柱状图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图16 | 电源模式对照柱状图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| 图17 | 并发吞吐折线图 | 早期脚本（已归档） | ⚠️ 命名已变更 |
| **图18** | 结构化生成内容正确率 | **`plot_structured.py`** | ✅ 已对齐输出名 |
| 图19 | 随机种子稳健性 | 早期脚本（已归档） | ⚠️ 命名已变更 |

> **关于「命名已变更」**：2026-10 早期的绘图脚本采用语义命名（如 `fig08_kv_cache.pdf` → 现为**图15**、`fig13_reason_total_score.pdf` → 现为**图10**），与论文当前的顺序命名（`fig01.png`~`fig19.png`）不一致。这些脚本已归档至 `scripts/04_plotting/_deprecated/`，仍可用于追溯，但**输出文件名需按上表映射后重命名**才能与论文对应。
>
> 图8–13、18 的脚本已按论文命名对齐，可直接复现。

## 复现性要点

- **评测参数**：`temperature=0`、`top_p=1`、`seed=42`；推理 `max_tokens=1024`，下游批量 `256`。
- **线程**：`llama-bench` 用 `-t 4`；批量下游推理与 PPL/KLD 用 `-t 8`。
- **KLD**：`--chunks 20`（n=20），以 FP16 输出 logits 为基准；`Same top-p` 报告均值。
- **统计判定**：ARC 用两比例 z 检验（效应量 Cohen's h），p 值经 BH-FDR 校正；**q<0.05 且 |效应量|≥0.2 方视为实质差异**。
- **保真度轨判定**：以 3.3 节 σ_pool 合成标准差规则（|ΔPPL|≥2σ_pool）为**主判据**；因 FP16 基准的 KLD 恒为零、方差为零，KLD 的 Welch t 检验实质退化为「KLD 是否异于零」的单样本检验，其结果为 15 档全部显著，仅作附录 A.8 的敏感性参考，不作主判据。
- **配对敏感性**：各档位在同一批 ARC-Easy 题目上作答，构成配对设计，附录 A.8 给出 McNemar 精确检验结果；正文表3 仍以边际口径（两比例 z 检验）为准。
- **区间判定脚本**：`scripts/03_analysis/calc_zone_judge.py` 复现表5A 与附录表A10，输出 `results/summary/table5A_zone_judgement.csv`。
- **人工评分**：400 题分层抽样（每档 30–50 题），双人独立盲审，加权 Cohen's κ = **0.9989**（95% CI 0.9966–1.0），Krippendorff's α = 0.9972。
- **上下文**：RSS 测 512~262144（`-c` 逐档，`n=1`）；速度测 4096~131072（`llama-bench -r 1`）；256K 速度未测（单次约 30 分钟）。
- **电源与能耗**：2×2 析因设计（插电/电池 × 最佳性能/最佳能效），每组 n=5，`llama-bench -p 512 -n 128 -r 5 -t 4`；插电用墙插功率计测整机输入（含适配器约 10~15% 损耗），电池用 BatteryInfoView 读 ACPI 传感器（DC 侧），二者测量点不同、绝对功耗不可直接比较；净能耗 =（稳态负载功耗 − 同模式待机功耗）/ tg128。脚本见 `scripts/01_benchmarks/power_mode_test.py` 与 `scripts/03_analysis/calc_energy_from_log.py`。
- **一致性核对**：`table_kld_summary.csv` 与 `table3_fidelity.csv` 的 `Same top-p` 为同一来源，已对齐。

---

## 已知限制

1. 硬件为单台 16 GB 无独显轻薄本，结论限于「纯 CPU + AVX2」场景。
2. 主实验覆盖 0.5B–3B，7B/8B 仅测 FP16/Q4_K_M/Q2_K 三档；13B 以上未测。
3. **跨词表 KLD 不可比**：Phi-3(32064) / Llama-3.2(128256) / Qwen2.5(151936) 词表不同，跨模型仅比较 ARC-Easy。
4. 多步推理分层题源不同（2 步层以 BBH 词排序为主），层间得分不单调，档位比较以「总体」列为准。
5. 长上下文测至 256K；512K 未测。检索任务（NIAH）仅在 4K/8K 执行。
6. Phi-3-mini-4k 原生上下文 4096，不适用于 16K 以上测试。
7. 上下文速度为单次测量（`-r 1`），未报告标准差；主表小上下文（512）为 `n=5` 均值。
8. 多步推理原始输出存在服务中断导致的空响应（各档有效响应数不同，最低约 158/400），已由 `clean_reasoning_jsonl.py` 清理空记录并通过修复后的续跑逻辑补测；相关结论以「有效响应」为分母，见附录说明。

---

## 引用

```bibtex
@misc{edge_llm_quant_bench_2026,
  title  = {基于纯 CPU 边缘设备的大模型量化评估：保真度指标失效边界与跨架构实证研究},
  author = {<作者姓名>},
  year   = {2026},
  doi    = {10.5281/zenodo.23656925},
  url    = {https://github.com/chenruocheng666-source/edge-llm-quant-bench}
}
```

## 许可证

本项目代码与文档采用 **MIT License**；模型权重遵循各自原始许可（Qwen / Phi-3 / Llama）。
