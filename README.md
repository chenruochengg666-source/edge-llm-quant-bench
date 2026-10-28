# 基于纯 CPU 边缘设备的大模型量化评估

> 在 16 GB 无独显轻薄本上，用 llama.cpp 对 Qwen2.5（0.5B/1.5B/3B）、Phi-3-mini、Llama-3.2-1B、Qwen2.5-7B、Llama-3.1-8B 做从 FP16 到 Q2_K 的系统量化评测，检验 **PPL/KLD 这类保真度指标能否代表下游能力**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b11223-blue)](https://github.com/ggml-org/llama.cpp)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23656926-blue)](https://doi.org/10.5281/zenodo.23656926)

- 主仓库（Gitee）：<https://gitee.com/yfddyqcxy/edge-llm-quant-bench>
- 镜像/归档（GitHub）：<https://github.com/chenruocheng666-source/edge-llm-quant-bench>
- 论文归档（Zenodo DOI）：<https://doi.org/10.5281/zenodo.23656926>
- 对应提交：`8875f9c2fdcb97843250484eca86874ec4d00978`

## 未完成工作
**PO-1统计检验规范化**
**PO-2imatrix敏感性**
**P0-3多种子复测**
**P0-4结构化生成任务**
**P1-5Needle深度扫描**
**P1-6安全对齐**
**P1-7决策树+能力矩阵**

---

## 核心发现

1. **效率红利**：Qwen1.5B 量化至 Q4_K_M，体积 2949→940 MiB（-68.1%），生成速度 11.84→34.35 t/s（约 2.9×）。
2. **保真度与下游能力存在系统性错配**：PPL/KLD 与 ARC-Easy 呈方向背离（部分档位准确率反升）与幅度背离（KLD 微变而下游无显著差异）；跨词表 KLD 不可比。
3. **量化正则化效应**（限定于 ARC-Easy 全量）：Qwen0.5B +15.2pp（p=1.3e-26）、Qwen3B +7.2pp（p=1.2e-10）；1.5B 不显著（q=0.60）。**该效应未在多步推理任务上复现**（统一 400 题呈单调下降）。
4. **崩塌区间统计显著**：1.5B 的 Q4_K_M 与 Q2_K 的 KLD 相差约 29.2σ，3B 约 101.2σ（2σ 判定规则）。
5. **量化安全边界具模型特异性**：Phi-3-mini 的 Q2_K 未完全崩塌（ARC 0.670），Llama-3.2-1B 的 Q4_K_M 即显著退化（0.361→0.120）；7B/8B 规模上 Q2_K 仍保有较强韧性（Qwen7B 85.00%）。
6. **长上下文瓶颈在 KV Cache**：1.5B Q4_K_M 可稳定运行至 256K，峰值 RSS 512→262144 严格线性（**28.05 KB/token**，与理论 28 KB/token 吻合），KV 占用约为权重的 8 倍；权重量化无法缓解这一开销。

---

## 仓库结构

```text
edge-llm-quant-bench/
├── scripts/                 # 全部实验脚本（本地对应 04_scripts/）
│   ├── measure_ctx_ext.ps1  # 上下文伸缩 + OOM 判定（512~256K）
│   ├── measure_rss.ps1      # 单档峰值 RSS
│   ├── run_arc_full.py      # ARC-Easy 全量 2376 题评测
│   ├── run_reasoning_full.py# 多步推理题库（120 题自建 / 400 题混合）
│   ├── calc_*_kappa*.py     # 双人评分信度（加权 Kappa / Bootstrap / Alpha）
│   └── ai_grader_final.py   # 逐题自动评分
├── data/                    # 题库与语料（本地对应 03_data/）
│   ├── arc/                 # ARC-Easy 全量 2376
│   ├── reasoning/           # 120 题自建 + 220/400 题混合题库
│   └── wikitext2/           # wiki.test.raw（PPL/KLD）
├── raw/                     # 原始输出（本地对应 05_raw/）
│   ├── arc/  reasoning/  kld/  perf/  rss/  ctx/  concurrency/
├── results/                 # 汇总表与图（本地对应 06_results/）
│   ├── table1_*.csv … tableA3_*.csv
│   └── figures_png/         # 论文图表
└── README.md
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
| Python | 3.10（pandas / numpy / scipy / matplotlib） |

**性能测试前请关闭浏览器、微信等常驻程序**，并将页面文件扩至 16–32 GiB（3B FP16 转换与 256K 上下文需要）。

---

## 快速开始

### 1. 准备模型与量化

```bash
# HF 权重 → FP16 GGUF
python convert_hf_to_gguf.py <hf_dir> --outfile model-f16.gguf --outtype f16

# 逐档量化（示例）
llama-quantize.exe model-f16.gguf model-Q8_0.gguf  Q8_0
llama-quantize.exe model-f16.gguf model-Q6_K.gguf  Q6_K
llama-quantize.exe model-f16.gguf model-Q5_K_M.gguf Q5_K_M
llama-quantize.exe model-f16.gguf model-Q4_K_M.gguf Q4_K_M
llama-quantize.exe model-f16.gguf model-Q2_K.gguf  Q2_K
```

### 2. 保真度（PPL / KLD）

```bash
# FP16 生成基准 logits
llama-perplexity.exe -m model-f16.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base model-f16.kld

# 各量化档位
llama-perplexity.exe -m model-Q4_K_M.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence --kl-divergence-base model-f16.kld
```

### 3. 性能与内存

```bash
llama-bench -m model-Q4_K_M.gguf -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 -fa on -ngl 0
```

```powershell
# 峰值 RSS
.\scripts\measure_rss.ps1 -Model ".\model-Q4_K_M.gguf" -Ctx 4096 -Out ".\raw\rss\q4km.csv"
# 上下文伸缩（512 ~ 256K，含 OOM 判定）
.\scripts\measure_ctx_ext.ps1 -Model ".\model-Q4_K_M.gguf" -Out ".\raw\ctx\q4km_ext.csv"
```

### 4. 下游能力

```bash
# 注：以下脚本均无命令行参数，运行前请编辑脚本顶部的模型路径 / 端口 / 输出目录
python scripts/run_arc_full.py           # ARC-Easy 全量 2376 题
python scripts/run_reasoning_full.py     # 多步推理：120 题自建 / 400 题混合题库
python scripts/stat_tests.py             # 两比例 z 检验 + BH-FDR / Welch t 检验
python scripts/plot_all_final_v3.py      # 重绘论文图表
```

---

## 结果文件对照

| 论文表/图 | 数据文件 |
|---|---|
| 表1 体积与推理性能 | `results/table2_engineering.csv` |
| 表2 保真度（PPL/KLD/Same top-p） | `results/table_kld_summary.csv`、`table3_fidelity.csv` |
| 表3 ARC-Easy（2376） | `results/table4_downstream_arc.csv` |
| 表4 多步推理得分 | `results/table5_reasoning_final_summary.csv`、`table5_reasoning_detail.csv` |
| 表5 关键统计检验 | `results/table_stat_tests.csv` |
| 表6 跨架构失败模式 | `results/table4_downstream_arc.csv` |
| 表7 7B/8B 对照 | `results/table7b_kld_summary.csv`、`7b_bench_*.csv`、`llama_8b_*.csv` |
| 表8 实例级一致性 | `results/table7_ca_kappa_final.csv` |
| 表9 长上下文 | `raw/ctx/ctx_*_ext.csv` |
| 表10/11 KV Cache、电源模式 | `results/tableA3_kv_cache.csv`、`tableA2_power.csv` |
| 双人评分信度 | `results/table_inter_rater_reliability.csv` |

---

## 复现性要点

- **评测参数**：`temperature=0`、`top_p=1`、`seed=42`、`max_tokens=1024`（推理）/`256`（下游批量）。
- **线程**：`llama-bench` 用 `-t 4`；批量下游推理与 PPL/KLD 用 `-t 8`（均为实测调优结果，见附录图 A1）。
- **KLD**：`--chunks 20`，以 FP16 输出 logits 为基准；`Same top-p` 报告均值。
- **统计判定**：`σ_pool = √(σ₁²+σ₂²)`，`|x̄₁−x̄₂| ≥ 2σ_pool` 视为显著；ARC 用两比例 z 检验 + BH-FDR，KLD 用 Welch t 检验。
- **人工评分**：400 题分层抽样（每档 30–50 题），双人独立盲审，加权 Cohen's κ = **0.9989**（95% CI 0.9966–1.0），Krippendorff's α = 0.9972。
- **一致性核对**：`results/table_kld_summary.csv` 与 `table3_fidelity.csv` 的 `Same top-p` 为同一来源，已对齐。

---

## 已知限制

1. 硬件为单台 16 GB 无独显轻薄本，结论限于「纯 CPU + AVX2」场景。
2. 主实验覆盖 0.5B–3B，7B/8B 仅测 FP16/Q4_K_M/Q2_K 三档；13B 以上未测。
3. **跨词表 KLD 不可比**：Phi-3(32064) / Llama-3.2(128256) / Qwen2.5(151936) 词表不同，跨模型仅比较 ARC-Easy。
4. 多步推理分层题源不同（2 步层以 BBH 词排序为主），层间得分不单调，档位比较以「总体」列为准。
5. 长上下文测至 256K；512K 未测（超出 16 GB 可用内存）。检索任务（NIAH）仅在 4K/8K 执行。
6. Phi-3-mini-4k 原生上下文 4096，不适用于 16K 以上测试。

---

## 引用

```bibtex
@misc{edge_llm_quant_bench_2026,
  title  = {基于纯 CPU 边缘设备的大模型量化评估：保真度指标失效边界与跨架构实证研究},
  author = {<作者姓名>},
  year   = {2026},
  doi    = {10.5281/zenodo.23656926},
  url    = {https://github.com/chenruocheng666-source/edge-llm-quant-bench}
}
```

## 许可证

本项目代码与文档采用 **MIT License**；模型权重遵循各自原始许可（Qwen / Phi-3 / Llama）。
