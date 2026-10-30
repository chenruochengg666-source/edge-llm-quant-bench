# 基于纯 CPU 边缘设备的大模型量化评估

> 在 16 GB 无独显轻薄本上，用 llama.cpp 对 Qwen2.5（0.5B/1.5B/3B/7B）、Phi-3-mini、Llama-3.2-1B、Llama-3.1-8B 做从 FP16 到 Q2_K 的系统量化评测，检验 **PPL/KLD 这类保真度指标能否代表下游能力**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b11223-blue)](https://github.com/ggml-org/llama.cpp)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23656926-blue)](https://doi.org/10.5281/zenodo.23656925)

- 主仓库（Gitee）：<https://gitee.com/yfddyqcxy/edge-llm-quant-bench>
- 镜像/归档（GitHub）：<https://github.com/chenruocheng666-source/edge-llm-quant-bench>
- 论文归档（Zenodo DOI）：<https://doi.org/10.5281/zenodo.23791404>
- 对应提交：`5684877b27db780e5c8abcc033364e124efd515e`

## 实验进度

- [x] **PO-1 统计检验规范化**：两比例 z（Cohen's h）+ Welch t（Cohen's d）+ BH-FDR，判定为「q<0.05 且 |效应量|≥0.2」
- [ ] PO-2 imatrix 敏感性
- [ ] P0-3 多种子复测
- [ ] P0-4 结构化生成任务
- [ ] P1-5 Needle 深度扫描
- [ ] P1-6 安全对齐
- [ ] P1-7 决策树 + 能力矩阵

---

## 核心发现

1. **效率红利**：Qwen1.5B 量化至 Q4_K_M，体积 2949→940 MiB（-68.1%），生成速度 11.84→34.35 t/s（约 2.9×）。
2. **保真度与下游能力存在系统性错配**：PPL/KLD 与 ARC-Easy 呈方向背离（部分档位准确率反升）与幅度背离（KLD 微变而下游无显著差异）；跨词表 KLD 不可比。
3. **量化正则化效应**（限定 ARC-Easy 全量）：Qwen0.5B 的 Q4_K_M 档 **+15.2pp**（q<0.001，Cohen's h=**0.311**，显著）；Qwen3B **+7.2pp**（q<0.001 但 h=**0.188**，未达 0.2 实质差异阈值，仅边缘证据）；Qwen1.5B 不显著。**该效应未在多步推理任务上复现**（统一 400 题呈单调下降）。
4. **崩塌区间统计显著**：Qwen1.5B/3B 的 Q2_K 相对 FP16，ARC 差异 q<0.001 且 h=**0.867**（1.5B）/ **2.165**（3B）；KLD 经 Welch t 检验 q<0.001、Cohen's d≥21，2σ 保守校核为 29.2σ / 101.2σ。
5. **量化安全边界具模型特异性**：Phi-3-mini 的 Q2_K 未完全崩塌（ARC 0.670，h=0.269）；Llama-3.2-1B 的 Q4_K_M 即显著退化（0.361→0.120，h=0.581），Q2_K 回升至 0.272 但 h=0.191 未达阈值；7B/8B 规模上 Q2_K 仍保有较强韧性（Qwen7B 85.00%）。
6. **长上下文瓶颈在 KV Cache**：1.5B Q4_K_M 可稳定运行至 256K，峰值 RSS 512→262144 严格线性（**28.05 KB/token**，与理论 28 KB/token 吻合），KV 占用约为权重的 8 倍；同一区间内预填充速度 155.81→19.91 t/s（**-87.2%**）而生成速度 34.08→32.80 t/s（**-3.8%**），说明长上下文瓶颈在内存带宽而非算力。

---

## 仓库结构

```text
edge-llm-quant-bench/
├── scripts/
│   ├── 01_benchmarks/        # 性能、内存、上下文伸缩与速度
│   │   ├── measure_ctx_ext.ps1     # -c 512~262144，峰值 RSS + OOM 判定
│   │   ├── measure_ctx_speed.ps1   # llama-bench -p 4096~131072，预填充/生成速度
│   │   ├── measure_rss.ps1         # 单档峰值 RSS
│   │   └── extract_performance.py  # llama-bench 结果解析
│   ├── 02_evaluation/        # ARC-Easy / 多步推理评测
│   │   ├── run_arc_full.py
│   │   └── run_reasoning_full.py
│   ├── 03_analysis/          # 统计检验、信度、汇总
│   │   ├── stat_tests_v2.py        # 两比例 z + Welch t + BH-FDR + Cohen's h/d
│   │   ├── calc_inter_rater_reliability.py
│   │   └── calc_ca_kappa_final_v2.py
│   └── 04_plotting/          # 论文图表
├── data/                     # 题库与语料（ARC 2376 / 推理 120·220·400 / wiki.test.raw）
├── results/
│   ├── summary/              # 汇总表（table*.csv）
│   └── raw/                  # 原始输出（arc / reasoning / kld / perf / rss / ctx）
├── figures/                  # 论文图表 PNG
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
| 表3 ARC-Easy（2376，含 q 值与效应量） | `results/summary/table4_downstream_arc.csv`、`table5_stat_final.csv` |
| 表4 多步推理得分 | `results/summary/table5_reasoning_final_summary.csv`、`table5_reasoning_detail.csv` |
| 表5 关键统计检验 | `results/summary/table5_stat_final.csv` |
| 表6 跨架构失败模式 | `results/summary/table4_downstream_arc.csv` |
| 表7 7B/8B 对照 | `results/summary/table7b_kld_summary.csv`、`7b_bench_*.csv`、`llama_8b_*.csv` |
| 表8 实例级一致性 | `results/summary/table7_ca_kappa_final.csv` |
| 表9 长上下文 | `results/raw/ctx/ctx_qwen1.5b_ext.csv`、`ctx_qwen1.5b_speed.csv` |
| 表10/11 KV Cache、电源模式 | `results/summary/tableA3_kv_cache.csv`、`tableA2_power.csv` |
| 双人评分信度 | `results/summary/table_inter_rater_reliability.csv` |

---

## 复现性要点

- **评测参数**：`temperature=0`、`top_p=1`、`seed=42`；推理 `max_tokens=1024`，下游批量 `256`。
- **线程**：`llama-bench` 用 `-t 4`；批量下游推理与 PPL/KLD 用 `-t 8`。
- **KLD**：`--chunks 20`（n=20），以 FP16 输出 logits 为基准；`Same top-p` 报告均值。
- **统计判定**：ARC 用两比例 z 检验（效应量 Cohen's h），KLD 用 Welch t 检验（Cohen's d），p 值经 BH-FDR 校正；**q<0.05 且 |效应量|≥0.2 方视为实质差异**。2σ 规则（σ_pool 合成）仅作保守校核。
- **人工评分**：400 题分层抽样（每档 30–50 题），双人独立盲审，加权 Cohen's κ = **0.9989**（95% CI 0.9966–1.0），Krippendorff's α = 0.9972。
- **上下文**：RSS 测 512~262144（`-c` 逐档，`n=1`）；速度测 4096~131072（`llama-bench -r 1`）；256K 速度未测（单次约 30 分钟）。
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
