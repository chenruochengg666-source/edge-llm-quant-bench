# 基于纯 CPU 边缘设备的大模型量化评估

> 在 16 GB 无独显轻薄本上，用 llama.cpp 对 Qwen2.5（0.5B/1.5B/3B/7B）、Phi-3-mini、Llama-3.2-1B、Llama-3.1-8B 做从 FP16 到 Q2_K 的系统量化评测，检验 **PPL/KLD 这类保真度指标能否代表下游能力**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b11223-blue)](https://github.com/ggml-org/llama.cpp)
[![Version](https://img.shields.io/badge/version-v1.6.0--unified-green)](CITATION.cff)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23656925-blue)](https://doi.org/10.5281/zenodo.23656925)

- 主仓库（Gitee）：<https://gitee.com/yfddyqcxy/edge-llm-quant-bench>
- 镜像/归档（GitHub）：<https://github.com/chenruocheng666-source/edge-llm-quant-bench>
- 论文归档（Zenodo DOI）：<https://doi.org/10.5281/zenodo.23656925>
- 对应提交：`b15729a45098b0a8a7de39ee8a52cae566dd0df8`

## 版本记录

| 版本 | 日期 | Zenodo 版本说明 | 主要变更 |
|---|---|---|---|
| **v1.6.0-unified** | 2026-11-28 | 见下方「v1.6.0 版本说明」 | 修正早期 ARC-Easy 主实验 FP16 与量化档协议不一致：15 个量化档按统一协议重跑，5 个 FP16 补齐 846 题并抽检 500 题（100% 一致）；修正数字标签题；补 0.5B、Phi-3 的 ARC-Challenge；新增答题格式 × 量化 2×2 对照、7B 子集补跑、HWiNFO RAPL 分阶段功耗补测（n=20）、`--no-repack` 内存复测（5 模型 × 3 次）；多步推理原始输出替换为表 S1 对应的重试后版本；0.5B「能力反升」更正为协议差异，改判为沉默区 |
| **v1.5.0-arcc** | 2026-11-25 | 新增 ARC-Challenge 判据复核与噪声对照实验 | 补 ARC-Challenge 全档位（4 模型 × 1172 题）作为第二判据；新增采样温度噪声对照（0.05~1.0 共 8 个温度 × 3 种子 × 2376 题）；修正「量化正则化」表述为「权重层量化误差导致的准确率反升」 |
| **v1.4.0-paper** | 2026-11-08 | 见下方「v1.4.0 版本说明」 | 补齐多步推理全量（26 档 × 400 题，含 7B/8B）；新增电源 2×2 析因与 J/token 能耗；新增表5A 双轨判定与表8 全量 CA/Kappa；修正 Extra 口径；文献按首引顺序重排至 41 条；修复 6 个脚本 BOM 与续跑缺陷 |
| v1.3.0 | 2026-10-31 | 文档与图表规范化修订（Documentation and figure normalization） | 论文表格统一为三线表；补充附录A（复现记录、实验环境与补充实验数据）；图01–图19 统一命名与归档；核心实验数据未变动 |
| v1.2.1 | 2026-10-26 | feat(p04): structured benchmark, figures, and docs sync | 80 题结构化评测（四级指标 + Extra）、图18、统计判定规则统一、README/env/docs 同步 |
| v1.1.1 | 2026-10-20 | Complete P0-1 statistical validation | 完成统计规范化：Bootstrap CI、BH-FDR、Cohen's h/d |

> 说明：v1.1.1 与 v1.2.1 的版本说明沿用早期提交信息（英文）；v1.3.0 起改为内容说明；v1.4.0 起采用中英文对照的规范化描述。


### v1.6.0 版本说明（Zenodo）

**中文：**

> 发现并修正早期 ARC-Easy 主实验中 FP16 与量化档评测协议不一致的问题（原脚本仅对 FP16 追加“只输出一个字母”约束，量化档 `max_tokens=8`，且无服务预热）：全部 15 个量化档按统一协议重跑；5 个 FP16 补齐因服务未就绪而缺失的 846 题，并抽检 500 道原有输出的题（逐字一致）；以 1~4 标注的答案统一映射为 A~D。统一协议下，Qwen2.5-0.5B 的 Q4_K_M 由“反升 14.9pp”更正为“下降 11.7pp”（保真度未报警，沉默区，选项 A 占比 73%），Llama-3.2-1B 的 Q4_K_M 由“下降 25.0pp”更正为“+2.3pp”（基准作答不稳定，判定不适用）。新增答题格式 × 量化精度 2×2 对照（自由作答下两档仅差 1.7pp，交互效应 −11.2pp）、Qwen2.5-0.5B 与 Phi-3-mini 的 ARC-Challenge 复核、7B 子集补跑，以及 HWiNFO/RAPL 分阶段功耗补测（插电与电池使用同一传感器，n=20）；`--no-repack` 复测证实 Q4_K_M 档多出的峰值内存即 CPU 重排副本（回落量与 Q4_K 张量体积之差 ≤ 0.15 MiB）。多步推理原始输出替换为与表 S1 一致的重试后版本，旧版移入 `results/legacy/`。早期协议下的 ARC-Easy 与噪声对照数据原样保留，仅供追溯。

**English:**

> Fixed an evaluation-protocol inconsistency in the earlier ARC-Easy main run (only FP16 received the "output a single letter" constraint; quantized variants were limited to `max_tokens=8`; no server warm-up). All 15 quantized variants were re-run under a unified protocol; the 846 FP16 items lost to a cold-start server were back-filled, and 500 previously answered items were re-run for audit (byte-identical outputs); numeric answer keys (1–4) were mapped to letters. Under the unified protocol, Qwen2.5-0.5B Q4_K_M changes from "+14.9 pp" to "−11.7 pp" (no fidelity alarm; silent zone, 73% of predictions on option A), and Llama-3.2-1B Q4_K_M from "−25.0 pp" to "+2.3 pp" (unstable baseline; judgement not applicable). Added a format-by-precision 2×2 control (free-form gap only 1.7 pp; interaction −11.2 pp), ARC-Challenge re-checks for Qwen2.5-0.5B and Phi-3-mini, a back-fill of the 7B subset, and phase-resolved HWiNFO/RAPL power measurements (same sensor on AC and battery, n = 20); a `--no-repack` re-measurement confirms that the extra peak memory of Q4_K_M variants is the CPU repacked copy (the drop matches the Q4_K tensor size within 0.15 MiB). Multi-step reasoning raw outputs were replaced by the retried version underlying Table S1, with the earlier version moved to `results/legacy/`. Early-protocol ARC-Easy and noise-control data are kept unchanged for traceability.

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
- [x] **P0-5 ARC-Challenge 判据复核**：1172 题；v1.6.0 扩展至 Qwen2.5-0.5B/1.5B、Phi-3-mini 三个可判定模型，7 个量化档中 6 档与 ARC-Easy 区间一致；Llama-3.2-1B 因 FP16 的 A 选项偏好（79%~86%）判定不适用
- [x] ~~**P0-6 噪声对照实验**~~：FP16 + 采样温度扰动（8 温度 × 3 种子 × 2376 题）。**v1.6.0：所对照的“能力反升”已证实为早期协议差异，本实验仅供追溯**
- [x] **P0-7 统一评测协议重测**（v1.6.0）：ARC-Easy 15 个量化档 × 2376 题；FP16 补齐 846 题、抽检 500 题 100% 一致；见 `results/summary/arc_easy_judgement.csv`、`scripts/02_evaluation/arc_unified/`
- [x] **P0-8 答题格式 × 量化 2×2 对照**（v1.6.0）：Qwen2.5-0.5B 约束作答 vs 自由作答；交互效应 −11.2pp；见 `results/summary/freeform_2x2_summary.csv`
- [x] **P0-9 RAPL 分阶段功耗补测**（v1.6.0）：2×2 电源模式 × n=20，插电与电池同一传感器；见 `results/summary/power_hwinfo_summary.csv`
- [x] **P0-10 `--no-repack` 内存复测**（v1.6.0）：5 模型 × 默认/关闭重排 × 3 次，另测 1.5B Q4_K_M 速度；见 `results/summary/rss_repack_summary.csv`
- [ ] P1-5 Needle 深度扫描
- [ ] P1-6 安全对齐
- [ ] P1-7 决策树 + 能力矩阵

---

## 核心发现（v1.6.0，统一评测协议）

1. **效率红利**：Qwen2.5-1.5B 量化至 Q4_K_M，体积 2949→940 MiB（-68.1%），生成速度 11.84→34.35 t/s（约 2.9×）；但 CPU 后端对 Q4_K 权重的在线重排（repack）使峰值内存达到文件体积的 1.83 倍（1720 MiB）。`--no-repack` 复测直接证实：关闭重排后峰值内存降至 1124 MiB（1.20 倍），5 个模型的回落量与各自 Q4_K 张量体积之差均在 0.15 MiB 以内（3B Q4_K_M 回落 1266 MiB；不含 Q4_K 的 Q5_K_M 不变）。交替 6 轮的速度复测显示，在本机 AVX2 平台上重排对预填充无可测收益（−0.9%，配对 t(5)=−1.67），反而使生成慢 6.1%（29.72 对 31.64 t/s，6 轮全部同向，t(5)=−15.3）——即默认重排在此平台上以约 600 MiB 内存换来的是更慢的解码。
2. **保真度与下游能力双向错配**：Qwen2.5-3B 的 Q4_K_M 保真度报警（ΔPPL 0.775 > 2σ_Δ 0.335）而 ARC-Easy 持平（93.18% 对 93.22%，**误报**）；Qwen2.5-0.5B 的 Q4_K_M 保真度未报警（ΔPPL 0.225 < 0.748，KLD 0.031）而 ARC-Easy 下降 **11.7pp**（h=0.235，q<0.001，**漏报 / 沉默区**）。
3. **沉默区的机理是受约束输出下的选项塌缩**：0.5B Q4_K_M 在“只输出一个字母”约束下 73% 的预测为 A（标签中 A 约占 25%）；答题格式 × 量化 2×2 对照中，约束作答两档相差 12.8pp，自由作答仅差 1.7pp（p=0.08），交互效应 **−11.2pp**（95% CI [−13.9, −8.5]）。
4. **评测协议可翻转结论**：早期主实验仅对 FP16 加约束，0.5B Q4_K_M 被报告为“反升 14.9pp”、Llama-3.2-1B Q4_K_M 为“下降 25.0pp”；统一协议后分别为 −11.7pp 与 +2.3pp。作答稳定的 Qwen2.5-1.5B/3B 与 Phi-3-mini 受影响在 1pp 以内。
5. **崩塌区与模型特异性**：Q2_K 在 1.5B、3B、Phi-3-mini 上显著退化（h = 1.150 / 2.615 / 0.806），并伴随大量无法解析的输出（470 / 2376 / 572 题）；同为 Q3_K_M，1.5B 下降 4.8pp（保真先行区）而 3B 下降 14.6pp（同步退化区）；Llama-3.2-1B 因 FP16 79% 的 A 选项偏好判定不适用；7B/8B 子集上 Qwen2.5-7B 的 Q2_K 仍保有 89.4%，Llama-3.1-8B 的 Q2_K 为 59.2%。
6. **解码受内存带宽约束**：同一 RAPL 传感器下，插电与电池最佳性能模式的解码核心频率相差 26%、封装功率相差 29%，tg128 仅差 2.6%；电池最佳能效模式使预填充 −22.3%、解码 −12.7%（交互 Welch t = 11.4 / 22.9，n=20）。长上下文下 1.5B Q4_K_M 峰值 RSS 512→262144 严格线性（**28.05 KB/token**），预填充 155.81→19.91 t/s（-87.2%）而生成 34.08→32.80 t/s（-3.8%）。

---

> **统计口径说明（v1.6.0）**：ARC 准确率在统一协议下按全部题目计算（ARC-Easy 2376 题、Challenge 1172 题），所有档位均无推理失败，数字标签已映射为字母。区间判定为 q<0.05 且 |h|≥0.2；判定前检查协议一致性、FP16 判别力与作答稳定性（FP16 单一选项占比 ≤ 50%），判定后附加“结构偏移 / 选项塌缩 / 格式失控”标注。
> `table3_ARC_*`、`table4_downstream_arc.csv`、`table5_stat_final.csv`、`table5A_zone_judgement.csv`、`table8_ca_kappa_full.csv`、`noise_control_summary.csv` 对应早期协议（按“剔除推理失败后的有效样本”计算），仅供追溯；当前结果见 `arc_easy_judgement.csv`、`arc_challenge_judgement.csv`。

## 仓库结构

```text
edge-llm-quant-bench/
├── scripts/
│   ├── 01_benchmarks/        # 性能、内存、上下文伸缩与速度
│   │   ├── measure_ctx_ext.ps1     # -c 512~262144，峰值 RSS + OOM 判定
│   │   ├── measure_ctx_speed.ps1   # llama-bench -p 4096~131072，预填充/生成速度
│   │   ├── measure_rss.ps1         # 单档峰值 RSS
│   │   ├── extract_performance.py  # llama-bench 结果解析
│   │   └── run_supplement_tests.ps1  # ★ v1.6.0 HWiNFO 分阶段功耗补测 / --no-repack 内存复测
│   ├── 02_evaluation/        # ARC-Easy / ARC-Challenge / 多步推理 / 结构化 / 种子复测 / 噪声对照
│   │   ├── arc_challenge/         # ★ v1.5.0 第二判据复核
│   │   │   ├── prepare_arc_challenge.py   # 下载 ARC-Challenge test（1172 题）
│   │   │   ├── run_arc_challenge.py       # 全档位评测（含预热/代理规避/续跑）
│   │   │   └── compare_easy_challenge.py  # 两套判据逐档对照
│   │   ├── noise_control/         # ★ v1.5.0 噪声对照实验
│   │   │   └── run_noise_control.py       # 温度梯度扰动（早期协议，仅供追溯）
│   │   ├── arc_unified/           # ★ v1.6.0 统一协议重测
│   │   │   ├── README_统一协议补充实验.md
│   │   │   └── run_arc_easy_unified.py    # ARC-Easy/Challenge 重跑、FP16 补跑与抽检、自由作答、7B 补跑
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
│   │   ├── analyze_arc_seed.py          # 种子间 SD / 逐题一致率统计
│   │   ├── analyze_arc_unified.py       # ★ v1.6.0 统一协议判定（z / BH-FDR / h / McNemar / 区间 / 标注）
│   │   ├── analyze_freeform_2x2.py      # ★ v1.6.0 答题格式 × 量化 2×2 对照
│   │   ├── mcnemar_reasoning400.py      # ★ v1.6.0 多步推理逐题配对检验
│   │   ├── analyze_power_hwinfo.py      # ★ v1.6.0 RAPL 分阶段功耗与频率汇总
│   │   └── analyze_rss_repack.py        # ★ v1.6.0 --no-repack 内存复测与 GGUF 张量类型统计
│   └── 04_plotting/          # 论文图表
├── data/                     # 题库与语料（不入库，由脚本下载/生成；见 data/README.md）
│   ├── arc/  reasoning/  wikitext2/  structured/
├── results/
│   ├── summary/              # 汇总表（table*.csv，含 table_struct_*_v2）
│   ├── raw/                  # 原始输出（arc / reasoning / kld / perf / rss / ctx / structured_v2）
│   │   ├── arc_challenge/        # ★ v1.5.0 ARC-Challenge 逐题结果（4 模型 × 1172 题）
│   │   ├── noise_control/        # 噪声对照逐题结果（早期协议，仅供追溯）
│   │   ├── arc_unified/          # ★ v1.6.0 ARC-Easy 统一协议（15 量化档 + 5 FP16，各 2376 题）
│   │   ├── arc_challenge_unified/ # ★ v1.6.0 0.5B / Phi-3 的 ARC-Challenge
│   │   ├── arc_freeform/         # ★ v1.6.0 0.5B 自由作答对照
│   │   ├── arc_7b_subset_fill/   # ★ v1.6.0 7B 子集补跑
│   │   ├── power_hwinfo/         # ★ v1.6.0 HWiNFO 日志与分阶段 llama-bench
│   │   └── rss_repack/           # ★ v1.6.0 --no-repack 内存与速度复测（详见 results/raw/README.md）
│   └── legacy/               # 历史版本结果（table_stat_tests_legacy.csv；reasoning_pre_retry/ 为多步推理重试前输出）
├── figures/                  # 论文图表 PNG（fig01–fig19；v1.6.0/ 为统一协议新图；_legacy 为历史图）
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

**v1.6.0 统一协议（推荐，脚本支持环境变量与相对路径）**：

```powershell
$env:LLAMA_CPP_BIN="D:\llama.cpp\build\bin\Release"; $env:EDGE_LLM_MODEL_DIR="D:\gguf"
python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --check     # 检查文件与预计耗时
python -u scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --plan A  # 详见该目录 README
python scripts/03_analysis/analyze_arc_unified.py [--task challenge]
python scripts/03_analysis/analyze_freeform_2x2.py
python scripts/03_analysis/mcnemar_reasoning400.py
python scripts/03_analysis/analyze_power_hwinfo.py
python scripts/03_analysis/analyze_rss_repack.py
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
| 表14、图19 种子稳健性 | `results/summary/table14_seed_stability.csv`、`table_p03_seed_anova.csv`（v1.6.0 修正 15 道数字标签题后均值约高 3pp，见论文表 14 修订） |
| ★ v1.6.0 ARC-Easy 统一协议判定（论文表 2 ARC 列、表 5、表 6、表 9） | `results/summary/arc_easy_judgement.csv`、逐题 `arc_easy_items.csv`；原始 `results/raw/arc_unified/` |
| ★ v1.6.0 ARC-Easy 与 ARC-Challenge 判据对照（论文表 3） | `results/summary/arc_challenge_judgement.csv`、`arc_challenge_items.csv`；原始 `results/raw/arc_challenge_unified/`、`arc_challenge/` |
| ★ v1.6.0 答题格式 × 量化 2×2 | `results/summary/freeform_2x2_summary.csv`、`freeform_2x2_items.csv`；原始 `results/raw/arc_freeform/` |
| ★ v1.6.0 多步推理逐题配对检验 | `results/summary/mcnemar_reasoning400.csv` |
| ★ v1.6.0 电源模式与能耗（论文表 12 修订） | `results/summary/power_hwinfo_summary.csv`；原始 `results/raw/power_hwinfo/` |
| ★ v1.6.0 7B 子集补跑 | `results/raw/arc_7b_subset_fill/` |
| ★ v1.6.0 CPU 重排内存与速度复测 | `results/summary/rss_repack_summary.csv`、`speed_repack_summary.csv`（交替 6 轮）、`gguf_tensor_types.csv`；原始 `results/raw/rss_repack/`（`speed_repack.csv` 为首批顺序测量，受顺序影响，仅供追溯） |

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

- **评测参数**：`temperature=0`、`top_p=1`、`seed=42`；ARC 所有档位统一追加“只输出一个大写字母”约束、`max_tokens=64`（v1.6.0 统一协议）；多步推理 `max_tokens=1024`；结构化生成 `256`。
- **评测协议一致性**（v1.6.0）：同一比较中的 FP16 与量化档必须使用相同的提示词、解码参数、输出上限与答案提取规则；服务启动后先预热再评测。
- **线程**：`llama-bench` 用 `-t 4`；批量下游推理与 PPL/KLD 用 `-t 8`。
- **KLD**：`--chunks 20`（n=20），以 FP16 输出 logits 为基准；`Same top-p` 报告均值。
- **统计判定**：ARC 用两比例 z 检验（效应量 Cohen's h），p 值经 BH-FDR 校正；**q<0.05 且 |效应量|≥0.2 方视为实质差异**。
- **保真度轨判定**：以 3.3 节 σ_pool 合成标准差规则（|ΔPPL|≥2σ_pool）为**主判据**；因 FP16 基准的 KLD 恒为零、方差为零，KLD 的 Welch t 检验实质退化为「KLD 是否异于零」的单样本检验，其结果为 15 档全部显著，仅作附录 A.8 的敏感性参考，不作主判据。
- **配对敏感性**：各档位在同一批 ARC-Easy 题目上作答，构成配对设计，附录 A.8 给出 McNemar 精确检验结果；正文表3 仍以边际口径（两比例 z 检验）为准。
- **区间判定脚本**：`scripts/03_analysis/calc_zone_judge.py` 复现表5A 与附录表A10，输出 `results/summary/table5A_zone_judgement.csv`。
- **人工评分**：400 题分层抽样（每档 30–50 题），双人独立盲审，加权 Cohen's κ = **0.9989**（95% CI 0.9966–1.0），Krippendorff's α = 0.9972。
- **上下文**：RSS 测 512~262144（`-c` 逐档，`n=1`）；速度测 4096~131072（`llama-bench -r 1`）；256K 速度未测（单次约 30 分钟）。
- **电源与能耗**（v1.6.0）：2×2 析因（插电/电池 × 最佳性能/最佳能效，Windows“平衡”计划，未开节电模式），`llama-bench` 分阶段测量（解码 `-p 0 -n 128`、预填充 `-p 512 -n 0`，各 `-r 20`）；功率由 HWiNFO64 以 1 s 间隔读取 CPU 封装功率（RAPL），插电与电池同一传感器；插电组另以墙插功率计校准系统总功率（偏差 ≤ 2.6 W）；净能耗 =（阶段平均功率 − 同组待机功率）/ 吞吐。脚本：`scripts/01_benchmarks/run_supplement_tests.ps1`、`scripts/03_analysis/analyze_power_hwinfo.py`。早期 n=5 批次（墙插功率计 + BatteryInfoView，二者不可比）见 `power_2x2.csv`、`table11_power_energy.csv`。
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
8. 多步推理原始输出（v1.6.0 起为经 3 次重试的新版）仍有少量空响应（每档约 5~25 题，Phi-3 FP16 为 242 题），相关结论以「双方均有效」的题目为分析集；重试前的旧版见 `results/legacy/reasoning_pre_retry/`。
9. Qwen2.5-0.5B 隐藏维度 896 不是 256 的整数倍，其「Q4_K_M」档权重以 Q5_0（44%）与 Q8_0（37%）为主，与其他模型的同名档位并非同一量化。
10. 早期 ARC-Easy 主实验与噪声对照存在 FP16 与量化档协议不一致的问题（见 `scripts/02_evaluation/arc_unified/README_统一协议补充实验.md`），相关数据仅供追溯；答题格式对照仅在 Qwen2.5-0.5B 上进行。

---

## 引用

```bibtex
@misc{edge_llm_quant_bench_2026,
  title  = {基于纯 CPU 边缘设备的大模型量化评估：保真度指标失效边界与跨架构实证研究},
  author = {陈家瑜 (Jia-yu CHEN)},
  year   = {2026},
  doi    = {10.5281/zenodo.23656925},
  url    = {https://github.com/chenruocheng666-source/edge-llm-quant-bench}
}
```

> **引用说明**：如果使用了本项目的代码或数据，请引用上述 DOI。若 Zenodo 归档的版本与最新版开源仓库有所出入，**以仓库最新版为准**（仓库持续维护，Zenodo 为阶段性快照）。
>
> **Citation notice**: If you use the code or data of this project, please cite the DOI above. Where the Zenodo-archived release differs from the latest repository version, **the latest repository version prevails** (the repository is continuously maintained, while Zenodo provides periodic snapshots).

## 许可证

本项目代码与文档采用 **MIT License**；模型权重遵循各自原始许可（Qwen / Phi-3 / Llama）。
