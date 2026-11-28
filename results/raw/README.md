# 原始输出说明（results/raw/）

| 目录 | 内容 | 协议 / 状态 |
|---|---|---|
| `arc_unified/` | ★ v1.6.0 ARC-Easy 统一协议：15 个量化档 + 5 个补齐的 FP16（各 2376 题），FP16 抽检 `*.audit.jsonl` | **当前主结果** |
| `arc_challenge_unified/` | ★ v1.6.0 ARC-Challenge：Qwen2.5-0.5B、Phi-3-mini 各 3 档 × 1172 题 | 当前 |
| `arc_freeform/` | ★ v1.6.0 Qwen2.5-0.5B FP16 / Q4_K_M 自由作答（不加约束）× 2376 题 | 当前 |
| `arc_7b_subset_fill/` | ★ v1.6.0 Qwen2.5-7B Q4_K_M 子集补跑后的完整 500 题与 30 题抽检 | 当前（替代 `arc_7b_subset/` 中的 Q4_K_M） |
| `power_hwinfo/` | ★ v1.6.0 HWiNFO64 日志、阶段时间标记、分阶段 llama-bench 结果（2×2 电源模式） | 当前 |
| `rss_repack/` | ★ v1.6.0 `--no-repack` 复测：5 模型 × 默认/关闭重排 × 3 次峰值 RSS（ctx=4096）；1.5B Q4_K_M 速度：`speed_repack_interleaved.csv` 为交替 6 轮（ABBA，每轮预热后各 3 次，主结果）；`speed_repack.csv` 为首批顺序测量（仅供追溯） | 当前；默认设置下的 RSS 与 `rss/` 相差 ≤ 1.1 MiB |
| `reasoning/` | 多步推理 400 题逐题输出（v1.6.0 起为经 3 次重试的新版，与论文表 S1 一致） | 当前 |
| `arc_challenge/` | ARC-Challenge：Qwen2.5-1.5B、Llama-3.2-1B（原协议本已对所有档位统一） | 当前（分析时修正 22 道数字标签题） |
| `arc/` | 早期 ARC-Easy 主实验 | **早期协议**：仅 FP16 加约束、量化档 max_tokens=8、无预热；仅供追溯，FP16 有效输出已并入 `arc_unified/` |
| `noise_control/` | 早期温度扰动对照（FP16 带约束） | **早期协议**：所对照的“能力反升”已证实为协议差异，仅供追溯 |
| `arc_7b_subset/`、`arc_llama8b_subset/` | 7B/8B 的 500 题子集 | 原协议；分析时修正 22 道数字标签题 |
| `kld/`、`kld_logs/`、`perf/`、`rss/`、`ctx/`、`concurrency/`、`power_log/` | 保真度、性能、内存、上下文、并发、早期电源实验 | 未变 |

多步推理重试前的旧版输出见 `results/legacy/reasoning_pre_retry/`。
