# 图表说明（figures/）

本目录的 **19 张 PNG 与论文图 1–19 严格一一对应**，均直接由论文内嵌图导出，分辨率与论文一致。

| 图号 | 文件 | 内容 |
|---|---|---|
| 图1 | `fig01.png` | pp512 / tg128 双轴对比 |
| 图2 | `fig02.png` | 体积–解码速度散点 |
| 图3 | `fig03.png` | 五模型峰值内存占用 |
| 图4 | `fig04.png` | PPL / KLD / Same top-p 三联（对数轴） |
| 图5 | `fig05.png` | Qwen1.5B 关键指标归一化热力图 |
| 图6 | `fig06.png` | KLD–ARC 双轴 |
| 图7 | `fig07.png` | 跨架构 ARC 柱状 |
| 图8 | `fig08.png` | Qwen1.5B 推理得分矩阵（统一 400 题） |
| 图9 | `fig09.png` | Qwen3B 推理得分矩阵 |
| 图10 | `fig10.png` | 各档位推理总准确率 |
| 图11 | `fig11.png` | 7B/8B 保真度–下游能力折线 |
| 图12 | `fig12.png` | 7B/8B 跨架构 ARC 柱状 |
| 图13 | `fig13.png` | Correctness Agreement 热力图 |
| 图14 | `fig14.png` | 长上下文峰值 RSS（512~256K）与预填充/生成速度（4K~128K） |
| 图15 | `fig15.png` | KV Cache 消融（f16 vs q8_0） |
| 图16 | `fig16.png` | 电源模式对照 |
| 图17 | `fig17.png` | 并发吞吐 |
| 图18 | `fig18.png` | 结构化生成能力：各模型档位内容正确率（斜纹柱=N/A，OOM 未测） |
| 图19 | `fig19.png` | 随机种子稳健性：6 档位 × 5 种子（42–46）的准确率与 SD |

## v1.6.0 新增图（`figures/v1.6.0/`，统一评测协议）

| 文件 | 内容 | 对应论文修订稿 |
|---|---|---|
| `fig_v16_01_framework_flow.png` | 带适用性自检的双轨判定框架流程图 | 新图 1 |
| `fig_v16_02_protocol_effect.png` | 早期协议与统一协议下 5 个模型的 ARC-Easy 准确率对照 | 3.3 节协议对照图 |
| `fig_v16_03_format_quant_2x2.png` | Qwen2.5-0.5B 答题格式 × 量化精度 2×2 对照与选项分布 | 3.3.2 节，替代早期温度扰动图 |
| `fig_v16_04_kld_vs_downstream.png` | Qwen2.5-1.5B 的 KLD 与 ARC-Easy、多步推理随档位变化 | 替代本目录 `fig06.png`（KLD–ARC 双轴） |
| `fig_v16_05_arc_cross_arch.png` | 统一协议下跨架构 ARC-Easy 准确率与区间归属 | 替代本目录 `fig07.png`（跨架构 ARC） |
| `fig_v16_06_option_collapse.png` | Qwen2.5-0.5B 预测选项分布（单独版） | 补充材料 |

> 注：本目录 `fig06.png`、`fig07.png` 与 `fig13_noise_control.png` 基于早期 ARC-Easy 协议（仅 FP16 加约束），v1.6.0 起仅供追溯；论文终稿图号以排版后的期刊版本为准。

## 其他目录

- `_legacy/`：历史绘图产物（含旧图号体系下的 `fig04_heatmap*`、`fig08_kv_cache`、`fig12_reason_matrix_*`、`fig14_kld_triple` 等），仅作追溯，**与论文图号不对应，勿直接引用**。
- `批量输出为图片/`：绘图脚本批量导出的原始结果。

## 重新生成

```bash
python scripts/04_plotting/plot_all_final_v3.py
```

> 注：绘图脚本输出为旧图号命名，重新生成后需按上表手工对应到论文图号。
