# 图表说明（figures/）

本目录的 **17 张 PNG 与论文图 1–17 严格一一对应**，均直接由论文内嵌图导出，分辨率与论文一致。

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

## 其他目录

- `_legacy/`：历史绘图产物（含旧图号体系下的 `fig04_heatmap*`、`fig08_kv_cache`、`fig12_reason_matrix_*`、`fig14_kld_triple` 等），仅作追溯，**与论文图号不对应，勿直接引用**。
- `批量输出为图片/`：绘图脚本批量导出的原始结果。

## 重新生成

```bash
python scripts/04_plotting/plot_all_final_v3.py
```

> 注：绘图脚本输出为旧图号命名，重新生成后需按上表手工对应到论文图号。
