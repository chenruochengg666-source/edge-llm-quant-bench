# 归档的早期绘图脚本

本目录保存 2026-10 期间使用的早期绘图脚本，**已不再维护**，仅供追溯历史版本。

## 为什么归档

这些脚本输出的是「语义命名」的 PDF（如 `fig08_kv_cache.pdf`、`fig13_reason_total_score.pdf`），
而论文与 `figures/` 目录使用「顺序命名」的 PNG（`fig01.png` ~ `fig19.png`），两套编号的含义完全不同——例如旧 `fig08_kv_cache.pdf` 对应现在的 **图15**，而非图8。

## 现行绘图脚本

| 论文图 | 脚本 |
|---|---|
| 图8 / 9 / 10 / 13 | `plot_reasoning_v2.py` |
| 图11 / 12 | `plot_7b8b_results.py` |
| 图18 | `plot_structured.py` |

图1–7、14–17、19 由早期脚本生成，命名与数据映射见仓库 `README.md` 的「图 ↔ 脚本对照」。
