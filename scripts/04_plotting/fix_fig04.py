# C:\Users\cjy13\quant_eval\04_scripts\fix_fig04.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 读取三张表
df_eng = pd.read_csv(os.path.join(res_dir, "table2_engineering.csv"), encoding="utf-8-sig")
df_fid = pd.read_csv(os.path.join(res_dir, "table3_fidelity.csv"), encoding="utf-8-sig")
df_arc = pd.read_csv(os.path.join(res_dir, "table4_downstream_arc.csv"), encoding="utf-8-sig")

# 统一 Quant 大写
for df in [df_eng, df_fid, df_arc]:
    df["Quant"] = df["Quant"].astype(str).str.upper()

# 只取 Qwen1.5B
df_e = df_eng[df_eng["Model"] == "qwen1.5b"].copy()
df_f = df_fid[df_fid["Model"] == "qwen1.5b"].copy()
df_a = df_arc[df_arc["Model"] == "qwen1.5b"].copy()

# 合并
df_h = pd.merge(df_e, df_f, on=["Model", "Quant"], how="inner")
df_h = pd.merge(df_h, df_a, on=["Model", "Quant"], how="inner")

# 量化顺序
QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
df_h["Quant"] = pd.Categorical(df_h["Quant"], categories=QUANT_ORDER, ordered=True)
df_h = df_h.sort_values("Quant").dropna(subset=["Quant"])

# 关键指标
cols = ["PPL_Quant", "KLD_mean", "Accuracy", "tg128 (t/s)", "Size (MiB)"]
available = [c for c in cols if c in df_h.columns]

# 归一化并画图
data_norm = df_h[available].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
data_norm.index = df_h["Quant"].astype(str)

plt.figure(figsize=(8, 5))
sns.heatmap(data_norm.T, annot=df_h[available].T, fmt=".2f", cmap="RdYlGn_r")
plt.title("Qwen1.5B: Key Metrics Heatmap")
plt.savefig(os.path.join(fig_dir, "fig04_heatmap.pdf"))
plt.close()
print("✅ fig04 已修复，包含列:", available)