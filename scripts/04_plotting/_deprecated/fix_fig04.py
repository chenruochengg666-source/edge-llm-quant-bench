# C:\Users\cjy13\quant_eval\04_scripts\fix_fig04.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 读取四张表
df_eng = pd.read_csv(os.path.join(res_dir, "table2_engineering.csv"), encoding="utf-8-sig")
df_fid = pd.read_csv(os.path.join(res_dir, "table3_fidelity.csv"), encoding="utf-8-sig")
df_arc = pd.read_csv(os.path.join(res_dir, "table4_downstream_arc.csv"), encoding="utf-8-sig")
df_reason = pd.read_csv(os.path.join(res_dir, "table5_reasoning_final_summary.csv"), encoding="utf-8-sig")

# 统一 Quant 大写
for df in [df_eng, df_fid, df_arc, df_reason]:
    df["Quant"] = df["Quant"].astype(str).str.upper()

# 只取 Qwen1.5B
df_e = df_eng[df_eng["Model"] == "qwen1.5b"].copy()
df_f = df_fid[df_fid["Model"] == "qwen1.5b"].copy()
df_a = df_arc[df_arc["Model"] == "qwen1.5b"].copy()
df_r = df_reason[df_reason["Model"] == "qwen1.5b"].copy()

# 区分列名，防止冲突
if "Accuracy" in df_a.columns:
    df_a = df_a.rename(columns={"Accuracy": "ARC_Acc"})
if "Accuracy" in df_r.columns:
    df_r = df_r.rename(columns={"Accuracy": "Reason_Acc"})

# 合并
df_h = pd.merge(df_e, df_f, on=["Model", "Quant"], how="inner")
df_h = pd.merge(df_h, df_a[["Model", "Quant", "ARC_Acc"]], on=["Model", "Quant"], how="left")
df_h = pd.merge(df_h, df_r[["Model", "Quant", "Reason_Acc"]], on=["Model", "Quant"], how="left")

# 为了保证 F16 正常显示，如果 F16 缺某些数据，补上默认值
# (通常 table5 不包含 F16 的汇总，如果你没有把它加进去，这里会显示为空)
# 手动补充 F16 的 Reason_Acc，根据你之前的数据，Qwen1.5B F16 得分为 17/50 = 34%
f16_idx = df_h[df_h["Quant"] == "F16"].index
if len(f16_idx) > 0:
    df_h.loc[f16_idx, "Reason_Acc"] = 0.34

# 量化顺序
QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
df_h["Quant"] = pd.Categorical(df_h["Quant"], categories=QUANT_ORDER, ordered=True)
df_h = df_h.sort_values("Quant").dropna(subset=["Quant"])

# 重命名列名，让图表看起来更专业
rename_dict = {
    "PPL_Quant": "PPL",
    "KLD_mean": "KLD",
    "ARC_Acc": "ARC-Easy Acc",
    "Reason_Acc": "Reasoning Acc",
    "tg128 (t/s)": "Decode Speed (t/s)",
    "Size (MiB)": "Model Size (MiB)"
}
df_h = df_h.rename(columns=rename_dict)

cols = ["PPL", "KLD", "ARC-Easy Acc", "Reasoning Acc", "Decode Speed (t/s)", "Model Size (MiB)"]
available = [c for c in cols if c in df_h.columns]

# 归一化
data_norm = df_h[available].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
data_norm.index = df_h["Quant"].astype(str)

# 准确率反转（因为热力图是绿色代表数值小，我们希望准确率越高越绿，所以反向处理）
data_norm_display = data_norm.copy()
if "ARC-Easy Acc" in data_norm_display.columns:
    data_norm_display["ARC-Easy Acc"] = 1 - data_norm_display["ARC-Easy Acc"]
if "Reasoning Acc" in data_norm_display.columns:
    data_norm_display["Reasoning Acc"] = 1 - data_norm_display["Reasoning Acc"]

plt.figure(figsize=(10, 6))
# 画图时 annot 依然用原始数据
sns.heatmap(data_norm_display.T, annot=df_h[available].T, fmt=".3f", cmap="RdYlGn_r")
plt.title("Qwen1.5B: Key Metrics Heatmap (with Reasoning)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig04_heatmap.pdf"))
plt.close()
print("✅ fig04 已修复，包含列:", available)