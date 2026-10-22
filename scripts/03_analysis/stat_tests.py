# C:\Users\cjy13\quant_eval\04_scripts\stat_tests.py
import os
import numpy as np
import pandas as pd
from scipy import stats

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fid_csv = os.path.join(res_dir, "table3_fidelity.csv")
arc_csv = os.path.join(res_dir, "table4_downstream_arc.csv")
out_csv = os.path.join(res_dir, "table_stat_tests.csv")

# ---------- 1. KLD 的 Welch t 检验 ----------
fid = pd.read_csv(fid_csv, encoding="utf-8-sig")
fid["Quant"] = fid["Quant"].astype(str).str.upper()

# 假设 KLD 是 chunks 均值，std 来自 RMS_dp？这里需要你确认列名
# 如果 table3 里有 KLD_mean 和 KLD_std，用它们；否则只用 mean 做近似
# 此处示例：KLD_mean, KLD_std（如果没有 std，需补）
if "KLD_std" not in fid.columns:
    print("⚠️ table3 缺少 KLD_std，将用 0.01 近似，请根据实际数据调整")
    fid["KLD_std"] = 0.01

rows = []
for model in fid["Model"].unique():
    sub = fid[fid["Model"] == model]
    base = sub[sub["Quant"] == "F16"]
    if base.empty:
        continue
    base_mean = base["KLD_mean"].values[0]
    base_std = base["KLD_std"].values[0]
    for _, r in sub.iterrows():
        if r["Quant"] == "F16":
            continue
        q_mean = r["KLD_mean"]
        q_std = r["KLD_std"]
        # Welch t 检验，n 假设为 chunks（如 20）
        n1 = n2 = 20
        se = np.sqrt(base_std**2 / n1 + q_std**2 / n2)
        if se == 0:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat = (base_mean - q_mean) / se
            df_welch = (base_std**2/n1 + q_std**2/n2)**2 / (
                (base_std**2/n1)**2/(n1-1) + (q_std**2/n2)**2/(n2-1)
            )
            p_val = 2 * stats.t.sf(np.abs(t_stat), df_welch)
        rows.append({
            "Model": model,
            "Quant": r["Quant"],
            "Metric": "KLD",
            "Base_Mean": base_mean,
            "Quant_Mean": q_mean,
            "t_stat": t_stat,
            "p_value": p_val
        })

# ---------- 2. ARC 的两比例 z 检验 ----------
arc = pd.read_csv(arc_csv, encoding="utf-8-sig")
arc["Quant"] = arc["Quant"].astype(str).str.upper()

for model in arc["Model"].unique():
    sub = arc[arc["Model"] == model]
    base = sub[sub["Quant"] == "F16"]
    if base.empty:
        continue
    base_correct = base["Correct"].values[0]
    base_total = base["Total"].values[0]
    base_p = base_correct / base_total
    for _, r in sub.iterrows():
        if r["Quant"] == "F16":
            continue
        q_correct = r["Correct"]
        q_total = r["Total"]
        q_p = q_correct / q_total
        # 两比例 z 检验
        p_pool = (base_correct + q_correct) / (base_total + q_total)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/base_total + 1/q_total))
        if se == 0:
            z_stat, p_val = np.nan, np.nan
        else:
            z_stat = (base_p - q_p) / se
            p_val = 2 * stats.norm.sf(np.abs(z_stat))
        rows.append({
            "Model": model,
            "Quant": r["Quant"],
            "Metric": "ARC",
            "Base_Mean": base_p,
            "Quant_Mean": q_p,
            "t_stat": z_stat,
            "p_value": p_val
        })

# ---------- 3. FDR 校正（Benjamini-Hochberg） ----------
df = pd.DataFrame(rows)
df = df.dropna(subset=["p_value"]).copy()
df = df.sort_values("p_value").reset_index(drop=True)
m = len(df)
df["rank"] = np.arange(1, m+1)
df["q_value"] = df["p_value"] * m / df["rank"]
# 单调化
for i in range(m-2, -1, -1):
    df.loc[i, "q_value"] = min(df.loc[i, "q_value"], df.loc[i+1, "q_value"])

df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"✅ 统计检验表已保存: {out_csv}")
print(df.to_string(index=False))