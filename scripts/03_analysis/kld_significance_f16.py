# C:\Users\cjy13\quant_eval\04_scripts\kld_significance_f16.py
import os
import numpy as np
import pandas as pd
from scipy import stats

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fid_csv = os.path.join(res_dir, "table3_fidelity.csv")
out_csv = os.path.join(res_dir, "kld_significance_f16.csv")

# 假设每个档位有 20 个 chunk（你已确认 chunks=20）
N_CHUNKS = 20

df = pd.read_csv(fid_csv, encoding="utf-8-sig")
df["Quant"] = df["Quant"].astype(str).str.upper()

# 检查 F16 行是否存在
models = df["Model"].unique()
for m in models:
    if "F16" not in df[df["Model"] == m]["Quant"].values:
        print(f"⚠️ {m} 缺少 F16 行，请先补 table3_fidelity.csv")

QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

rows = []
for model in models:
    sub = df[df["Model"] == model].copy()
    base = sub[sub["Quant"] == "F16"]
    if base.empty:
        continue

    base_mean = base["KLD_mean"].values[0]
    base_sd = base["KLD_sd"].values[0]
    base_n = N_CHUNKS

    for q in QUANT_ORDER:
        qrow = sub[sub["Quant"] == q]
        if qrow.empty:
            continue
        q_mean = qrow["KLD_mean"].values[0]
        q_sd = qrow["KLD_sd"].values[0]
        q_n = N_CHUNKS

        # 95% CI（正态近似）
        se_q = q_sd / np.sqrt(q_n)
        ci_low = q_mean - 1.96 * se_q
        ci_high = q_mean + 1.96 * se_q

        # Welch t-test：F16 vs 量化档位
        if base_sd == 0 and q_sd == 0:
            t_stat, p_val = np.nan, np.nan
        else:
            se_diff = np.sqrt(base_sd**2 / base_n + q_sd**2 / q_n)
            if se_diff == 0:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat = (base_mean - q_mean) / se_diff
                df_w = (base_sd**2/base_n + q_sd**2/q_n)**2 / (
                    (base_sd**2/base_n)**2/(base_n-1)
                    + (q_sd**2/q_n)**2/(q_n-1)
                )
                p_val = 2 * stats.t.sf(np.abs(t_stat), df_w)

        rows.append({
            "Model": model,
            "Quant": q,
            "KLD_mean": q_mean,
            "KLD_sd": q_sd,
            "95%_CI_low": round(ci_low, 4),
            "95%_CI_high": round(ci_high, 4),
            "t_stat_vs_F16": round(t_stat, 4) if not pd.isna(t_stat) else "",
            "p_value_vs_F16": p_val,
        })

# FDR 校正（Benjamini-Hochberg）
out_df = pd.DataFrame(rows).dropna(subset=["p_value_vs_F16"]).copy()
out_df = out_df.sort_values("p_value_vs_F16").reset_index(drop=True)
m = len(out_df)
out_df["rank"] = np.arange(1, m+1)
out_df["q_value_vs_F16"] = out_df["p_value_vs_F16"] * m / out_df["rank"]
for i in range(m-2, -1, -1):
    out_df.loc[i, "q_value_vs_F16"] = min(
        out_df.loc[i, "q_value_vs_F16"],
        out_df.loc[i+1, "q_value_vs_F16"]
    )

out_df["significant_vs_F16"] = np.where(
    out_df["q_value_vs_F16"] < 0.05, "Yes", "No"
)

out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"✅ 已保存: {out_csv}")
print(out_df[[
    "Model", "Quant", "KLD_mean", "KLD_sd",
    "95%_CI_low", "95%_CI_high",
    "t_stat_vs_F16", "p_value_vs_F16",
    "q_value_vs_F16", "significant_vs_F16"
]].to_string(index=False))