# C:\Users\cjy13\quant_eval\04_scripts\kld_stat_test.py
import pandas as pd
import numpy as np
from scipy import stats

csv_path = r"C:\Users\cjy13\quant_eval\06_results\table3_fidelity.csv"
out_csv = r"C:\Users\cjy13\quant_eval\06_results\kld_significance.csv"

df = pd.read_csv(csv_path)

# 只保留有 KLD 数据的行
df = df.dropna(subset=["KLD_mean", "RMS_dp"])
# 注意：RMS_dp 是 Δp RMS，不是 KLD 的标准差。
# 如果你有 KLD 的标准差列（例如 KLD_sd），请替换。如果没有，我们用 RMS_dp 做近似。
# 注意：RMS_dp 与 KLD 标准差不同，这里仅作为演示，实际应该使用 KLD 的 SD。
# 为了严谨，我们需要从日志里提取 KLD 的标准差（± 后面的数值）。
# 如果你还没有 KLD_sd 列，请回到 extract_kld.py 把它加上。

# 假设你有一个 KLD_sd 列（如果没有，请补充提取）
if "KLD_sd" not in df.columns:
    print("⚠️ 缺少 KLD_sd 列，请先在 extract_kld.py 中补充提取 KLD 的 ± 标准差")
    # 这里暂用 RMS_dp 做示例
    df["KLD_sd"] = df["RMS_dp"] / 10  # 仅为占位，不可用于论文

results = []

# 1. 计算每个档位的 95% 置信区间
for _, row in df.iterrows():
    mean = row["KLD_mean"]
    sd = row["KLD_sd"]
    n = 20  # chunks=20
    se = sd / np.sqrt(n)
    ci_low, ci_high = stats.norm.interval(0.95, loc=mean, scale=se)
    results.append({
        "Model": row["Model"],
        "Quant": row["Quant"],
        "KLD_mean": mean,
        "KLD_sd": sd,
        "95%_CI_low": round(ci_low, 4),
        "95%_CI_high": round(ci_high, 4)
    })

# 2. 两两比较（以 Q4_K_M 与 Q2_K 为例）
for model in df["Model"].unique():
    sub = df[df["Model"] == model]
    if "Q4_K_M" in sub["Quant"].values and "Q2_K" in sub["Quant"].values:
        q4 = sub[sub["Quant"] == "Q4_K_M"].iloc[0]
        q2 = sub[sub["Quant"] == "Q2_K"].iloc[0]
        # Welch t-test
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=q4["KLD_mean"], std1=q4["KLD_sd"], nobs1=20,
            mean2=q2["KLD_mean"], std2=q2["KLD_sd"], nobs2=20,
            equal_var=False
        )
        results.append({
            "Model": model,
            "Quant": "Q4_K_M vs Q2_K",
            "KLD_mean": None,
            "KLD_sd": None,
            "95%_CI_low": None,
            "95%_CI_high": None,
            "p_value": p_val,
            "significant": "Yes" if p_val < 0.05 else "No"
        })

out_df = pd.DataFrame(results)
out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"✅ 统计检验结果已保存至: {out_csv}")
print(out_df.to_string(index=False))