# C:\Users\cjy13\quant_eval\04_scripts\kld_practical_significance.py
import os
import pandas as pd

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
in_csv = os.path.join(res_dir, "kld_significance_f16.csv")
out_csv = os.path.join(res_dir, "kld_significance_f16_v2.csv")

df = pd.read_csv(in_csv, encoding="utf-8-sig")

def judge(kld):
    if pd.isna(kld):
        return "N/A"
    if kld < 0.05:
        return "Near-baseline"
    if kld < 0.5:
        return "Moderate"
    return "Collapse"

df["Practical_Significance"] = df["KLD_mean"].apply(judge)

df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"✅ 已保存: {out_csv}")
print(df[["Model", "Quant", "KLD_mean", "same_top_p" if "same_top_p" in df.columns else "KLD_mean",
          "p_value_vs_F16", "significant_vs_F16", "Practical_Significance"]].to_string(index=False))