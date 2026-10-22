# C:\Users\cjy13\quant_eval\04_scripts\calc_ca_kappa_final_v2
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

base = r"C:\Users\cjy13\quant_eval\06_results"
detail_csv = os.path.join(base, "table5_reasoning_detail_human_final.csv")
out_csv = os.path.join(base, "table7_ca_kappa_final.csv")

df = pd.read_csv(detail_csv, encoding='utf-8-sig')

# 只保留有 Human_Score 的行
df = df.dropna(subset=["Human_Score"]).copy()

models = ["qwen1.5b", "qwen3b", "phi3", "ll32"]
quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
fp16_label = "f16"
THRESHOLD = 1.0  # 1.0 才算完全正确

records = []
for model in models:
    fp16 = df[(df["Model"] == model) & (df["Quant"] == fp16_label)]
    if fp16.empty:
        print(f"⚠️ {model} 没有 FP16 人工评分，跳过")
        continue
    fp16_dict = dict(zip(fp16["ID"], fp16["Human_Score"]))

    for quant in quants:
        qdf = df[(df["Model"] == model) & (df["Quant"] == quant)]
        if qdf.empty:
            continue
        quant_dict = dict(zip(qdf["ID"], qdf["Human_Score"]))

        common = sorted(set(fp16_dict) & set(quant_dict))
        if not common:
            continue

        y_fp16, y_quant = [], []
        both = fp16_ok_n = quant_ok_n = 0

        for qid in common:
            f = 1 if fp16_dict[qid] >= THRESHOLD else 0
            q = 1 if quant_dict[qid] >= THRESHOLD else 0
            y_fp16.append(f)
            y_quant.append(q)
            fp16_ok_n += f
            quant_ok_n += q
            if f == 1 and q == 1:
                both += 1

        total = len(common)
        ca = both / total if total else 0

        # Kappa 只在两类都出现时才有定义
        if len(set(y_fp16)) > 1 and len(set(y_quant)) > 1:
            kappa = cohen_kappa_score(y_fp16, y_quant)
        else:
            kappa = float('nan')

        records.append({
            "Model": model,
            "Quant": quant,
            "Total_Questions": total,
            "FP16_Accuracy": round(fp16_ok_n / total, 4),
            "Quant_Accuracy": round(quant_ok_n / total, 4),
            "Correctness_Agreement": round(ca, 4),
            "Cohen_Kappa": round(kappa, 4) if not pd.isna(kappa) else "N/A"
        })

out = pd.DataFrame(records)
out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"表7最终版已保存: {out_csv}")
print(out.to_string(index=False))