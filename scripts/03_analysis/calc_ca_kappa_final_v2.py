# C:\Users\cjy13\quant_eval\04_scripts\calc_ca_kappa_final_v2.py
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

base = r"C:\Users\cjy13\quant_eval\06_results"
detail_csv = os.path.join(base, "table5_reasoning_detail_human_final.csv")
out_csv = os.path.join(base, "table7_ca_kappa_final.csv")

df = pd.read_csv(detail_csv, encoding='utf-8-sig')

# 统一量化档位的命名规范
df["Quant"] = df["Quant"].astype(str).str.upper()
df["Quant"] = df["Quant"].str.replace(r'.*-', '', regex=True)

models = df["Model"].unique()
print(f"检测到以下模型: {models}")

quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
THRESHOLD = 1.0

records = []
for model in models:
    fp16 = df[(df["Model"] == model) & (df["Quant"] == "F16")]
    if fp16.empty:
        continue

    fp16 = fp16.dropna(subset=["Human_Score"])
    # ✅ 核心修复：用 ID 来匹配同一道题，而不是 Task_ID
    fp16_dict = dict(zip(fp16["ID"], fp16["Human_Score"]))

    for quant in quants:
        qdf = df[(df["Model"] == model) & (df["Quant"] == quant)].dropna(subset=["Human_Score"])
        if qdf.empty:
            continue

        # ✅ 核心修复：同理，这里也用 ID
        quant_dict = dict(zip(qdf["ID"], qdf["Human_Score"]))

        # 寻找两个档位都做过的同一批题目
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

        # 只有当两边都有对有错时，Kappa 才有意义
        if len(set(y_fp16)) > 1 and len(set(y_quant)) > 1:
            kappa = cohen_kappa_score(y_fp16, y_quant)
            kappa_str = round(kappa, 4)
        else:
            kappa_str = "N/A"

        records.append({
            "Model": model,
            "Quant": quant,
            "Total_Questions": total,
            "FP16_Accuracy": round(fp16_ok_n / total, 4) if total > 0 else 0,
            "Quant_Accuracy": round(quant_ok_n / total, 4) if total > 0 else 0,
            "Correctness_Agreement": round(ca, 4),
            "Cohen_Kappa": kappa_str
        })

columns = ["Model", "Quant", "Total_Questions", "FP16_Accuracy", "Quant_Accuracy", "Correctness_Agreement",
           "Cohen_Kappa"]
out = pd.DataFrame(records, columns=columns)

if not out.empty:
    out = out.sort_values(by=["Model", "Quant"]).reset_index(drop=True)

out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 表7（模型间一致性）已保存，共 {len(out)} 行数据。")
print(out.to_string(index=False))