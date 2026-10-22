# "C:\Users\cjy13\quant_eval\04_scripts\merge_manual_final.py"
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

base = r"C:\Users\cjy13\quant_eval\06_results"
manual_dir = os.path.join(base, "manual_scoring")
rater1_csv = os.path.join(manual_dir, "rater1_scored.csv")
rater2_csv = os.path.join(manual_dir, "rater2_scored.csv")
detail_csv = os.path.join(base, "table5_reasoning_detail.csv")
out_detail = os.path.join(base, "table5_reasoning_detail_human_final.csv")
out_summary = os.path.join(base, "table5_reasoning_final_summary.csv")

# ---------- 1. 读取评分 ----------
r1 = pd.read_csv(rater1_csv, encoding='utf-8-sig')
r2 = pd.read_csv(rater2_csv, encoding='utf-8-sig')

# ---------- 2. 读取 detail 并清理重复列 ----------
detail = pd.read_csv(detail_csv, encoding='utf-8-sig')

# 去掉重复列名（保留第一个）
detail = detail.loc[:, ~detail.columns.duplicated()].copy()

# 删掉 detail 里原有的空 Manual_Score，避免 merge 时加后缀
if "Manual_Score" in detail.columns:
    detail = detail.drop(columns=["Manual_Score"])

print("detail 清理后列名:", detail.columns.tolist())

key = ["Model", "Quant", "ID"]

# ---------- 3. 合并双人评分 ----------
m = pd.merge(
    r1[key + ["Manual_Score"]],
    r2[key + ["Manual_Score"]],
    on=key,
    suffixes=("_r1", "_r2")
)
m = m.dropna(subset=["Manual_Score_r1", "Manual_Score_r2"])
print(f"两轮评分都有的题目数: {len(m)}")

# ---------- 4. Kappa ----------
def map_score(s):
    if pd.isna(s):
        return None
    if s == 1.0:
        return 2
    if s == 0.5:
        return 1
    return 0

y1 = m["Manual_Score_r1"].apply(map_score)
y2 = m["Manual_Score_r2"].apply(map_score)
kappa = cohen_kappa_score(y1, y2, weights="quadratic")
print(f"双人评分 Cohen's Kappa (quadratic) = {kappa:.4f}")

# 取平均作为最终分
m["Human_Score"] = (m["Manual_Score_r1"] + m["Manual_Score_r2"]) / 2

# ---------- 5. 合并回 detail（用 Human_Score 列名） ----------
final = pd.merge(
    detail,
    m[key + ["Human_Score"]],
    on=key,
    how="left"
)
final.to_csv(out_detail, index=False, encoding="utf-8-sig")
print(f"最终人工评分明细已保存: {out_detail}")

# ---------- 6. 按难度层汇总 ----------
level_order = ["2step", "3step", "4step", "5step+"]
rows = []
for (model, quant), g in final.groupby(["Model", "Quant"]):
    row = {"Model": model, "Quant": quant}
    total_score = 0
    total_q = 0
    for lv in level_order:
        sub = g[g["Level"] == lv]
        score = sub["Human_Score"].sum(skipna=True)
        n = len(sub)
        row[f"{lv}_Score"] = score
        row[f"{lv}_Total"] = n
        total_score += score
        total_q += n
    row["Total_Score"] = total_score
    row["Total_Questions"] = total_q
    row["Accuracy"] = total_score / total_q if total_q > 0 else 0
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
print(f"最终表5汇总已保存: {out_summary}")
print(summary.to_string(index=False))