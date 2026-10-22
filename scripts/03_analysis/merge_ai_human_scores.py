# C:\Users\cjy13\quant_eval\04_scripts\merge_ai_human_scores.py
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ai_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail_AI_final.csv"
human_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail_human.csv"
out_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_final.csv"

# 读取两个文件
df_ai = pd.read_csv(ai_csv)
df_human = pd.read_csv(human_csv)

# 检查必要列是否存在
assert "AI_Manual_Score" in df_ai.columns, "AI 文件里缺少 AI_Manual_Score 列"
assert "Manual_Score" in df_human.columns, "人工文件里缺少 Manual_Score 列"

# 合并（按 Model, Quant, ID 对齐）
merged = pd.merge(
    df_ai[["Model", "Quant", "ID", "Level", "Gold", "Pred", "AI_Manual_Score"]],
    df_human[["Model", "Quant", "ID", "Manual_Score"]],
    on=["Model", "Quant", "ID"],
    how="inner"
)

# 计算 AI 与人工评分的一致性（Cohen's Kappa）
# 注意：Kappa 需要评分为整数或类别，如果你的评分是 0/0.5/1.0，可以先转换成字符串
kappa = cohen_kappa_score(
    merged["AI_Manual_Score"].astype(str),
    merged["Manual_Score"].astype(str)
)
print(f"AI 与人工评分之间的 Cohen's Kappa = {kappa:.4f}")

# 取平均作为最终得分
merged["Final_Score"] = (merged["AI_Manual_Score"] + merged["Manual_Score"]) / 2

# 按模型和档位汇总
final_summary = merged.groupby(["Model", "Quant"]).agg(
    Total_Score=("Final_Score", "sum"),
    Total_Questions=("Final_Score", "count"),
    Avg_Score=("Final_Score", "mean")
).reset_index()

final_summary["Accuracy"] = final_summary["Avg_Score"].round(4)

final_summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"✅ 最终表5数据已保存至: {out_csv}")
print(final_summary.to_string(index=False))