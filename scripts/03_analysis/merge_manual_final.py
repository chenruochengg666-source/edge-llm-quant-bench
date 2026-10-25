# C:\Users\cjy13\quant_eval\04_scripts\merge_manual_final.py
import pandas as pd
import os

base = r"C:\Users\cjy13\quant_eval\06_results"
manual_dir = os.path.join(base, "manual_scoring")
rater1_csv = os.path.join(manual_dir, "rater1_scored.csv")
rater2_csv = os.path.join(manual_dir, "rater2_scored.csv")
# ✅ 修复 1：读取老师填好的仲裁文件
arbitration_csv = os.path.join(manual_dir, "arbitration_scored.csv")
detail_csv = os.path.join(base, "table5_reasoning_detail.csv")
out_detail = os.path.join(base, "table5_reasoning_detail_human_final.csv")
out_summary = os.path.join(base, "table5_reasoning_final_summary.csv")

# 1. 读取评分 (为了防止 WPS/Excel 保存导致的 GBK 编码报错，这里统一使用 gbk 读取)
try:
    r1 = pd.read_csv(rater1_csv, encoding='utf-8-sig')
except UnicodeDecodeError:
    r1 = pd.read_csv(rater1_csv, encoding='gbk')

try:
    r2 = pd.read_csv(rater2_csv, encoding='utf-8-sig')
except UnicodeDecodeError:
    r2 = pd.read_csv(rater2_csv, encoding='gbk')

# 2. 合并两人评分
m = pd.merge(
    r1[["Task_ID", "Manual_Score"]],
    r2[["Task_ID", "Manual_Score"]],
    on="Task_ID",
    suffixes=("_r1", "_r2")
)

# 3. 确定最终分数
final_scores = m.copy()
final_scores["Final_Score"] = final_scores["Manual_Score_r1"]  # 默认用r1

# ✅ 修复 2：正确读取仲裁结果并覆盖
if os.path.exists(arbitration_csv):
    # 这里加上 try-except 自动兼容 utf-8 和 gbk
    try:
        arb_df = pd.read_csv(arbitration_csv, encoding='utf-8-sig')
    except UnicodeDecodeError:
        print("检测到仲裁文件为 GBK 编码（WPS 默认），正在自动切换...")
        arb_df = pd.read_csv(arbitration_csv, encoding='gbk')

    # 确保分数是浮点数，去除可能存在的空值
    arb_df["Arbitrated_Score"] = pd.to_numeric(arb_df["Arbitrated_Score"], errors='coerce')
    # ... 后面的代码保持不变
# 4. 合并回总明细表
detail = pd.read_csv(detail_csv, encoding='utf-8-sig')
# 把 detail 里原有的空 Manual_Score 列删掉，避免冲突
if "Manual_Score" in detail.columns:
    detail = detail.drop(columns=["Manual_Score"])

final_detail = pd.merge(detail, final_scores[["Task_ID", "Final_Score"]], on="Task_ID", how="left")
final_detail.rename(columns={"Final_Score": "Human_Score"}, inplace=True)

# ✅ 修复 3：只保留被抽中打分的那部分数据（剔除没打分的空行）
final_detail = final_detail.dropna(subset=["Human_Score"]).copy()
final_detail.to_csv(out_detail, index=False, encoding="utf-8-sig")
print(f"✅ 最终人工评分明细已保存: {out_detail}，共 {len(final_detail)} 条有效评分。")

# ================= 5. 生成汇总表 (你缺失的部分) =================
level_order = ["2step", "3step", "4step", "5step+"]
summary_rows = []

for (model, quant), group in final_detail.groupby(["Model", "Quant"]):
    row = {"Model": model, "Quant": quant, "Total_Score": 0.0, "Total_Questions": 0}
    for lv in level_order:
        sub = group[group["Level"] == lv]
        # 计算该层得分 (注意处理可能的空值)
        score = sub["Human_Score"].sum(skipna=True)
        total = len(sub)
        row[f"{lv}_Score"] = score
        row[f"{lv}_Total"] = total
        row["Total_Score"] += score
        row["Total_Questions"] += total

    row["Accuracy"] = row["Total_Score"] / row["Total_Questions"] if row["Total_Questions"] > 0 else 0
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
print(f"✅ 多步推理汇总表已生成: {out_summary}")
print("\n=== 汇总预览 ===")
print(summary_df.sort_values(by=["Model", "Quant"]).to_string(index=False))