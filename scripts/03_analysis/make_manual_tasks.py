# C:\Users\cjy13\quant_eval\04_scripts\make_manual_tasks.py
import pandas as pd
import os
import numpy as np

detail_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail.csv"
out_dir = r"C:\Users\cjy13\quant_eval\06_results\manual_scoring"
os.makedirs(out_dir, exist_ok=True)

print("正在读取全量明细数据...")
df = pd.read_csv(detail_csv)

# ================= 1. 分层抽样逻辑 =================
print("正在执行分层抽样（每个模型档位抽 30 题）...")
sampled_df = pd.DataFrame()
for (model, quant), group in df.groupby(["Model", "Quant"]):
    n_samples = min(30, len(group))
    # 对于特定重要档位（比如小模型的主实验），你可以调大 n_samples，比如 50
    if model == "qwen1.5b" and quant in ["F16", "Q4_K_M", "Q2_K"]:
        n_samples = min(50, len(group))

    sampled_df = pd.concat([sampled_df, group.sample(n=n_samples, random_state=42)])

print(f"✅ 抽样完成，共计 {len(sampled_df)} 条数据需人工评分。")

# ================= 2. 提取盲审列 =================
# 提取需要评分的核心列（隐藏 Model 和 Quant，保留 Source 方便评分员套用标准）
cols = ["Task_ID", "Level", "Source", "Question", "Gold", "Raw_Output"]
# ⭐️ 关键修复：这里必须用 sampled_df
eval_df = sampled_df[cols].copy()

# 增加一列供填写分数，一列供填写备注（便于仲裁）
eval_df["Manual_Score"] = ""
eval_df["Reviewer_Comment"] = ""

# ================= 3. 生成两份打乱顺序的盲审表 =================
np.random.seed(42)
rater1_df = eval_df.sample(frac=1).reset_index(drop=True)

np.random.seed(2026)  # 不同的种子，保证两人看到的顺序不同
rater2_df = eval_df.sample(frac=1).reset_index(drop=True)

rater1_path = os.path.join(out_dir, "rater1.csv")
rater2_path = os.path.join(out_dir, "rater2.csv")

rater1_df.to_csv(rater1_path, index=False, encoding="utf-8-sig")
rater2_df.to_csv(rater2_path, index=False, encoding="utf-8-sig")

print(f"✅ 盲审任务已生成！")
print(f" - Rater 1 表格: {rater1_path}")
print(f" - Rater 2 表格: {rater2_path}")