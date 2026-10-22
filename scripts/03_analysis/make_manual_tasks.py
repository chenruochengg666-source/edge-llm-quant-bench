# C:\Users\cjy13\quant_eval\04_scripts\make_manual_tasks.py
import pandas as pd
import os

# 改这里：你的 detail.csv 路径
detail_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail.csv"
out_dir = r"C:\Users\cjy13\quant_eval\06_results\manual_scoring"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(detail_csv)

# 你要评分的档位，先按最关键的来
targets = [
    ("qwen1.5b", "f16"),
    ("qwen1.5b", "Q4_K_M"),
    ("qwen1.5b", "Q3_K_M"),
    ("qwen1.5b", "Q2_K"),
    ("qwen3b", "f16"),
    ("qwen3b", "Q4_K_M"),
    ("qwen3b", "Q2_K"),
    ("phi3", "f16"),
    ("phi3", "Q4_K_M"),
    ("phi3", "Q2_K"),
    ("ll32", "f16"),
    ("ll32", "Q4_K_M"),
    ("ll32", "Q2_K"),
]

mask = df.set_index(["Model", "Quant"]).index.isin(targets)
sub = df[mask].copy()

# 只保留评分需要的列
cols = ["Model", "Quant", "ID", "Level", "Gold", "Pred", "Raw_Output"]
sub = sub[cols].copy()
sub["Manual_Score"] = ""  # 留给同学填：0 / 0.5 / 1.0

# 生成两个独立评分员文件
for rater in ["rater1", "rater2"]:
    out_path = os.path.join(out_dir, f"{rater}.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已生成 {out_path}，共 {len(sub)} 条")