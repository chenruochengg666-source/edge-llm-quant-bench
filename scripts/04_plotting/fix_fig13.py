# C:\Users\cjy13\quant_eval\04_scripts\fix_fig13.py
import os
import pandas as pd
import matplotlib.pyplot as plt

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(os.path.join(res_dir, "table5_reasoning_final_summary.csv"), encoding="utf-8-sig")

# 只保留有评分的档位（Total_Score > 0 或 有任意难度分数）
df = df[df["Total_Score"] > 0].copy()

# 构造标签
df["Label"] = df["Model"] + "-" + df["Quant"]

# 排序
df = df.sort_values(["Model", "Quant"])

plt.figure(figsize=(10, 5))
bars = plt.bar(df["Label"], df["Total_Score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Score (max 120)")
plt.title("Multi-step Reasoning: Total Score (Human Scoring)")

# 标数值
for b, v in zip(bars, df["Total_Score"]):
    plt.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig13_reason_total_score.pdf"))
plt.close()
print("✅ fig13 已重新生成")