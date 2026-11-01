# C:\Users\cjy13\quant_eval\04_scripts\plot_structured.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 读取汇总表
df = pd.read_csv(os.path.join(res_dir, "table_struct_summary.csv"), encoding="utf-8-sig")

# 按照你的实验规划，重点展示这三个档位
target_quants = ["F16", "Q4_K_M", "Q2_K"]
df = df[df["Quant"].isin(target_quants)].copy()

# 为了让图好看，排序一下模型大小
model_order = ["qwen0.5b", "ll32", "qwen1.5b", "qwen3b", "phi3", "qwen2.5-7b-instruct", "llama3.1-8b-instruct"]
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df["Quant"] = pd.Categorical(df["Quant"], categories=target_quants, ordered=True)
df = df.sort_values(["Model", "Quant"])

# 绘制分组柱状图 (只画 Schema_Valid_Rate 最能说明问题)
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=df,
    x="Model",
    y="Schema_Valid_Rate(%)",
    hue="Quant",
    palette={"F16": "#5B9BD5", "Q4_K_M": "#70AD47", "Q2_K": "#FF0000"}
)

# 在柱子上标注数值
for p in ax.patches:
    height = p.get_height()
    if pd.notna(height):
        ax.annotate(f'{height:.0f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.title("Structured Output Capability (Schema Valid Rate)", fontsize=14, fontweight='bold')
plt.ylabel("Schema Valid Rate (%)", fontsize=12)
plt.xlabel("Model Architecture", fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=20)
plt.legend(title="Quantization", loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig_struct_collapse.pdf"))
plt.close()
print("✅ 图表已生成: fig_struct_collapse.pdf")