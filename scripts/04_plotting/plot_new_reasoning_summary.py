# C:\Users\cjy13\quant_eval\04_scripts\plot_new_reasoning_summary.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 读取最新的人工评分汇总表
df = pd.read_csv(os.path.join(res_dir, "table5_reasoning_final_summary.csv"), encoding="utf-8-sig")

# ==========================================
# 图 1：多步推理准确率对比（展示 OOM 与量化崩塌）
# ==========================================
df['Label'] = df['Model'] + "-" + df['Quant']
# 按照模型和量化顺序排序，为了让图好看，把同模型放一起
df['Model_Size'] = df['Model'].apply(
    lambda x: 1.5 if '1.5b' in x else (3 if '3b' in x else (8 if '8b' in x or '7b' in x else 1)))
df = df.sort_values(by=['Model_Size', 'Model', 'Quant'])

fig, ax = plt.subplots(figsize=(12, 6))

# 颜色映射：F16 蓝色，Q4 绿色，Q2 红色
colors = {'F16': '#5B9BD5', 'Q4_K_M': '#70AD47', 'Q2_K': '#FF0000',
          'Q3_K_M': '#ED7D31', 'Q5_K_M': '#A5A5A5', 'Q6_K': '#A5A5A5', 'Q8_0': '#A5A5A5'}
bar_colors = [colors.get(q, '#808080') for q in df['Quant']]

bars = ax.bar(df['Label'], df['Accuracy'], color=bar_colors, edgecolor='black', alpha=0.8)

# 在柱子上打标签
for bar, acc, model, quant in zip(bars, df['Accuracy'], df['Model'], df['Quant']):
    h = bar.get_height()
    # 如果是 F16 且分数为 0，说明是 OOM 崩了
    if h == 0 and "F16" in quant:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01, 'OOM\nFailed',
                ha='center', va='bottom', color='red', fontweight='bold', fontsize=10)
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{acc * 100:.1f}%',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (Stratified Sample)', fontsize=12)
ax.set_title('Multi-step Reasoning Accuracy: Quantization & Architecture Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(df['Label'])))
ax.set_xticklabels(df['Label'], rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 0.45)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 自定义图例
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=colors['F16'], label='F16 (Baseline)'),
                   Patch(facecolor=colors['Q4_K_M'], label='Q4_K_M (Optimal)'),
                   Patch(facecolor=colors['Q2_K'], label='Q2_K (Extreme)')]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig_reasoning_accuracy_new.pdf"))
plt.close()
print("✅ 图 1 已生成: fig_reasoning_accuracy_new.pdf")

# ==========================================
# 图 2：保真度指标失效图（KLD vs 准确率 散点图）
# ==========================================
# 需要合并一下 KLD 数据
kld_csv = os.path.join(res_dir, "table_kld_summary.csv")
if os.path.exists(kld_csv):
    df_kld = pd.read_csv(kld_csv, encoding="utf-8-sig")
    df_kld["Quant"] = df_kld["Quant"].str.upper()

    # 只取有准确率和 KLD 的数据
    merged = pd.merge(df, df_kld, on=["Model", "Quant"], how="inner")

    # 排除 OOM 的数据点
    merged = merged[merged["Accuracy"] > 0]

    plt.figure(figsize=(8, 6))

    # 按模型分组画散点
    models = merged['Model'].unique()
    markers = ['o', 's', '^', 'D', 'v']

    for i, model in enumerate(models):
        sub = merged[merged['Model'] == model]
        plt.scatter(sub['Mean_KLD'], sub['Accuracy'],
                    label=model, marker=markers[i % len(markers)], s=100, alpha=0.8, edgecolors='k')

        # 标上量化档位文本
        for _, row in sub.iterrows():
            plt.annotate(row['Quant'], (row['Mean_KLD'], row['Accuracy']),
                         textcoords="offset points", xytext=(5, 5), fontsize=8, color='gray')

    plt.xlabel('KLD (KL Divergence)', fontsize=12)
    plt.ylabel('Multi-step Reasoning Accuracy', fontsize=12)
    plt.title('The Fidelity Illusion: KLD vs. Downstream Reasoning Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_fidelity_illusion.pdf"))
    plt.close()
    print("✅ 图 2 已生成: fig_fidelity_illusion.pdf")