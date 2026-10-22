# C:\Users\cjy13\quant_eval\04_scripts\plot_7b8b_results.py
import os
import matplotlib.pyplot as plt
import numpy as np

# ================= 字体与全局配置 =================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 输出目录
OUT_DIR = r"C:\Users\cjy13\quant_eval\06_results\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ================= 数据配置（硬编码你的实测数据） =================
# 注意：因为没有跑 FP16 的 ARC，X 轴我们只画 Q4_K_M 和 Q2_K
quants = ['Q4_K_M', 'Q2_K']

# Qwen 7B 数据
qwen_kld = [0.1249, 1.2960]
qwen_arc = [0.8580, 0.8500]  # 85.80% 和 85.00%

# Llama 8B 数据
llama_kld = [0.1905, 1.3089]
llama_arc = [0.7620, 0.5380]  # 76.20% 和 53.80%


# ================= 图 A：双轴折线图 (KLD vs ARC) =================
def plot_fig_a():
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左轴：KLD
    ax1.set_xlabel('Quantization Level', fontsize=12)
    ax1.set_ylabel('KLD', color='tab:red', fontsize=12)
    ax1.plot(quants, qwen_kld, 'o-', color='tab:red', label='Qwen7B KLD', linewidth=2, markersize=8)
    ax1.plot(quants, llama_kld, 's--', color='tab:orange', label='Llama8B KLD', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # 为了防止两条KLD线重叠，可以稍微设置一下Y轴范围（可选）
    ax1.set_ylim(0, 1.5)

    # 右轴：ARC Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('ARC-Easy Accuracy', color='tab:blue', fontsize=12)
    ax2.plot(quants, qwen_arc, 'o-', color='tab:blue', label='Qwen7B ARC', linewidth=2, markersize=8)
    ax2.plot(quants, llama_arc, 's--', color='tab:cyan', label='Llama8B ARC', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0.4, 1.0)

    # 添加数值标注
    for i, txt in enumerate(qwen_arc):
        ax2.annotate(f'{txt * 100:.1f}%', (quants[i], qwen_arc[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', color='tab:blue', fontweight='bold')
    for i, txt in enumerate(llama_arc):
        ax2.annotate(f'{txt * 100:.1f}%', (quants[i], llama_arc[i]), textcoords="offset points", xytext=(0, -15),
                     ha='center', color='tab:cyan', fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('7B/8B Scale: KLD vs ARC-Easy (Cross-Architecture)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    out_path = os.path.join(OUT_DIR, "fig_7b8b_kld_vs_arc.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 图 A 已生成: {out_path}")


# ================= 图 B：跨架构 ARC 对比柱状图 =================
def plot_fig_b():
    x = np.arange(len(quants))  # 标签位置
    width = 0.35  # 柱子的宽度

    fig, ax = plt.subplots(figsize=(8, 5))

    # 画柱状图
    rects1 = ax.bar(x - width / 2, qwen_arc, width, label='Qwen7B', color='#5B9BD5')
    rects2 = ax.bar(x + width / 2, llama_arc, width, label='Llama8B', color='#ED7D31')

    # 添加标签和标题
    ax.set_ylabel('ARC-Easy Accuracy', fontsize=12)
    ax.set_title('Cross-Architecture ARC-Easy Accuracy (7B/8B)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quants, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # 在柱子上标数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height * 100:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    out_path = os.path.join(OUT_DIR, "fig_7b8b_arc_compare.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 图 B 已生成: {out_path}")


if __name__ == "__main__":
    plot_fig_a()
    plot_fig_b()
    print("🎉 所有图像绘制完毕，已保存至 06_results/figures 目录。")