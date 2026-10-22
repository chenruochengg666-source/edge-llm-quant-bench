# -*- coding: utf-8 -*-
r"""
读取指定目录下的 CSV 文件，绘制各模型不同量化方式的峰值内存占用对比图。
支持：统一纵轴、柱顶加单位、体积对照线、3+2布局。
CSV 目录：C:\Users\cjy13\quant_eval\05_raw\rss
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- 配置 -----------------------------
CSV_DIR = r"C:\Users\cjy13\quant_eval\05_raw\rss"
OUT_FILE = 'memory_usage_comparison.png'

# 量化方式排序
QUANT_ORDER = ['Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'f16', 'FP16']

# 模型排序（按规模由小到大），第一行3个，第二行2个
MODEL_ORDER = ['qwen0.5b', 'll32', 'qwen1.5b', 'qwen3b', 'phi3']

# 英文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 兜底体积数据：如果 CSV 中没有 Size (MiB) 列，就用这个字典
DISK_SIZE_MIB = {
    ('qwen0.5b', 'Q2_K'): 322.92, ('qwen0.5b', 'Q4_K_M'): 379.38, ('qwen0.5b', 'f16'): 948.1,
    ('qwen1.5b', 'Q2_K'): 644.97, ('qwen1.5b', 'Q3_K_M'): 786.0, ('qwen1.5b', 'Q4_K_M'): 940.37,
    ('qwen1.5b', 'Q5_K_M'): 1072.93, ('qwen1.5b', 'Q6_K'): 1213.78, ('qwen1.5b', 'Q8_0'): 1570.29,
    ('qwen1.5b', 'f16'): 2950.35,
    ('qwen3b', 'Q2_K'): 1215.7, ('qwen3b', 'Q3_K_M'): 1516.8, ('qwen3b', 'Q4_K_M'): 1840.5,
    ('qwen3b', 'f16'): 5892.1,
    ('ll32', 'Q2_K'): 553.96, ('ll32', 'Q4_K_M'): 770.28, ('ll32', 'f16'): 2364.73,
    ('phi3', 'Q2_K'): 1438.35, ('phi3', 'Q4_K_M'): 2285.74, ('phi3', 'f16'): 7289.22
}


# ----------------------------- 读取数据 -----------------------------
def load_all_csv(csv_dir):
    """读取目录下所有 CSV，筛选包含 Peak_RSS_MiB 的文件"""
    all_data = []
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"目录下没有找到 CSV 文件: {csv_dir}")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
            if "Peak_RSS_MiB" in df.columns and "Model" in df.columns:
                all_data.append(df)
            else:
                print(f"[跳过] 缺少必要列: {os.path.basename(file)}")
        except Exception as e:
            print(f"[跳过] 读取失败 {os.path.basename(file)}: {e}")

    if not all_data:
        raise ValueError("未找到包含 'Peak_RSS_MiB' 和 'Model' 列的 CSV 文件。")
    return pd.concat(all_data, ignore_index=True)


def parse_model_name(model_str):
    """解析 'qwen1.5b-Q4_K_M' -> ('qwen1.5b', 'Q4_K_M')"""
    model_str = str(model_str).strip()
    if '-' not in model_str:
        return model_str, 'unknown'
    return model_str.split('-', 1)


# ----------------------------- 绘图 -----------------------------
def main():
    # 1. 加载数据
    df = load_all_csv(CSV_DIR)
    df[['ModelName', 'Quant']] = df['Model'].apply(lambda x: pd.Series(parse_model_name(x)))

    models = [m for m in MODEL_ORDER if m in df['ModelName'].unique()]
    if not models:
        models = sorted(df['ModelName'].unique())

    # 2. 创建画布，调整为紧凑布局，消除右侧留白
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # 统一颜色和样式
    BAR_COLOR = '#5B9BD5'   # 清爽的蓝色
    LINE_COLOR = '#E74C3C'  # 醒目的红色，与蓝色柱子区分开
    TEXT_COLOR = '#333333'

    for idx, model in enumerate(models):
        ax = axes[idx]
        sub = df[df['ModelName'] == model].copy()

        # 排序量化方式
        sub['Quant'] = pd.Categorical(
            sub['Quant'],
            categories=QUANT_ORDER + [q for q in sub['Quant'].unique() if q not in QUANT_ORDER],
            ordered=True
        )
        sub = sub.sort_values('Quant')

        quant_labels = sub['Quant'].astype(str).tolist()
        rss_vals = sub['Peak_RSS_MiB'].tolist()

        # 绘制 RSS 柱状图，设定 zorder=2
        bars = ax.bar(quant_labels, rss_vals, color=BAR_COLOR, width=0.5,
                      label='Peak RSS', zorder=2, alpha=0.85)

        # 柱顶加单位标注（字体稍小，避免拥挤）
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.1f} MiB',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

        # 体积对照线：提取数据
        size_vals = []
        for q in quant_labels:
            size = DISK_SIZE_MIB.get((model, q))
            if size is None and 'Size (MiB)' in sub.columns:
                size = sub[sub['Quant'].astype(str) == q]['Size (MiB)'].values[0]
            size_vals.append(size)

        # 绘制体积折线，设定 zorder=3 确保显示在柱子上方
        if any(v is not None for v in size_vals):
            ax.plot(quant_labels, size_vals, color=LINE_COLOR, linestyle='--',
                    marker='o', markersize=5, linewidth=2,
                    label='Disk Size', zorder=3)

        # 统一纵轴范围，优化Y轴刻度线
        ax.set_ylim(0, 8500)
        ax.set_yticks(range(0, 8500, 2000))  # 0, 2000, 4000, 6000, 8000

        # 标题与坐标轴美化
        ax.set_title(f'{model} Peak Memory Usage', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Quantization', fontsize=10, labelpad=5)
        ax.set_ylabel('Memory Usage (MiB)', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # 调整X轴标签，字号适当缩小防止拥挤
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # 将图例移到左上角，避免遮挡右侧较高的柱子和折线（只在第一个子图显示）
        if idx == 0:
            ax.legend(frameon=True, loc='upper left', fontsize=9,
                      facecolor='white', edgecolor='#cccccc')

    # 隐藏多余的第六个格子（右下角）
    for ax in axes[len(models):]:
        ax.set_visible(False)

    # 全局标题
    fig.suptitle('Peak Memory Usage vs Disk Size across Quantization Methods (Unified Y-axis 0–8500 MiB)',
                 fontsize=16, fontweight='bold')

    # 输出图片
    fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    print(f"[OK] 图已保存 -> {os.path.abspath(OUT_FILE)}")
    plt.show()


if __name__ == '__main__':
    main()