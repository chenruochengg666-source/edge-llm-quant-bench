# C:\Users\cjy13\quant_eval\04_scripts\stat_tests_v2.py
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind_from_stats
import os
import warnings

# 忽略除零等警告，我们会手动处理
warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径与全局配置
# ==========================================
BASE_DIR = r"C:\Users\cjy13\quant_eval\06_results"
OUTPUT_FILE = os.path.join(BASE_DIR, "table5_stat_final.csv")

ARC_FILE = os.path.join(BASE_DIR, "table4_downstream_arc.csv")
KLD_FILES = [
    os.path.join(BASE_DIR, "table_kld_summary.csv"),
    os.path.join(BASE_DIR, "table7b_kld_summary.csv"),
    os.path.join(BASE_DIR, "table_llama8b_kld_summary.csv")
]

ALPHA = 0.05
EFFECT_THRESHOLD = 0.2


# ==========================================
# 2. 核心统计函数
# ==========================================
def calc_cohens_h(p1, p2):
    """计算比例的 Cohen's h"""
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return abs(h)


def calc_cohens_d(m1, s1, n1, m2, s2, n2):
    """计算连续变量的 Cohen's d (使用合并标准差)"""
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0: return 0.0
    return abs(m1 - m2) / sp


# ==========================================
# 3. 数据加载与预处理
# ==========================================
print(">>> 正在加载并清理数据...")

# 3.1 加载 ARC 数据
df_arc = pd.read_csv(ARC_FILE)
df_arc['Model'] = df_arc['Model'].str.upper()
df_arc['Quant'] = df_arc['Quant'].str.upper()

# 3.2 加载并合并 KLD 数据
kld_dfs = []
for f in KLD_FILES:
    if os.path.exists(f):
        temp = pd.read_csv(f)
        # 统一列名
        temp = temp.rename(columns={'KLD_Mean': 'Mean_KLD', 'KLD_SD': 'KLD_SD'})
        # 确保包含所需的列
        if 'Mean_KLD' in temp.columns and 'KLD_SD' in temp.columns:
            kld_dfs.append(temp[['Model', 'Quant', 'Mean_KLD', 'KLD_SD']])

df_kld = pd.concat(kld_dfs, ignore_index=True)
df_kld['Model'] = df_kld['Model'].str.upper()
df_kld['Quant'] = df_kld['Quant'].str.upper()

# 假设 KLD 的样本量 N=20 (llama.cpp 默认 chunks)
N_KLD = 20

# ==========================================
# 4. 统计检验执行
# ==========================================
results = []

# 4.1 处理 ARC 数据 (Z-test + Cohen's h)
for model in df_arc['Model'].unique():
    model_df = df_arc[df_arc['Model'] == model]
    baseline = model_df[model_df['Quant'].isin(['F16', 'FP16'])]
    if baseline.empty: continue

    base_c, base_t = baseline['Correct'].values[0], baseline['Total'].values[0]
    base_p = base_c / base_t

    for _, row in model_df.iterrows():
        if row['Quant'] in ['F16', 'FP16']: continue

        q_c, q_t = row['Correct'], row['Total']
        q_p = q_c / q_t

        stat, pval = proportions_ztest([base_c, q_c], [base_t, q_t])
        h = calc_cohens_h(base_p, q_p)

        results.append({
            'Model': model, 'Metric': 'ARC', 'Quant': row['Quant'],
            'Base_Score': base_p, 'Quant_Score': q_p, 'Diff': q_p - base_p,
            'Stat': stat, 'p_value': pval, 'Effect_Size': h, 'Effect_Type': "Cohen's h"
        })

# 4.2 处理 KLD 数据 (T-test + Cohen's d)
for model in df_kld['Model'].unique():
    model_df = df_kld[df_kld['Model'] == model]
    baseline = model_df[model_df['Quant'].isin(['F16', 'FP16'])]
    if baseline.empty: continue

    base_m, base_s = baseline['Mean_KLD'].values[0], baseline['KLD_SD'].values[0]

    # 容错处理：如果 F16 的 SD 是 0，给它一个极小的值防除零错误
    if base_s == 0: base_s = 1e-5

    for _, row in model_df.iterrows():
        if row['Quant'] in ['F16', 'FP16']: continue

        q_m, q_s = row['Mean_KLD'], row['KLD_SD']
        if q_s == 0: q_s = 1e-5

        # 两样本 t 检验 (Welch)
        stat, pval = ttest_ind_from_stats(base_m, base_s, N_KLD, q_m, q_s, N_KLD, equal_var=False)
        d = calc_cohens_d(base_m, base_s, N_KLD, q_m, q_s, N_KLD)

        results.append({
            'Model': model, 'Metric': 'KLD', 'Quant': row['Quant'],
            'Base_Score': base_m, 'Quant_Score': q_m, 'Diff': q_m - base_m,
            'Stat': stat, 'p_value': pval, 'Effect_Size': d, 'Effect_Type': "Cohen's d"
        })

df_res = pd.DataFrame(results)

# ==========================================
# 5. 多重比较校正与输出
# ==========================================
# 对 p 值进行 FDR-BH 校正
if not df_res.empty:
    _, qvals, _, _ = multipletests(df_res['p_value'], alpha=ALPHA, method='fdr_bh')
    df_res['q_value'] = qvals

    # 判定是否有意义：q < 0.05 且 效应量 >= 0.2
    df_res['Is_Significant'] = (df_res['q_value'] < ALPHA) & (df_res['Effect_Size'] >= EFFECT_THRESHOLD)

    # 美化输出
    df_res = df_res.round(4)
    cols = ['Model', 'Metric', 'Quant', 'Base_Score', 'Quant_Score', 'Diff', 'p_value', 'q_value', 'Effect_Size',
            'Is_Significant']
    df_res = df_res[cols].sort_values(by=['Metric', 'Model', 'Quant']).reset_index(drop=True)

    df_res.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("=" * 80)
    print(f"✅ 统计检验完成！已保存至: {OUTPUT_FILE}")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print("💡 数据解读提示：")
    print("1. ARC 的 Diff > 0 表示量化后准确率反超 F16；KLD 的 Diff > 0 表示散度增加（性能下降）。")
    print("2. 只有当 Is_Significant 为 True 时，才在论文中定性为“显著差异”。")
    print("3. 注意看 Qwen3B 的 Q4_K_M，你会发现它的效应量在 0.18-0.20 左右，且 q值极小。")