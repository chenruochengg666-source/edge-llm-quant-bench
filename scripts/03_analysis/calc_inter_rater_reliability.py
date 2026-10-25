# C:\Users\cjy13\quant_eval\04_scripts\calc_inter_rater_reliability.py
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
import krippendorff
import os

base = r"C:\Users\cjy13\quant_eval\06_results"
manual_dir = os.path.join(base, "manual_scoring")
out_csv = os.path.join(base, "table_inter_rater_reliability.csv")

rater1_path = os.path.join(manual_dir, "rater1_scored.csv")
rater2_path = os.path.join(manual_dir, "rater2_scored.csv")

# 1. 读取评分 (兼容 WPS/Excel 可能带来的 GBK 编码)
try:
    r1 = pd.read_csv(rater1_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    r1 = pd.read_csv(rater1_path, encoding='gbk')

try:
    r2 = pd.read_csv(rater2_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    r2 = pd.read_csv(rater2_path, encoding='gbk')

# 2. 合并评分表
m = pd.merge(r1[["Task_ID", "Manual_Score"]], r2[["Task_ID", "Manual_Score"]], on="Task_ID", suffixes=("_r1", "_r2"))
m = m.dropna(subset=["Manual_Score_r1", "Manual_Score_r2"])

# 3. 将 0, 0.5, 1.0 映射为 0, 1, 2 用于有序变量计算
def map_score(s):
    if pd.isna(s): return None
    if float(s) == 1.0: return 2
    if float(s) == 0.5: return 1
    return 0

y1 = m["Manual_Score_r1"].apply(map_score).values
y2 = m["Manual_Score_r2"].apply(map_score).values

# 4. 原始一致率 (Raw Agreement)
raw_agreement = np.mean(y1 == y2)

# 5. 加权 Cohen's Kappa
kappa = cohen_kappa_score(y1, y2, weights='quadratic')

# 6. Bootstrap 95% CI (重采样 1000 次)
kappas = []
for _ in range(1000):
    y1_res, y2_res = resample(y1, y2)
    # 如果重采样后方差为0（全一样），跳过防止报错
    if len(set(y1_res)) > 1 and len(set(y2_res)) > 1:
        kappas.append(cohen_kappa_score(y1_res, y2_res, weights='quadratic'))

ci_low = np.percentile(kappas, 2.5)
ci_high = np.percentile(kappas, 97.5)

# 7. Krippendorff's Alpha (ordinal 测量级别)
# 要求输入格式为 (评估者数量, 样本数量)
reliability_data = np.array([y1, y2])
try:
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='ordinal')
except Exception as e:
    alpha = np.nan
    print(f"Alpha 计算失败: {e}")

# 8. 保存结果
res = pd.DataFrame([{
    "Total_Samples": len(y1),
    "Raw_Agreement": round(raw_agreement, 4),
    "Weighted_Cohen_Kappa": round(kappa, 4),
    "Kappa_95_CI_Low": round(ci_low, 4),
    "Kappa_95_CI_High": round(ci_high, 4),
    "Krippendorff_Alpha": round(alpha, 4) if pd.notna(alpha) else "N/A"
}])

res.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 标注者间信度计算完成，已保存至: {out_csv}")
print(res.T.to_string(header=False))