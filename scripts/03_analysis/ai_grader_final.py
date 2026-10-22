# C:\Users\cjy13\quant_eval\04_scripts\ai_grader_final.py
import pandas as pd
import re, os

detail_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail.csv"
out_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_reasoning_detail_AI_final.csv"

df = pd.read_csv(detail_csv)


def ai_score(pred, gold):
    """AI 评分员 1 的完整评分逻辑"""
    p = str(pred).strip().upper().replace(" ", "")
    g = str(gold).strip().upper().replace(" ", "")

    # 1. 空/无效直接 0
    if p in ["", "?", "[ERROR]", "NAN", "无", "无答案"]:
        return 0.0

    # 2. 完全匹配
    if p == g:
        return 1.0

    # 3. 去掉标点后匹配
    p_clean = re.sub(r'[，。：；、,\.\:;]', '', p)
    g_clean = re.sub(r'[，。：；、,\.\:;]', '', g)
    if p_clean == g_clean:
        return 1.0

    # 4. Gold 是排序题（含 >），提取所有字母比较
    if ">" in g:
        g_tokens = re.findall(r'[A-Z]', g)
        p_tokens = re.findall(r'[A-Z]', p)
        # 如果预测中包含了完全一样的顺序（逗号分隔也算对）
        if g_tokens and p_tokens == g_tokens:
            return 0.5  # 顺序对，格式错
        # 如果预测中包含了 gold 的全部字母（顺序相同）
        if g_tokens and len(p_tokens) >= len(g_tokens) and p_tokens[:len(g_tokens)] == g_tokens:
            return 0.5

    # 5. Gold 是数字
    if g.isdigit():
        # 如果预测中含该数字，给 0.5
        if g in p:
            return 0.5
        return 0.0

    return 0.0


df["AI_Manual_Score"] = df.apply(lambda r: ai_score(r["Pred"], r["Gold"]), axis=1)

# 保存完整表
df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"✅ 完整 AI 评分表已保存至: {out_csv}")
print(f"总行数: {len(df)}")

# 汇总统计
summary = df.groupby(["Model", "Quant"]).agg(
    Total_AI_Score=("AI_Manual_Score", "sum"),
    Total_Questions=("AI_Manual_Score", "count"),
    AI_Accuracy=("AI_Manual_Score", "mean")
).reset_index()
summary["AI_Accuracy"] = summary["AI_Accuracy"].round(4)

summary_csv = r"C:\Users\cjy13\quant_eval\06_results\table5_AI_summary_final.csv"
summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f"✅ AI 评分汇总表已保存至: {summary_csv}")
print(summary.to_string())