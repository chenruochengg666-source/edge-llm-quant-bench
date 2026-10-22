# C:\Users\cjy13\quant_eval\04_scripts\extract_reasoning.py
import os, json, math
import pandas as pd

raw_dir = r"C:\Users\cjy13\quant_eval\05_raw\reasoning"
out_dir = r"C:\Users\cjy13\quant_eval\06_results"
os.makedirs(out_dir, exist_ok=True)

records = []
summary_data = {}

# 遍历所有推理结果 JSONL 文件
for filename in os.listdir(raw_dir):
    if not filename.endswith(".jsonl"):
        continue

    # 解析文件名，例如 qwen1.5b-Q4_K_M.gguf.jsonl
    parts = filename.replace(".gguf.jsonl", "").split("-")
    model = parts[0]
    quant = "-".join(parts[1:])

    with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # 严格自动评分规则：pred 和 gold 完全一致（忽略空格和大小写）
            pred_clean = str(item.get("pred", "")).strip().upper().replace(" ", "")
            gold_clean = str(item.get("gold", "")).strip().upper().replace(" ", "")

            auto_score = 1.0 if pred_clean == gold_clean else 0.0

            # 如果预测是空或者问号，标记为“空/无效输出”
            if pred_clean in ["", "?", "[ERROR]"]:
                auto_score = 0.0
                status = "空/无效输出"
            elif auto_score == 1.0:
                status = "完全正确"
            else:
                status = "错误"

            records.append({
                "Model": model,
                "Quant": quant,
                "Level": item.get("level", ""),
                "ID": item.get("id", ""),
                "Gold": gold_clean,
                "Pred": pred_clean,
                "Auto_Score": auto_score,  # 严格得分（1.0 或 0.0）
                "Manual_Score": "",  # 留空，供你双人评分填写（可填 0.5）
                "Status": status,
                "Raw_Output": item.get("raw", "")  # 原始输出，用于核对过程
            })

# 导出完整明细表 (用于人工复核和计算 Kappa)
df_detail = pd.DataFrame(records)
detail_csv = os.path.join(out_dir, "table5_reasoning_detail.csv")
df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')
print(f"✅ 推理明细已保存至: {detail_csv}")

# 自动汇总各个模型在四个难度层的得分（基于严格自动评分，满分按题目数计）
summary_list = []
for (model, quant), group in df_detail.groupby(["Model", "Quant"]):
    level_scores = {"2step": 0, "3step": 0, "4step": 0, "5step+": 0}
    level_totals = {"2step": 0, "3step": 0, "4step": 0, "5step+": 0}

    for _, row in group.iterrows():
        lvl = row["Level"]
        if lvl in level_scores:
            level_totals[lvl] += 1
            if row["Auto_Score"] == 1.0:
                level_scores[lvl] += 1

    total_correct = sum(level_scores.values())
    total_questions = sum(level_totals.values())

    summary_list.append({
        "Model": model,
        "Quant": quant,
        "2step_Score": level_scores["2step"],
        "2step_Total": level_totals["2step"],
        "3step_Score": level_scores["3step"],
        "3step_Total": level_totals["3step"],
        "4step_Score": level_scores["4step"],
        "4step_Total": level_totals["4step"],
        "5step+_Score": level_scores["5step+"],
        "5step+_Total": level_totals["5step+"],
        "Total_Correct": total_correct,
        "Total_Questions": total_questions,
        "Strict_Accuracy": round(total_correct / total_questions, 4) if total_questions > 0 else 0
    })

df_summary = pd.DataFrame(summary_list)
summary_csv = os.path.join(out_dir, "table5_reasoning_summary.csv")
df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')

print(f"✅ 推理汇总表已保存至: {summary_csv}")
print("\n=== 推理任务汇总预览 (严格准确率) ===")
print(df_summary.sort_values(by=["Model", "Quant"]).to_string(index=False))