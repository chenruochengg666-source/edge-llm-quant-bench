# C:\Users\cjy13\quant_eval\04_scripts\extract_reasoning.py
import os, json, re
import pandas as pd

raw_dir = r"C:\Users\cjy13\quant_eval\05_raw\reasoning"
out_dir = r"C:\Users\cjy13\quant_eval\06_results"
os.makedirs(out_dir, exist_ok=True)

records = []
summary_data = {}


def extract_pred_from_raw(raw_text):
    """从模型的完整输出中智能提取最终答案"""
    if not raw_text or pd.isna(raw_text):
        return ""

    raw_text = str(raw_text).strip()

    # 1. 优先寻找显式的“答案：”或“答案:”
    for line in raw_text.splitlines()[::-1]:
        if "答案：" in line or "答案:" in line:
            return line.split("：")[-1].split(":")[-1].strip()

    # 2. 兜底：如果没写“答案：”，去最后 50 个字符里找数字或字母
    last_50_chars = raw_text[-50:]

    # 找数字（兼容 GSM8K 的纯数字，或带负号小数）
    nums = re.findall(r'-?\d+\.?\d*', last_50_chars)
    if nums:
        return nums[-1]

    # 找字母（兼容 ARC 题或逻辑题 A>B>C）
    letters = re.findall(r'[A-E]', last_50_chars)
    if letters:
        return letters[-1]

    # 3. 实在找不到，返回最后半句话
    words = raw_text.split()
    return words[-1] if words else ""


def check_correct(pred, gold, source):
    """增强的语义级自动评分逻辑"""
    p = str(pred).strip()
    g = str(gold).strip()

    if not p:
        return 0.0

    # 统一转小写，去除多余空白
    p_clean = p.lower().replace(" ", "").replace("\n", "")
    g_clean = g.lower().replace(" ", "").replace("\n", "")

    # 1. 基础精确匹配
    if p_clean == g_clean:
        return 1.0

    # ================= 2. 针对 GSM8K / BBH 算术题 =================
    if "gsm8k" in source or "arithmetic" in source:
        # 去除千分位逗号 (如 10,000 -> 10000)
        p_num_str = p.replace(",", "")
        g_num_str = g.replace(",", "")

        # 提取最后一个数字
        p_nums = re.findall(r'-?\d+\.?\d*', p_num_str)
        g_nums = re.findall(r'-?\d+\.?\d*', g_num_str)

        if p_nums and g_nums:
            try:
                # 浮点数比较，容忍极小的精度误差
                if abs(float(p_nums[-1]) - float(g_nums[-1])) < 1e-5:
                    return 1.0
            except ValueError:
                pass

    # ================= 3. 针对 BBH 单词排序题 =================
    if "word_sorting" in source:
        # 核心逻辑：提取所有纯英文字母组成的单词，然后按顺序比对
        # 这样可以完美过滤掉模型输出的 "1.", ",", "\n" 等干扰项
        p_words = re.findall(r'[a-z]+', p_clean)
        g_words = re.findall(r'[a-z]+', g_clean)
        if p_words == g_words:
            return 1.0

    # ================= 4. 针对自建逻辑/比较题 (A > B > C) =================
    if ">" in g or "sorting" in source or "self_made" in source:
        # 只提取字母和大于号
        p_logic = re.sub(r'[^a-z>]', '', p_clean)
        g_logic = re.sub(r'[^a-z>]', '', g_clean)

        # 兼容模型用逗号代替大于号的情况（如 "A,B,C" 变成 "A>B>C"）
        if p_logic == g_logic:
            return 1.0

        # 如果模型输出了排序，但没用大于号，尝试按字母顺序直接比对
        p_letters = re.findall(r'[a-z]', p_logic)
        g_letters = re.findall(r'[a-z]', g_logic)
        if p_letters == g_letters and len(g_letters) > 0:
            # 注意：只有在原题包含字母 A, B, C 时才适用
            if all(c in 'abcdef' for c in g_letters):
                return 1.0

    return 0.0


print("开始解析推理结果...")
for filename in os.listdir(raw_dir):
    if not filename.endswith(".jsonl"):
        continue

    parts = filename.replace(".gguf.jsonl", "").split("-")
    model = parts[0]
    quant = "-".join(parts[1:])

    # ✅ 新增：将 base 统一映射为 F16，避免模型重复
    if quant == "base":
        quant = "F16"

    with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue

            source = item.get("source", "self_made")

            # ⭐ 核心修改：不直接读 pred，而是根据 raw 重新智能提取！
            raw_text = item.get("raw", "")
            pred = extract_pred_from_raw(raw_text)

            auto_score = check_correct(pred, item.get("gold", ""), source)

            records.append({
                "Task_ID": f"{model}_{quant}_{item['id']}",
                "Model": model,
                "Quant": quant,
                "Level": item.get("level", ""),
                "Source": source,
                "ID": item.get("id", ""),
                "Question": item.get("question", ""),
                "Gold": item.get("gold", ""),
                "Pred": pred,  # 这里存入智能提取后的答案
                "Auto_Score": auto_score,
                "Manual_Score": "",
                "Raw_Output": raw_text
            })

df_detail = pd.DataFrame(records)
detail_csv = os.path.join(out_dir, "table5_reasoning_detail.csv")
df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')
print(f"✅ 推理明细已保存至: {detail_csv}")

# 自动汇总逻辑保持不变
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