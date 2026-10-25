# C:\Users\cjy13\quant_eval\04_scripts\build_mixed_reasoning_set.py
import json, os, random
from datasets import load_dataset

# ================= 配置区 =================
SELF_MADE_PATH = r"C:\Users\cjy13\quant_eval\03_data\reasoning\reasoning120.jsonl"
# 注意：输出文件名改为了 400
OUTPUT_PATH = r"C:\Users\cjy13\quant_eval\03_data\reasoning\reasoning_mixed_400.jsonl"
CACHE_DIR = r"C:\Users\cjy13\quant_eval\cache\hf_datasets"
# ==========================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. 加载自建题库
with open(SELF_MADE_PATH, "r", encoding="utf-8") as f:
    self_made = [json.loads(line) for line in f]
print(f"✅ 已加载自建题库: {len(self_made)} 题")

new_questions = []

# 2. 加载并转换 GSM8K (测试集前 130 题)
# GSM8K 主要是 2-4 步的数学应用题，我们将其归入 3step 和 4step
gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split="test",
    cache_dir=CACHE_DIR
)
for i, item in enumerate(gsm8k.select(range(130))):
    final_ans = item["answer"].split("####")[-1].strip()
    # 前65题归为3步，后65题归为4步
    level = "3step" if i < 65 else "4step"
    new_questions.append({
        "id": f"gsm8k_{i+1:03d}",
        "level": level,
        "question": item["question"],
        "answer": final_ans,
        "source": "GSM8K"
    })
print(f"✅ 已转换 GSM8K: 130 题")

# 3. 加载并转换 BBH 的 multistep_arithmetic_two 子任务 (前 75 题)
# 多步算术，归入 4step 和 5step+
bbh_math = load_dataset(
    "lukaemon/bbh",
    "multistep_arithmetic_two",
    split="test",
    cache_dir=CACHE_DIR
)
for i, item in enumerate(bbh_math.select(range(75))):
    level = "4step" if i < 35 else "5step+"
    new_questions.append({
        "id": f"bbh_math_{i+1:03d}",
        "level": level,
        "question": item["input"],  # ✅ 修正：BBH 的字段是 input
        "answer": item["target"],
        "source": "BBH_multistep_arithmetic_two"
    })
print(f"✅ 已转换 BBH (multistep_arithmetic_two): 75 题")

# 4. 加载并转换 BBH 的 word_sorting 子任务 (前 75 题)
# 单词排序，归入 2step 和 3step
bbh_sort = load_dataset(
    "lukaemon/bbh",
    "word_sorting",
    split="test",
    cache_dir=CACHE_DIR
)
for i, item in enumerate(bbh_sort.select(range(75))):
    level = "2step" if i < 35 else "3step"
    new_questions.append({
        "id": f"bbh_sort_{i+1:03d}",
        "level": level,
        "question": item["input"],  # ✅ 修正：BBH 的字段是 input
        "answer": item["target"],
        "source": "BBH_word_sorting"
    })
print(f"✅ 已转换 BBH (word_sorting): 75 题")

# 5. 合并、打乱并保存
all_questions = self_made + new_questions
random.seed(42)
random.shuffle(all_questions)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for q in all_questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"\n🎉 合并完成！总计 {len(all_questions)} 题")
print(f"📁 已保存至: {OUTPUT_PATH}")