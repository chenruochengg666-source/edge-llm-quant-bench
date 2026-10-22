# C:\Users\cjy13\quant_eval\04_scripts\extract_arc.py
import os, json, math
import pandas as pd
from scipy import stats

arc_dir = r"C:\Users\cjy13\quant_eval\05_raw\arc"
out_csv = r"C:\Users\cjy13\quant_eval\06_results\table4_downstream_arc.csv"

data = []
for filename in os.listdir(arc_dir):
    if not filename.endswith(".jsonl"):
        continue

    # 解析文件名，例如 qwen1.5b-Q4_K_M.gguf.jsonl
    parts = filename.replace(".gguf.jsonl", "").split("-")
    model = parts[0]
    quant = "-".join(parts[1:])

    correct = 0
    total = 0
    with open(os.path.join(arc_dir, filename), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item.get("ok", False):
                correct += 1

    if total > 0:
        acc = correct / total
        # 计算 95% 置信区间 (正态近似)
        se = math.sqrt(acc * (1 - acc) / total)
        ci_low = max(0, acc - 1.96 * se)
        ci_high = min(1, acc + 1.96 * se)

        data.append({
            "Model": model,
            "Quant": quant,
            "Correct": correct,
            "Total": total,
            "Accuracy": round(acc, 4),
            "95%_CI_Low": round(ci_low, 4),
            "95%_CI_High": round(ci_high, 4)
        })

df = pd.DataFrame(data)
# 排序，让表更美观
df = df.sort_values(by=["Model", "Quant"]).reset_index(drop=True)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"✅ ARC 数据提取完毕，共 {len(df)} 条记录，已保存至: {out_csv}")
print(df.to_string())