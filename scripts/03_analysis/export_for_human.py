# C:\Users\cjy13\quant_eval\04_scripts\export_for_human.py
import json, csv, os

# 推理结果目录
reasoning_dir = r"C:\Users\cjy13\quant_eval\05_raw\reasoning"
# 输出目录
out_dir = r"C:\Users\cjy13\quant_eval\06_results"
os.makedirs(out_dir, exist_ok=True)

# 挑选几个有代表性的模型档位（覆盖不同量化程度）
targets = [
    "qwen1.5b-f16.gguf.jsonl",
    "qwen1.5b-Q4_K_M.gguf.jsonl",
    "qwen1.5b-Q2_K.gguf.jsonl",
    "qwen3b-Q4_K_M.gguf.jsonl"
]

for target in targets:
    src = os.path.join(reasoning_dir, target)
    if not os.path.exists(src):
        print(f"⚠️ 找不到 {src}，跳过")
        continue

    # 输出文件名：qwen1.5b-Q4_K_M_human.csv
    out_name = target.replace(".gguf.jsonl", "_human.csv")
    out_path = os.path.join(out_dir, out_name)

    rows = []
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "ID": item["id"],
                "Level": item["level"],
                "Gold": item["gold"],
                "Raw_Output": item.get("raw", ""),
                "Manual_Score": ""  # 留空，等同学填 0 / 0.5 / 1.0
            })

    # 写 CSV
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Level", "Gold", "Raw_Output", "Manual_Score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 已导出 {len(rows)} 条数据 -> {out_path}")

print("\n全部导出完成！请把 CSV 发给同学，按以下规则打分：")
print("  1.0 = 最终答案正确（忽略空格、大小写、标点）")
print("  0.5 = 排序正确但格式有误（如用逗号代替大于号）")
print("  0.0 = 错误、乱码、空白、复读")