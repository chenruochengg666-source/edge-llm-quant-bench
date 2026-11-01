# C:\Users\cjy13\quant_eval\04_scripts\debug_struct.py
import json, os

RAW_DIR = r"C:\Users\cjy13\quant_eval\05_raw\structured"
# 找出所有得 0% 的罪魁祸首
target_files = [
    "qwen1.5b-base.gguf.jsonl",
    "qwen3b-base.gguf.jsonl",
    "qwen2.5-7b-instruct-f16.gguf.jsonl",
    "llama3.1-8b-instruct-f16.gguf.jsonl",
    "phi3-base.gguf.jsonl"
]

for file in target_files:
    filepath = os.path.join(RAW_DIR, file)
    if not os.path.exists(filepath):
        continue

    print(f"\n==================================================")
    print(f"🔥 诊断文件: {file}")
    print(f"==================================================")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 我们只看前 2 题（1个提取题，1个表格题）
        for i in range(min(2, len(lines))):
            data = json.loads(lines[i])
            print(f"\n--- 题目ID: {data['id']} | 题型: {data['type']} ---")
            print(f"【模型原始输出】:\n{repr(data['raw_output'])}\n")