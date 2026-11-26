# -*- coding: utf-8 -*-
"""
下载 ARC-Challenge test 划分，导出为 JSONL。

与 ARC-Easy 的区别：ARC-Challenge 过滤掉了检索与词共现可解的题目，
保留的是真正需要推理的题目，因此更适合检验"量化对推理能力的损害"。

用法：
    pip install datasets
    python prepare_arc_challenge.py
"""
from datasets import load_dataset
import json, os

# 输出目录：与 ARC-Easy 并列，不覆盖原数据
OUT_DIR = os.environ.get("ARC_OUT_DIR", r"C:\Users\cjy13\quant_eval\03_data\arc")
os.makedirs(OUT_DIR, exist_ok=True)

print("正在从 Hugging Face 加载 ARC-Challenge test split ...")
try:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
except Exception as e:
    print(f"加载失败：{e}")
    print("若无法访问 HuggingFace，可改用镜像：")
    print("  set HF_ENDPOINT=https://hf-mirror.com   （Windows）")
    print("  export HF_ENDPOINT=https://hf-mirror.com （Linux）")
    raise

print(f"加载完成，共 {len(ds)} 题（ARC-Easy 为 2376 题，Challenge 为 1172 题）")

out_path = os.path.join(OUT_DIR, "ARC-Challenge-test-full.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in ds:
        record = {
            "id": item["id"],
            "question": item["question"],
            "choices": {
                "text": item["choices"]["text"],
                "label": item["choices"]["label"],
            },
            "answerKey": item["answerKey"],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ ARC-Challenge 全量已保存至: {out_path}")
print("下一步：修改 run_arc_challenge.py 的 DATA_FILE 指向此文件，然后运行。")
