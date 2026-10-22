# C:\Users\cjy13\quant_eval\04_scripts\convert_arc_to_csv.py
import os
import json
import pandas as pd

# 配置你的输入目录（哪里存了 jsonl）和输出目录（想存在哪）
INPUT_DIRS = [
    r"C:\Users\cjy13\quant_eval\05_raw\arc_7b_subset",
    r"C:\Users\cjy13\quant_eval\05_raw\arc_llama8b_subset"
]
OUTPUT_DIR = r"C:\Users\cjy13\quant_eval\06_results\arc_csv_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_jsonl_to_csv():
    for in_dir in INPUT_DIRS:
        if not os.path.exists(in_dir):
            continue

        for filename in os.listdir(in_dir):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(in_dir, filename)

            # 从文件名中提取模型名字（去掉 .jsonl 后缀）
            model_name = filename.replace(".jsonl", "")

            data_rows = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # 提取我们关心的字段
                    data_rows.append({
                        "Question_ID": item.get("id", ""),
                        "Prediction": item.get("pred", ""),
                        "Gold_Answer": item.get("gold", ""),
                        "Is_Correct": item.get("ok", False),
                        "Raw_Output": item.get("raw", "").replace("\n", " ")  # 去掉换行，防止CSV串行
                    })

            if data_rows:
                df = pd.DataFrame(data_rows)
                # 计算一下总分，方便核对
                correct_count = df["Is_Correct"].sum()
                total_count = len(df)
                acc = correct_count / total_count if total_count > 0 else 0

                out_csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_arc_results.csv")
                df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

                print(f"✅ 已生成: {out_csv_path}")
                print(f"   -> 模型: {model_name}")
                print(f"   -> 题目数: {total_count} | 正确数: {correct_count} | 准确率: {acc:.4f}\n")


if __name__ == "__main__":
    convert_jsonl_to_csv()
    print("🎉 全部转换完成！去 06_results\\arc_csv_results 查看吧。")