# C:\Users\cjy13\quant_eval\04_scripts\gen_reasoning.py
import json, random, os

random.seed(42)
out_dir = r"C:\Users\cjy13\quant_eval\03_data\reasoning"
os.makedirs(out_dir, exist_ok=True)

questions = []
qid = 0

# 生成传递关系题目（A > B > C > ...）
for level, count in [("2step", 25), ("3step", 25), ("4step", 25), ("5step+", 25)]:
    for i in range(count):
        qid += 1
        names = ["A", "B", "C", "D", "E", "F"]
        random.shuffle(names)
        n = int(level[0])  # 难度层对应的人数
        chain = names[:n]

        # 构造描述：chain[0] > chain[1] > ... > chain[n-1]
        desc = []
        for j in range(n - 1):
            desc.append(f"{chain[j]}大于{chain[j + 1]}")
        random.shuffle(desc)  # 打乱顺序增加难度
        question = f"{'，'.join(desc)}。请将这几个数从大到小排序，只输出排序结果。"
        answer = ">".join(chain)

        questions.append({
            "id": f"{level}_{i + 1:02d}",
            "level": level,
            "question": question,
            "answer": answer
        })

# 补充一些算术链和数列题，凑齐100题
for i in range(20):
    qid += 1
    a = random.randint(2, 10)
    d = random.randint(2, 5)
    seq = [a + d * j for j in range(5)]
    question = f"一列数的规律是：{', '.join(map(str, seq))}。请问下一个数是多少？只输出数字。"
    answer = str(seq[-1] + d)
    questions.append({
        "id": f"seq_{i + 1:02d}",
        "level": "4step" if i < 10 else "5step+",
        "question": question,
        "answer": answer
    })

# 打乱并写入文件
random.shuffle(questions)
out_path = os.path.join(out_dir, "reasoning100.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"生成完毕！共 {len(questions)} 题，已保存至 {out_path}")