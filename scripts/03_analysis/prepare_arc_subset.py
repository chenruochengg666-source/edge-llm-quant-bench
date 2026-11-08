# prepare_arc_subset.py (v2)
# 固定 seed(0) 抽取 500 题；若已存在则校验题数与 id 集合，不覆盖
import json, os, random

DATA_DIR = r"C:\Users\cjy13\quant_eval\03_data\arc"
IN_FILE = os.path.join(DATA_DIR, "ARC-Easy-test-full.jsonl")
OUT_FILE = os.path.join(DATA_DIR, "ARC-Easy-subset-500.jsonl")
N_SAMPLE = 500

def load(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def main():
    if not os.path.exists(IN_FILE):
        print(f"[ERR] 找不到原始题库: {IN_FILE}")
        return
    data = load(IN_FILE)
    print(f"原始题库共 {len(data)} 题")
    if len(data) < N_SAMPLE:
        print(f"[ERR] 题库不足 {N_SAMPLE} 题，无法抽样")
        return

    random.seed(0)
    subset = sorted(random.sample(data, N_SAMPLE), key=lambda x: x["id"])

    # 已存在则校验，避免无意覆盖
    if os.path.exists(OUT_FILE):
        old = load(OUT_FILE)
        same = (len(old) == len(subset) and [x["id"] for x in old] == [x["id"] for x in subset])
        print(f"[INFO] 子集已存在，题数 {len(old)} | 与本次抽样一致: {same}")
        if same:
            print("[SKIP] 内容一致，无需重写")
            return
        print("[WARN] 内容不一致，将覆盖（旧文件另存 .bak）")
        os.replace(OUT_FILE, OUT_FILE + ".bak")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] 已保存 {len(subset)} 题 -> {OUT_FILE}")

if __name__ == "__main__":
    main()
