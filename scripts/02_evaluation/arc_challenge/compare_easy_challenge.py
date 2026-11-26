# -*- coding: utf-8 -*-
"""
对比 ARC-Easy 与 ARC-Challenge 上的准确率与区间归属。

用途：直接回应审稿意见第 3 条。输出一张对照表，用于论文 4.3.1 节。

用法：
    python compare_easy_challenge.py

依赖目录（与 run_arc_*.py 输出的 jsonl 对应）：
    05_raw/arc/             ARC-Easy 结果（已有）
    05_raw/arc_challenge/   ARC-Challenge 结果（本次新增）

输出：
    arc_easy_vs_challenge.csv
"""
import json, os, csv

EASY_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc"
CHAL_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc_challenge"
OUT_CSV  = r"C:\Users\cjy13\quant_eval\06_results\arc_easy_vs_challenge.csv"


def load_acc(directory):
    """读取目录下所有 jsonl，返回 {模型文件名: (正确数, 总题数, 准确率)}"""
    result = {}
    if not os.path.isdir(directory):
        print(f"⚠️ 目录不存在：{directory}")
        return result
    for fn in os.listdir(directory):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(directory, fn)
        correct = total = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not str(r.get("pred", "")).strip():
                    continue          # 跳过空预测（服务失败），与论文口径一致
                total += 1
                if r.get("ok"):
                    correct += 1
        if total:
            result[fn] = (correct, total, round(correct / total * 100, 2))
    return result


def main():
    easy = load_acc(EASY_DIR)
    chal = load_acc(CHAL_DIR)

    if not easy:
        print("❌ 未找到 ARC-Easy 结果，请先运行 run_arc_full.py")
        return
    if not chal:
        print("❌ 未找到 ARC-Challenge 结果，请先运行 run_arc_challenge.py")
        return

    rows = []
    common = sorted(set(easy) & set(chal))
    print(f"\n共 {len(common)} 个模型在两套题库上都有结果\n")
    print(f"{'模型':<48}{'Easy':>8}{'Challenge':>12}{'差值':>9}")
    print("-" * 78)
    for m in common:
        e_acc = easy[m][2]
        c_acc = chal[m][2]
        diff = round(c_acc - e_acc, 2)
        rows.append([m, e_acc, easy[m][1], c_acc, chal[m][1], diff])
        print(f"{m:<48}{e_acc:>8.2f}{c_acc:>12.2f}{diff:>+9.2f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型文件", "Easy准确率(%)", "Easy有效N",
                    "Challenge准确率(%)", "Challenge有效N", "差值(pp)"])
        w.writerows(rows)
    print(f"\n✅ 对照表已保存：{OUT_CSV}")


if __name__ == "__main__":
    main()
