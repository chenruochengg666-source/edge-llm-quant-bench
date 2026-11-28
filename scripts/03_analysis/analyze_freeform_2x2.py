# -*- coding: utf-8 -*-
"""
0.5B「答题格式 × 量化精度」2×2 对照分析
======================================
四个格：{FP16, Q4_K_M} × {约束作答（arc_unified，只输出一个字母）, 自由作答（arc_freeform，不加约束）}
两种作答格式的 max_tokens 均为 64，T=0，题库与标签修正相同。

自由作答的答案提取：
  1) 先用与主实验完全相同的 extract_letter()（单字母 / 开头字母 / 明确答案句式）
  2) 若失败，再匹配结论句式，如 “The correct answer is:\\n\\nC. ...”“正确答案是 B”
     —— 字母后必须紧跟标点、换行或句末，避免把冠词 a 或选项正文误判为字母
  3) 仍失败记为“未提取”（多为 64 token 内只逐项分析、未给出结论）
两档使用同一提取规则。

主分析集：四个格均成功提取答案的题目（避免“未提取”混淆格式效应）；
敏感性分析：全部 2376 题，未提取计为答错。
交互效应：逐题 d_i = (Q约束 − F约束) − (Q自由 − F自由)，配对 t 检验（大样本近似为 z）。

用法（仓库根目录下）：python scripts/03_analysis/analyze_freeform_2x2.py
输出：results/summary/freeform_2x2_summary.csv、freeform_2x2_items.csv
"""
import csv, json, math, os, re
from collections import Counter

# ---------- 路径配置（相对仓库根目录；模型与 llama.cpp 位置用环境变量指定） ----------
from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("EDGE_LLM_DATA_DIR", str(REPO / "data"))   # 题库：data/arc/*.jsonl（仓库不含，见 data/README.md）
RAW_DIR = str(REPO / "results" / "raw")              # 逐题原始输出
SUMMARY_DIR = str(REPO / "results" / "summary")      # 汇总表
LLAMA_BIN = os.environ.get("LLAMA_CPP_BIN", str(REPO.parent / "llama.cpp" / "build" / "bin" / "Release"))
MODEL_DIRS = [p for p in os.environ.get("EDGE_LLM_MODEL_DIR", str(REPO / "models")).split(os.pathsep) if p]
# 示例（PowerShell）：$env:LLAMA_CPP_BIN="D:\llama.cpp\build\bin\Release"; $env:EDGE_LLM_MODEL_DIR="D:\gguf"
# 注意：llama-bench 无法读取含中文字符的模型路径，模型目录请使用纯英文路径。
# -----------------------------------------------------------------------------------
DATA = os.path.join(DATA_DIR, "arc", "ARC-Easy-test-full.jsonl")
CON = os.path.join(RAW_DIR, "arc_unified")
FREE = os.path.join(RAW_DIR, "arc_freeform")
OUT = SUMMARY_DIR
MODELS = [("FP16", "qwen0.5b-f16"), ("Q4_K_M", "qwen0.5b-Q4_K_M")]


def extract_letter(ans):
    """与 run_arc_easy_unified.py / analyze_arc_unified.py 完全相同。"""
    if not ans:
        return "?"
    a = ans.strip().upper()
    if a in ("A", "B", "C", "D", "E"):
        return a
    m = re.match(r"^[\s\.\:\,\u3001\uff0c\uff1a\-\*\uff08\(]*([A-E])(?![A-Z])", a)
    if m:
        return m.group(1)
    for pat in [r"(?:ANSWER\s*(?:IS|:))\s*\(?([A-E])\)?", r"(?:答案\s*(?:是|为)?\s*[：:])\s*\(?([A-E])\)?",
                r"(?:答案\s*(?:是|为)\s*)\(?([A-E])\)?(?![\u4e00-\u9fff])", r"^\(([A-E])\)",
                r"^([A-E])[\.\:、，,]", r"(?:选项|OPTION|CHOICE)\s*([A-E])\s*(?:正确|IS|CORRECT)"]:
        m = re.search(pat, a)
        if m:
            return m.group(1)
    if len(a) <= 3:
        m = re.search(r"\b([A-E])\b", a)
        if m:
            return m.group(1)
    return "?"


END = r"(?=[\.\)\:：,，。\*]|\s*$|\s*\n|\s+(?:IS|WOULD|BECAUSE|SINCE))"
FREE_PATTERNS = [
    r"(?:CORRECT|BEST|RIGHT|MOST\s+LIKELY|MOST\s+ACCURATE|MOST\s+APPROPRIATE)[^\n]{0,80}?\b(?:IS|ARE|WOULD\s+BE)\s*[:：]?\s*\n*\s*\**\s*(?:OPTION\s*|CHOICE\s*)?\(?([A-D])\)?" + END,
    r"\bANSWER\s*(?:IS|:|：)\s*[:：]?\s*\n*\s*\**\s*(?:OPTION\s*|CHOICE\s*)?\(?([A-D])\)?" + END,
    r"\b(?:OPTION|CHOICE)\s*\(?([A-D])\)?\s*(?:IS|WOULD\s+BE)\s+(?:THE\s+)?(?:CORRECT|BEST|RIGHT|MOST)",
    r"(?:正确答案|答案|正确选项|最佳答案|正确的选项)\s*(?:是|为|应该是|应为)?\s*[：:]?\s*\**\s*(?:选项\s*)?\(?([A-D])(?![A-Z])",
    r"(?:选|选择)\s*\(?([A-D])\)?\s*(?:项)?\s*(?:正确|是正确的)",
]


def extract_free(raw):
    p = extract_letter(raw)
    if p != "?":
        return p
    u = (raw or "").upper()
    for pat in FREE_PATTERNS:
        m = re.search(pat, u)
        if m:
            return m.group(1)
    return "?"


def load_gold():
    g = {}
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line); n = len(q["choices"]["text"]); k = str(q["answerKey"]).strip().upper()
                g[q["id"]] = chr(64 + int(k)) if (k.isdigit() and 0 < int(k) <= n) else k
    return g


def load(path, free):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line); raw = str(it.get("raw", "") or "")
                if raw.strip() or it["id"] not in rows:
                    rows[it["id"]] = raw
    fn = extract_free if free else extract_letter
    return {i: fn(r) for i, r in rows.items()}


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(min(b, c) + 1)]
    mx = max(lp)
    return min(1.0, 2 * math.exp(mx) * sum(math.exp(x - mx) for x in lp))


def dist(preds):
    c = Counter(p for p in preds if p in "ABCD"); t = sum(c.values()) or 1
    return "/".join(f"{k}{c[k] / t * 100:.0f}" for k in "ABCD"), c.most_common(1)[0][1] / t if c else 0


def main():
    gold = load_gold()
    P = {}
    for lab, m in MODELS:
        P[(lab, "约束")] = load(os.path.join(CON, f"{m}.gguf.jsonl"), free=False)
        P[(lab, "自由")] = load(os.path.join(FREE, f"{m}.gguf.jsonl"), free=True)
    all_ids = [i for i in gold if all(i in P[k] for k in P)]
    sets = {"四格均可提取": [i for i in all_ids if all(P[k][i] != "?" for k in P)], "全部题目（未提取计错）": all_ids}
    summary = []
    for sname, ids in sets.items():
        N = len(ids)
        acc = {k: sum(P[k][i] == gold[i] for i in ids) / N for k in P}
        print(f"\n===== {sname}  N={N} =====")
        for fmt in ("约束", "自由"):
            f, q = acc[("FP16", fmt)], acc[("Q4_K_M", fmt)]
            b = sum(1 for i in ids if P[("FP16", fmt)][i] == gold[i] and P[("Q4_K_M", fmt)][i] != gold[i])
            c = sum(1 for i in ids if P[("FP16", fmt)][i] != gold[i] and P[("Q4_K_M", fmt)][i] == gold[i])
            dF, sF = dist([P[("FP16", fmt)][i] for i in ids]); dQ, sQ = dist([P[("Q4_K_M", fmt)][i] for i in ids])
            row = dict(analysis_set=sname, N=N, format=fmt, fp16_acc=round(f * 100, 2), q4_acc=round(q * 100, 2),
                       delta_pp=round((q - f) * 100, 2), h=round(cohen_h(f, q), 3), b_fp16_only=b, c_q4_only=c,
                       mcnemar_p=mcnemar_exact(b, c), fp16_dist=dF, q4_dist=dQ, q4_top_share=round(sQ * 100, 1))
            summary.append(row)
            print(f"  {fmt}作答：FP16 {f * 100:.2f}%  Q4_K_M {q * 100:.2f}%  Δ={(q - f) * 100:+.2f}pp  h={cohen_h(f, q):+.3f}  "
                  f"b/c={b}/{c}  McNemar p={mcnemar_exact(b, c):.3g}  选项分布 FP16 {dF} | Q4 {dQ}")
        d = [((P[("Q4_K_M", "约束")][i] == gold[i]) - (P[("FP16", "约束")][i] == gold[i]))
             - ((P[("Q4_K_M", "自由")][i] == gold[i]) - (P[("FP16", "自由")][i] == gold[i])) for i in ids]
        m = sum(d) / N; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (N - 1)); se = sd / math.sqrt(N); z = m / se
        p = math.erfc(abs(z) / math.sqrt(2))
        print(f"  交互效应（格式 × 量化）：{m * 100:+.2f}pp，95% CI [{(m - 1.96 * se) * 100:+.2f}, {(m + 1.96 * se) * 100:+.2f}]，z={z:.2f}，p={p:.3g}")
        summary.append(dict(analysis_set=sname, N=N, format="交互（约束Δ − 自由Δ）", delta_pp=round(m * 100, 2),
                            ci_low=round((m - 1.96 * se) * 100, 2), ci_high=round((m + 1.96 * se) * 100, 2), z=round(z, 2), p=p))
    # 未提取率
    print("\n未提取率：")
    for k in P:
        n_q = sum(1 for i in all_ids if P[k][i] == "?")
        print(f"  {k[0]:7s} {k[1]}作答：{n_q}/{len(all_ids)} = {n_q / len(all_ids) * 100:.1f}%")
        summary.append(dict(analysis_set="未提取率", N=len(all_ids), format=k[1], **{("fp16_acc" if k[0] == "FP16" else "q4_acc"): round(n_q / len(all_ids) * 100, 2)}))
    # 塌缩题的知识保留
    ids = sets["四格均可提取"]
    col = [i for i in ids if P[("Q4_K_M", "约束")][i] == "A" and gold[i] != "A"]
    kf = sum(P[("Q4_K_M", "自由")][i] == gold[i] for i in col) / len(col)
    print(f"\n约束作答下 Q4_K_M 误选 A 的 {len(col)} 题：同一模型自由作答答对 {kf * 100:.1f}%；FP16 自由作答答对 "
          f"{sum(P[('FP16', '自由')][i] == gold[i] for i in col) / len(col) * 100:.1f}%")
    try:
        os.makedirs(OUT, exist_ok=True)
        keys = sorted({k for r in summary for k in r}, key=lambda x: list(summary[0]).index(x) if x in summary[0] else 99)
        with open(os.path.join(OUT, "freeform_2x2_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
        with open(os.path.join(OUT, "freeform_2x2_items.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["id", "gold"] + [f"{a}_{b}" for a, b in P])
            for i in all_ids:
                w.writerow([i, gold[i]] + [P[k][i] for k in P])
        print(f"\n已写入 {OUT}\\freeform_2x2_summary.csv")
    except Exception as e:
        print(f"\n（未写入文件：{e}）")


if __name__ == "__main__":
    main()
