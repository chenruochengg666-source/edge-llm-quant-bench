# 多步推理 400 题逐题配对 McNemar 精确检验（第二版复现脚本）
# 输入：results/raw/reasoning/*.gguf.jsonl（论文表 S1 所用版本）
# 用法（仓库根目录下）：python scripts/03_analysis/mcnemar_reasoning400.py
# 评分：与 04_scripts/extract_reasoning.py 的 extract_pred_from_raw + check_correct 完全一致
# 输出：results/summary/mcnemar_reasoning400.csv
# 口径：both_valid（双方均有非空输出，主口径）/ all400（空输出计错）
# 检验：精确二项 McNemar（双侧）；BH-FDR 检验族 = 同一模型、同一口径下全部量化档 vs FP16
import glob, json, math, os, re
import pandas as pd

# ---------- 路径配置（相对仓库根目录） ----------
from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]
RAW_DIR = str(REPO / "results" / "raw")
SUMMARY_DIR = str(REPO / "results" / "summary")
REASONING_DIR = os.path.join(RAW_DIR, "reasoning")   # 论文表 S1 所用版本（经 3 次重试的新版日志）
# -----------------------------------------------


def extract_pred_from_raw(raw_text):
    if not raw_text:
        return ""
    for line in raw_text.splitlines()[::-1]:
        if "答案：" in line or "答案:" in line:
            return line.split("：")[-1].split(":")[-1].strip()
    last = raw_text[-50:]
    nums = re.findall(r'-?\d+\.?\d*', last)
    if nums:
        return nums[-1]
    letters = re.findall(r'[A-E]', last)
    if letters:
        return letters[-1]
    words = raw_text.split()
    return words[-1] if words else ""


def check_correct(pred, gold, source):
    p, g = str(pred).strip(), str(gold).strip()
    if not p:
        return 0.0
    pc = p.lower().replace(" ", "").replace("\n", "")
    gc = g.lower().replace(" ", "").replace("\n", "")
    if pc == gc:
        return 1.0
    if "gsm8k" in source or "arithmetic" in source:
        pn = re.findall(r'-?\d+\.?\d*', p.replace(",", ""))
        gn = re.findall(r'-?\d+\.?\d*', g.replace(",", ""))
        if pn and gn:
            try:
                if abs(float(pn[-1]) - float(gn[-1])) < 1e-5:
                    return 1.0
            except ValueError:
                pass
    if "word_sorting" in source and re.findall(r'[a-z]+', pc) == re.findall(r'[a-z]+', gc):
        return 1.0
    if ">" in g or "sorting" in source or "self_made" in source:
        pl, gl = re.sub(r'[^a-z>]', '', pc), re.sub(r'[^a-z>]', '', gc)
        if pl == gl:
            return 1.0
        pls, gls = re.findall(r'[a-z]', pl), re.findall(r'[a-z]', gl)
        if pls == gls and gls and all(ch in 'abcdef' for ch in gls):
            return 1.0
    return 0.0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def bh(ps):
    n = len(ps); order = sorted(range(n), key=lambda i: ps[i]); q = [0.0] * n; prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]; prev = min(prev, ps[i] * n / (rank + 1)); q[i] = prev
    return q


data = {}
for path in glob.glob(os.path.join(REASONING_DIR, "*.jsonl")):
    name = os.path.basename(path).replace(".gguf.jsonl", "")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            raw = it.get("raw", "") or ""
            rows[it["id"]] = (bool(raw.strip()),
                              check_correct(extract_pred_from_raw(raw), it.get("gold", ""), it.get("source", "self_made")))
    if rows:
        data[name] = rows

FAMILIES = {"qwen1.5b": ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"],
            "qwen3b": ["Q4_K_M", "Q3_K_M", "Q2_K"], "qwen0.5b": ["Q4_K_M", "Q2_K"],
            "phi3": ["Q4_K_M", "Q2_K"], "ll32": ["Q4_K_M", "Q2_K"]}
out = []
for model, quants in FAMILIES.items():
    base = data[f"{model}-base"]
    for scope in ("both_valid", "all400"):
        res = []
        for q in quants:
            Q = data[f"{model}-{q}"]
            ids = [i for i in base if i in Q and (scope == "all400" or (base[i][0] and Q[i][0]))]
            b = sum(1 for i in ids if base[i][1] == 1 and Q[i][1] == 0)
            c = sum(1 for i in ids if base[i][1] == 0 and Q[i][1] == 1)
            pb = sum(base[i][1] for i in ids) / len(ids); pq = sum(Q[i][1] for i in ids) / len(ids)
            res.append([model, q, scope, len(ids), 100 * pb, 100 * pq, 100 * (pq - pb), cohen_h(pb, pq), b, c, mcnemar_exact(b, c)])
        for r, qv in zip(res, bh([r[-1] for r in res])):
            out.append(r + [qv])

df = pd.DataFrame(out, columns=["model", "quant", "scope", "N", "acc_fp16", "acc_quant", "delta_pp", "h",
                                "fp16_only_correct", "quant_only_correct", "p_exact", "q_BH"])
df.round(4).to_csv(os.path.join(SUMMARY_DIR, "mcnemar_reasoning400.csv"), index=False, encoding="utf-8-sig")
print(df.round(4).to_string())
