# -*- coding: utf-8 -*-
"""
ARC 统一协议结果的重算与区间判定（ARC-Easy / ARC-Challenge）
============================================================
用与 run_arc_easy_unified.py 相同的 extract_letter() 和数字标签修正，重新提取全部答案，输出：
  1) 各档准确率、失败率、未提取数、选项分布与单一选项最大占比（表 2 / 适用性自检 C2：仅检查 FP16 基准）
  2) 两比例 z 检验 + 模型内 BH-FDR + Cohen's h（表 5）
  3) McNemar 精确配对检验、翻转率、CA、Cohen's Kappa（表 9 / 结构偏移）
  4) 与保真度轨交叉后的区间判定（表 6 / 表 3）

数据来源：
  easy      量化档只读 results/raw/arc_unified（旧协议数据不混用）；
            FP16 优先读 arc_unified，否则读 results/raw/arc 原始 FP16（原 FP16 已带约束），统一重新提取。
  challenge 优先读 results/raw/arc_challenge_unified；没有则读 results/raw/arc_challenge
            （原 Challenge 脚本已对所有档位统一协议，只需修正数字标签）。

用法：
  python scripts/03_analysis/analyze_arc_unified.py                    # ARC-Easy
  python scripts/03_analysis/analyze_arc_unified.py --task challenge   # ARC-Challenge
  python scripts/03_analysis/analyze_arc_unified.py --quant-dir ..\\results\\raw\\arc   # 对照：旧协议量化档 + 新提取规则
结果写入 results/summary/arc_<task>_judgement.csv 与 arc_<task>_items.csv
"""
import argparse, csv, json, math, os, re
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
OUT_DIR = SUMMARY_DIR
TASK_CFG = {
    "easy": dict(data=os.path.join(DATA_DIR, "arc", "ARC-Easy-test-full.jsonl"),
                 new=os.path.join(RAW_DIR, "arc_unified"), old=os.path.join(RAW_DIR, "arc"),
                 fp16_old="-f16.gguf.jsonl", allow_old_quant=False,
                 families={"qwen1.5b": ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"],
                           "qwen3b": ["Q4_K_M", "Q3_K_M", "Q2_K"], "qwen0.5b": ["Q4_K_M", "Q2_K"],
                           "phi3": ["Q4_K_M", "Q2_K"], "ll32": ["Q4_K_M", "Q2_K"]}),
    "challenge": dict(data=os.path.join(DATA_DIR, "arc", "ARC-Challenge-test-full.jsonl"),
                      new=os.path.join(RAW_DIR, "arc_challenge_unified"),
                      old=os.path.join(RAW_DIR, "arc_challenge"),
                      fp16_old="-base.gguf.jsonl", allow_old_quant=True,
                      families={"qwen1.5b": ["Q4_K_M", "Q3_K_M", "Q2_K"], "qwen0.5b": ["Q4_K_M", "Q2_K"],
                                "phi3": ["Q4_K_M", "Q2_K"], "ll32": ["Q4_K_M", "Q2_K"]}),
    # 自由作答对照（不加约束、max_tokens=64），与 easy 组成“答题格式 × 量化精度”2×2
    "freeform": dict(data=os.path.join(DATA_DIR, "arc", "ARC-Easy-test-full.jsonl"),
                     new=os.path.join(RAW_DIR, "arc_freeform"), old=os.path.join(RAW_DIR, "arc_freeform"),
                     fp16_old="-f16.gguf.jsonl", allow_old_quant=False,
                     families={"qwen0.5b": ["Q4_K_M"]}),
}

# 保真度轨（results/raw/kld_logs 的 Mean PPL(Q)/Mean PPL(base) 与标准误，ctx=512, chunks=20）
BASE_PPL = {"qwen1.5b": (4.868249, 0.154599), "qwen3b": (3.286915, 0.100812),
            "qwen0.5b": (7.287735, 0.259267), "phi3": (3.766443, 0.112309), "ll32": (8.402524, 0.307761)}
QUANT_PPL = {  # (PPL, SE, KLD)
    "ll32-Q2_K": (23.574310, 1.019014, 1.089459), "ll32-Q4_K_M": (8.733169, 0.320986, 0.050625),
    "phi3-Q2_K": (7.060545, 0.254518, 0.720813), "phi3-Q4_K_M": (4.097967, 0.127224, 0.096109),
    "qwen0.5b-Q2_K": (8.739620, 0.323378, 0.196909), "qwen0.5b-Q4_K_M": (7.513174, 0.269844, 0.030620),
    "qwen1.5b-Q2_K": (14.366354, 0.582035, 1.117978), "qwen1.5b-Q3_K_M": (6.111597, 0.206693, 0.233885),
    "qwen1.5b-Q4_K_M": (5.289946, 0.173976, 0.078397), "qwen1.5b-Q5_K_M": (5.002034, 0.160294, 0.030711),
    "qwen1.5b-Q6_K": (4.964587, 0.158984, 0.016774), "qwen1.5b-Q8_0": (4.881814, 0.155127, 0.002315),
    "qwen3b-Q2_K": (21046.570339, 983.675745, 9.059998), "qwen3b-Q3_K_M": (7.386920, 0.276488, 1.040567),
    "qwen3b-Q4_K_M": (4.061950, 0.133726, 0.302117),
}
ALPHA, TAU_H, DELTA_PP, TAU_OPT = 0.05, 0.2, 5.0, 0.50


def extract_letter(ans):
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


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            n = len(q["choices"]["text"]); g = str(q["answerKey"]).strip().upper()
            if g.isdigit() and 0 < int(g) <= n:              # 数字标签 1~4 → A~D
                g = chr(64 + int(g))
            gold[q["id"]] = g
    return gold


def load(path, gold):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                raw = str(it.get("raw", "") or "")
                if raw.strip() or it["id"] not in rows:      # 续跑时有输出的记录覆盖空记录
                    rows[it["id"]] = raw
    out = {}
    for i, raw in rows.items():
        pred = extract_letter(raw)
        out[i] = {"valid": bool(raw.strip()), "pred": pred, "ok": int(pred == gold.get(i))}
    return out


def zprop(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(min(b, c) + 1)]
    mx = max(lp)
    return min(1.0, 2 * math.exp(mx) * sum(math.exp(x - mx) for x in lp))


def bh(ps):
    n = len(ps); order = sorted(range(n), key=lambda i: ps[i]); q = [0.0] * n; prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]; prev = min(prev, ps[i] * n / (rank + 1)); q[i] = prev
    return q


def summarize(d):
    v = [x for x in d.values() if x["valid"]]
    n = len(v); acc = sum(x["ok"] for x in v) / n if n else 0
    c = Counter(x["pred"] for x in v if x["pred"] in "ABCD")
    tot = sum(c.values()) or 1
    top, share = (c.most_common(1)[0][0], c.most_common(1)[0][1] / tot) if c else ("-", 0)
    return {"n_valid": n, "n_fail": len(d) - n, "n_qmark": sum(1 for x in v if x["pred"] == "?"),
            "acc": acc, "top_opt": top, "top_share": share,
            "dist": "/".join(f"{k}{c[k] / tot * 100:.0f}" for k in "ABCD")}


def pick(cfg, fname, allow_old):
    p = os.path.join(cfg["new"], fname)
    if os.path.exists(p):
        return p, "unified"
    if allow_old:
        p = os.path.join(cfg["old"], fname)
        if os.path.exists(p):
            return p, "original"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["easy", "challenge", "freeform"], default="easy")
    ap.add_argument("--quant-dir", default=None, help="指定量化档目录（对照分析用）")
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]
    gold = load_gold(cfg["data"])
    rows, items, missing = [], [], []
    for fam, quants in cfg["families"].items():
        f16p, f16_src = pick(cfg, f"{fam}-f16.gguf.jsonl", False)
        if f16p is None:
            f16p = os.path.join(cfg["old"], fam + cfg["fp16_old"]); f16_src = "original"
        if not os.path.exists(f16p):
            missing.append(f"{fam}-FP16"); continue
        B = load(f16p, gold); sb = summarize(B)
        fam_rows = []
        for q in quants:
            if args.quant_dir:
                qp = os.path.join(args.quant_dir, f"{fam}-{q}.gguf.jsonl"); q_src = "quant-dir"
                qp = qp if os.path.exists(qp) else None
            else:
                qp, q_src = pick(cfg, f"{fam}-{q}.gguf.jsonl", cfg["allow_old_quant"])
            if qp is None:
                missing.append(f"{fam}-{q}"); continue
            Q = load(qp, gold); sq = summarize(Q)
            z, p = zprop(sb["acc"], sb["n_valid"], sq["acc"], sq["n_valid"])
            h = cohen_h(sb["acc"], sq["acc"])
            ids = [i for i in B if i in Q and B[i]["valid"] and Q[i]["valid"]]
            b = sum(1 for i in ids if B[i]["ok"] and not Q[i]["ok"]); c = sum(1 for i in ids if not B[i]["ok"] and Q[i]["ok"])
            n11 = sum(1 for i in ids if B[i]["ok"] and Q[i]["ok"]); N = len(ids)
            po = (N - b - c) / N if N else 0
            pb = (n11 + b) / N if N else 0; pq = (n11 + c) / N if N else 0
            pe = pb * pq + (1 - pb) * (1 - pq); kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
            ppl, se, kld = QUANT_PPL[f"{fam}-{q}"]; bp, bse = BASE_PPL[fam]
            fam_rows.append(dict(task=args.task, model=fam, quant=q, fp16_src=f16_src, quant_src=q_src,
                                 fp16_acc=sb["acc"] * 100, fp16_n=sb["n_valid"], fp16_fail=sb["n_fail"], fp16_qmark=sb["n_qmark"],
                                 fp16_top=f'{sb["top_opt"]}{sb["top_share"] * 100:.0f}%', fp16_dist=sb["dist"],
                                 quant_acc=sq["acc"] * 100, quant_n=sq["n_valid"], quant_fail=sq["n_fail"], quant_qmark=sq["n_qmark"],
                                 quant_top=f'{sq["top_opt"]}{sq["top_share"] * 100:.0f}%', quant_dist=sq["dist"],
                                 delta_pp=(sq["acc"] - sb["acc"]) * 100, z=z, p=p, h=h,
                                 pair_N=N, b_fp16_only=b, c_quant_only=c, mcnemar_p=mcnemar_exact(b, c),
                                 flip_rate=(b + c) / N * 100 if N else 0, CA=n11 / N * 100 if N else 0, kappa=kappa,
                                 dPPL=ppl - bp, two_sigma=2 * math.sqrt(bse ** 2 + se ** 2), KLD=kld,
                                 _top_share=sb["top_share"], _q_top_share=sq["top_share"], _fp16_acc=sb["acc"], _fp16_n=sb["n_valid"]))
            for i in ids:
                items.append([args.task, fam, q, i, B[i]["pred"], Q[i]["pred"], gold.get(i), B[i]["ok"], Q[i]["ok"]])
        for r, qv in zip(fam_rows, bh([r["p"] for r in fam_rows])):
            r["q_BH"] = qv
            alarm = abs(r["dPPL"]) >= r["two_sigma"]
            D = (1 if r["delta_pp"] > 0 else -1) if (qv < ALPHA and abs(r["h"]) >= TAU_H) else 0
            zc = (r["_fp16_acc"] - 0.25) / math.sqrt(0.25 * 0.75 / r["_fp16_n"])
            if zc < 1.645:
                zone = "不适用（判别力不足）"
            elif r["_top_share"] > TAU_OPT:          # C2 只检查 FP16 基准：基准本身作答不稳定时判据失效
                zone = "不适用（作答行为不稳定）"
            elif alarm:
                zone = {-1: "同步退化区", 0: "保真先行区", 1: "能力反升区（反向报警）"}[D]
            elif D == -1 and abs(r["delta_pp"]) > DELTA_PP:
                zone = "沉默区"
            elif D == 1:
                zone = "能力反升区（无报警反升）"
            else:
                zone = "统计等价区"
            tags = ["结构偏移"] if (zone == "统计等价区" and r["mcnemar_p"] < ALPHA) else []
            if not zone.startswith("不适用") and r["_q_top_share"] > TAU_OPT:   # 量化导致的单一选项塌缩：属于退化结果，作标注
                tags.append(f"选项塌缩（{r['quant_top']}）")
            if r["quant_qmark"] > 0.05 * r["quant_n"]:
                tags.append(f"格式失控（未提取 {r['quant_qmark']} 题）")
            r.update(alarm="报警" if alarm else "未报警", zone=zone, tags="；".join(tags))
            rows.append(r)

    cols = ["model", "quant", "fp16_acc", "quant_acc", "delta_pp", "h", "q_BH", "b_fp16_only", "c_quant_only",
            "mcnemar_p", "flip_rate", "fp16_top", "quant_top", "quant_qmark", "dPPL", "two_sigma", "alarm", "zone", "tags", "quant_src"]
    print(f"\n题库：{args.task}")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print("\nFP16 摘要：")
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            seen.add(r["model"])
            print(f"  {r['model']}: acc={r['fp16_acc']:.2f}% n={r['fp16_n']} 失败={r['fp16_fail']} 未提取={r['fp16_qmark']} "
                  f"选项分布={r['fp16_dist']} 来源={r['fp16_src']}")
    if missing:
        print("\n尚未跑完 / 缺失：", "、".join(missing))
    if not rows:
        return
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = "_quantdir" if args.quant_dir else ""
        with open(os.path.join(OUT_DIR, f"arc_{args.task}_judgement{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            keys = [k for k in rows[0] if not k.startswith("_")]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        with open(os.path.join(OUT_DIR, f"arc_{args.task}_items{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["task", "model", "quant", "id", "pred_fp16", "pred_quant", "gold", "ok_fp16", "ok_quant"]); w.writerows(items)
        print(f"\n已写入 {OUT_DIR}\\arc_{args.task}_judgement{tag}.csv")
    except Exception as e:
        print(f"\n（未写入文件：{e}）")


if __name__ == "__main__":
    main()
