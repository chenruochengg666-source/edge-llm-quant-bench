# -*- coding: utf-8 -*-
"""
plot_reasoning_v2.py —— 多步推理相关图（图8/图9/图10/图13）重绘脚本

对应正文：表4（多步推理得分矩阵）、表8（实例级一致性 CA）
输出：fig08.png / fig09.png / fig10.png / fig13.png

用法：
  python plot_reasoning_v2.py --raw ../../05_raw/reasoning \
      --questions ../../03_data/reasoning/reasoning_mixed_400.jsonl \
      --ca-table ../../results/summary/table8_ca_kappa_full.csv \
      --fig-dir ../../figures

说明：
  - 评分口径与 extract_reasoning.py 一致（语义级自动判定）；
  - 图8/图9 为按推理步数分层的得分矩阵热力图，色标自适应；
  - 图10 为各模型档位总准确率柱状图；
  - 图13 为 FP16 与量化模型的实例级一致率（CA）热力图。
"""
import argparse, json, os, re, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAME = {"qwen1.5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "qwen0.5b": "Qwen2.5-0.5B",
        "ll32": "Llama-1B", "phi3": "Phi-3", "qwen2.5-7b-instruct": "Qwen2.5-7B",
        "llama3.1-8b-instruct": "Llama-8B"}
LEVELS = ["2step", "3step", "4step", "5step+"]
LVL = {"2step": "2步", "3step": "3步", "4step": "4步", "5step+": "5步+"}
COL = {"FP16": "#1F4E79", "Q8_0": "#2E75B6", "Q6_K": "#5B9BD5", "Q5_K_M": "#9DC3E6",
       "Q4_K_M": "#70AD47", "Q3_K_M": "#ED7D31", "Q2_K": "#FF0000"}


def norm(s):
    s = str(s or "").strip().lower().replace(" ", "")
    for a, b in (("，", ","), ("：", ":"), ("（", "("), ("）", ")")):
        s = s.replace(a, b)
    return s


def extract_pred(raw):
    if not raw:
        return ""
    for line in str(raw).splitlines()[::-1]:
        if "答案：" in line or "答案:" in line:
            return line.split("：")[-1].split(":")[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", str(raw)[-50:])
    if nums:
        return nums[-1]
    letters = re.findall(r"[A-E]", str(raw)[-50:])
    if letters:
        return letters[-1]
    w = str(raw).split()
    return w[-1] if w else ""


def check_correct(pred, gold, source):
    p, g = norm(pred), norm(gold)
    if not p:
        return 0.0
    if p == g:
        return 1.0
    if "gsm8k" in source or "arithmetic" in source:
        pn = re.findall(r"-?\d+\.?\d*", p.replace(",", ""))
        gn = re.findall(r"-?\d+\.?\d*", g.replace(",", ""))
        if pn and gn:
            try:
                if abs(float(pn[-1]) - float(gn[-1])) < 1e-5:
                    return 1.0
            except ValueError:
                pass
    if "word_sorting" in source:
        if re.findall(r"[a-z]+", p) == re.findall(r"[a-z]+", g):
            return 1.0
    if ">" in g or "sorting" in source or "self_made" in source:
        pl = re.sub(r"[^a-z>]", "", p)
        gl = re.sub(r"[^a-z>]", "", g)
        if pl == gl:
            return 1.0
        if re.findall(r"[a-z]", pl) == re.findall(r"[a-z]", gl) and gl:
            if all(c in "abcdef" for c in re.findall(r"[a-z]", gl)):
                return 1.0
    return 0.0


def build_stats(raw_dir, questions_path):
    q = {}
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                j = json.loads(line)
                q[j["id"]] = j
    rows = []
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".jsonl"):
            continue
        base = fn.replace(".gguf.jsonl", "")
        mk = next((k for k in NAME if base.startswith(k)), None)
        if not mk:
            continue
        quant = base[len(mk) + 1:] or "base"
        quant = "FP16" if quant.lower() in ("base", "f16") else quant.upper()
        acc = collections.defaultdict(lambda: [0, 0])
        with open(os.path.join(raw_dir, fn), encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qq = q.get(r.get("id"))
                raw = r.get("raw", "")
                if not qq or not str(raw).strip():
                    continue
                L = qq.get("level", "?")
                acc[L][0] += check_correct(extract_pred(raw), qq.get("answer") or qq.get("gold"), qq.get("source", "self_made"))
                acc[L][1] += 1
        ts = sum(v[0] for v in acc.values()); tn = sum(v[1] for v in acc.values())
        row = {"Model": NAME[mk], "Quant": quant}
        for L in LEVELS:
            row[L] = round(acc[L][0] / acc[L][1] * 100, 1) if acc[L][1] else np.nan
        row["Valid"] = tn
        row["Acc"] = round(ts / tn * 100, 2) if tn else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap(df, model, qorder, path, title, vmax, figsize):
    sub = df[df.Model == model].set_index("Quant").reindex(qorder)
    labels, M = [], []
    for qq in qorder:
        if qq not in sub.index or pd.isna(sub.loc[qq, "Acc"]):
            continue
        labels.append(qq)
        M.append([0 if pd.isna(sub.loc[qq, L]) else sub.loc[qq, L] for L in LEVELS] + [sub.loc[qq, "Acc"]])
    M = np.array(M, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.Blues.copy(); cmap.set_bad("#f7f7f7")
    im = ax.imshow(np.ma.masked_where(M == 0, M), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([LVL[l] for l in LEVELS] + ["总体"], fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11); ax.xaxis.tick_top()
    for i in range(len(labels)):
        for j in range(5):
            v = M[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                        color="white" if v > vmax * 0.55 else "#333333")
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03); cb.set_label("准确率 (%)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    with open(path, "wb") as f:
        fig.canvas.print_png(f)
    plt.close(fig)
    print("已生成:", path)


def bar_total(df, path):
    order = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Phi-3", "Llama-1B", "Qwen2.5-7B", "Llama-8B"]
    lab = {"Qwen2.5-0.5B": "Qwen0.5B", "Qwen2.5-1.5B": "Qwen1.5B", "Qwen2.5-3B": "Qwen3B",
           "Phi-3": "Phi-3", "Llama-1B": "Llama-1B", "Qwen2.5-7B": "Qwen7B", "Llama-8B": "Llama8B"}
    qo = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = np.arange(len(order)); w = 0.11
    for k, qq in enumerate(qo):
        vals = []
        for m in order:
            row = df[(df.Model == m) & (df.Quant == qq)]
            vals.append(float(row["Acc"].iloc[0]) if len(row) and not pd.isna(row["Acc"].iloc[0]) else 0.0)
        pos = x + (k - 3) * w
        ax.bar(pos, vals, width=w, color=COL[qq], label=qq.replace("FP16", "F16"), edgecolor="none")
        for xi, v in zip(pos, vals):
            if v > 0:
                ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in order], fontsize=11)
    ax.set_ylabel("准确率 (%)", fontsize=11); ax.set_ylim(0, 68)
    ax.set_title("图10 各模型量化档位的多步推理总准确率（统一 400 题全量有效响应）", fontsize=13, pad=12)
    ax.legend(ncol=7, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.22, top=0.88)
    fig.text(0.01, 0.02, "注：Qwen2.5-7B 与 Llama-8B 的 F16 档因 16GB 内存 OOM 输出为空（准确率记 0），其量化档可正常运行，故不宜将这两档的 F16 与量化档直接比较。",
             fontsize=8.5, color="#555555")
    with open(path, "wb") as f:
        fig.canvas.print_png(f)
    plt.close(fig)
    print("已生成:", path)


def ca_heatmap(ca_csv, path):
    t8 = pd.read_csv(ca_csv, encoding="utf-8-sig")
    t8["CA"] = pd.to_numeric(t8["CA"], errors="coerce")
    models = ["Llama-1B", "Phi-3", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Llama-8B"]
    quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    M = np.full((len(models), len(quants)), np.nan)
    for i, m in enumerate(models):
        for j, qq in enumerate(quants):
            row = t8[(t8.Model == m) & (t8.Quant == qq)]
            if len(row):
                M[i, j] = float(row["CA"].iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cmap = plt.cm.Blues.copy(); cmap.set_bad("#f5f5f5")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(quants))); ax.set_xticklabels(quants, fontsize=10)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(len(quants)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 55 else "#333333")
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03); cb.set_label("CA (%)", fontsize=10)
    ax.set_title("图13 实例级行为一致率（CA）热力图（FP16 vs 量化，ARC-Easy 全量交集）", fontsize=12, pad=12)
    ax.set_xticks(np.arange(-.5, len(quants), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    with open(path, "wb") as f:
        fig.canvas.print_png(f)
    plt.close(fig)
    print("已生成:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--ca-table", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    df = build_stats(a.raw, a.questions)
    df.to_csv(os.path.join(a.fig_dir, "reason_levels.csv"), index=False, encoding="utf-8-sig")
    heatmap(df, "Qwen2.5-1.5B", ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"],
            os.path.join(a.fig_dir, "fig08.png"),
            "图8 Qwen2.5-1.5B 多步推理得分矩阵（统一 400 题，按推理步数分层，单位 %）", 50, (11, 4.6))
    heatmap(df, "Qwen2.5-3B", ["FP16", "Q4_K_M", "Q3_K_M", "Q2_K"],
            os.path.join(a.fig_dir, "fig09.png"),
            "图9 Qwen2.5-3B 多步推理得分矩阵（统一 400 题，按推理步数分层，单位 %）", 80, (8.2, 4.6))
    bar_total(df, os.path.join(a.fig_dir, "fig10.png"))
    ca_heatmap(a.ca_table, os.path.join(a.fig_dir, "fig13.png"))


if __name__ == "__main__":
    main()
