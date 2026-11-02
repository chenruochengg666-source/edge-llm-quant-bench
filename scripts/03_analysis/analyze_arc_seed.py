# analyze_arc_seed.py (v2)
# 修正要点（v1 → v2）：
#   1) 结论判定以「跨 seed 标准差 SD」为主判据（SD<1pp 视为稳定），ANOVA 仅作参考
#      —— v1 把 ANOVA p>0.05 与 SD<1 并列，但 n=500 时 ANOVA 过度敏感，易误判
#   2) 新增「跨 seed 逐题一致率」：同一题在 5 个 seed 下对错是否完全一致
#   3) 绘图改用 matplotlib（bar + errorbar），不再依赖 seaborn 的 errorbar 参数名差异
#   4) 兼容旧数据：若无 is_correct 字段，则用 clean_pred == answer 现算
import os, json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RAW_DIR     = r"C:\Users\cjy13\quant_eval\05_raw\arc_seed"
RES_DIR     = r"C:\Users\cjy13\quant_eval\06_results"
FIG_DIR     = os.path.join(RES_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_SUMMARY = os.path.join(RES_DIR, "table_p03_seed_anova.csv")
OUT_CONSIST = os.path.join(RES_DIR, "table_p03_seed_consistency.csv")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

def parse_name(f):
    b = f.replace(".gguf_seed.jsonl", "").replace(".gguf", "").lower()
    if "1.5b" in b: return "Qwen1.5B"
    if "3b" in b:    return "Qwen3B"
    return f

def parse_quant(f):
    b = f.lower()
    if "base" in b or "f16" in b: return "F16"
    if "q4_k_m" in b: return "Q4_K_M"
    if "q2_k" in b:   return "Q2_K"
    return "UNKNOWN"

def main():
    recs = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".jsonl"): continue
        model, quant = parse_name(fname), parse_quant(fname)
        with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: r = json.loads(line)
                except Exception: continue
                st = str(r.get("status", ""))
                # 仅排除技术故障（请求失败/服务异常）；empty（有输出但无可用字母）属于模型能力问题，计入分母并算错
                if st.startswith("HTTP_") or st.startswith("EXC_") or st in ("UNKNOWN", "NO_OUTPUT"):
                    continue
                ic = r.get("is_correct")
                if ic is None:
                    ic = 1 if (st == "ok" and r.get("clean_pred") == r.get("answer")) else 0
                is_parse = 1 if st == "ok" else 0
                recs.append({"Model": model, "Quant": quant, "Seed": r["seed"],
                             "Question_ID": r["id"], "Is_Correct": int(ic), "Is_Parseable": is_parse})

    df = pd.DataFrame(recs)
    if df.empty:
        print("[ERR] 无有效记录，请先运行 run_arc_seed.py")
        return
    print(f"有效记录 {len(df)} 条 | 模型档位 {df.groupby(['Model','Quant']).ngroups} 个")

    # 每档位 × seed 的准确率
    seed_acc = (df.groupby(["Model","Quant","Seed"])["Is_Correct"]
                  .agg(["mean","count"]).reset_index())
    seed_acc["Accuracy"] = seed_acc["mean"] * 100

    rows, cons = [], []
    for (m, q), g in df.groupby(["Model","Quant"]):
        accs = seed_acc[(seed_acc.Model == m) & (seed_acc.Quant == q)]["Accuracy"].values
        n_per_seed = g.groupby("Seed").size().iloc[0]

        mean_acc = float(np.mean(accs))
        sd_acc   = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0

        # ANOVA（n 较大时仅作参考）
        groups = [g[g.Seed == s]["Is_Correct"].values for s in sorted(g.Seed.unique())]
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_val = float("nan"), 1.0

        # 跨 seed 逐题一致率
        pivot = g.pivot_table(index="Question_ID", columns="Seed", values="Is_Correct")
        if pivot.shape[1] > 1:
            consistency = float((pivot.nunique(axis=1) == 1).mean() * 100)
        else:
            consistency = 100.0

        verdict = "稳定（SD<1pp）" if sd_acc < 1.0 else "不稳定（SD≥1pp）"
        parse_rate = float(g["Is_Parseable"].mean() * 100)
        rows.append({"Model": m, "Quant": q, "N_per_seed": int(n_per_seed),
                     "Parse_Rate (%)": round(parse_rate, 1),
                     "Mean Acc (%)": round(mean_acc, 2), "SD (pp)": round(sd_acc, 2),
                     "Min Acc (%)": round(float(np.min(accs)), 2),
                     "Max Acc (%)": round(float(np.max(accs)), 2),
                     "ANOVA p": round(float(p_val), 4),
                     "Item Consistency (%)": round(consistency, 1),
                     "Conclusion": verdict})
        cons.append({"Model": m, "Quant": q, "Item Consistency (%)": round(consistency, 1)})

    df_sum = pd.DataFrame(rows)
    order_m = ["Qwen1.5B", "Qwen3B"]; order_q = ["F16", "Q4_K_M", "Q2_K"]
    df_sum["Model"] = pd.Categorical(df_sum["Model"], order_m, ordered=True)
    df_sum["Quant"] = pd.Categorical(df_sum["Quant"], order_q, ordered=True)
    df_sum = df_sum.sort_values(["Model","Quant"])
    df_sum.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n=== P0-3 种子稳定性（主判据：SD）===")
    print(df_sum.to_string(index=False))
    print(f"\n[OK] 表格: {OUT_SUMMARY}")

    # 绘图：matplotlib
    labels, means, sds, colors = [], [], [], []
    cmap = {"F16": "#5B9BD5", "Q4_K_M": "#70AD47", "Q2_K": "#C00000"}
    for _, r in df_sum.iterrows():
        labels.append(f"{r['Model']}\n{r['Quant']}")
        means.append(r["Mean Acc (%)"]); sds.append(r["SD (pp)"])
        colors.append(cmap.get(str(r["Quant"]), "#808080"))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="black", capsize=4, linewidth=1)
    for xi, mv, sv in zip(x, means, sds):
        ax.text(xi, mv + max(sv, 1.2) + 1.0, f"{mv:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ARC-Easy Accuracy (%)", fontsize=12)
    ax.set_title("ARC-Easy Accuracy across Random Seeds (42–46), error bar = SD", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100); ax.grid(axis="y", linestyle="--", alpha=0.45)
    handles = [plt.Rectangle((0,0),1,1, color=v) for v in cmap.values()]
    ax.legend(handles, list(cmap.keys()), title="Quant", loc="upper right", frameon=True)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_p03_seed_stability.pdf")
    plt.savefig(fig_path); plt.close()
    print(f"[OK] 图: {fig_path}")

if __name__ == "__main__":
    main()
