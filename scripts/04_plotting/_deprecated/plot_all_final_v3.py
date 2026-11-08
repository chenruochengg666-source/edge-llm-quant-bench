# C:\Users\cjy13\quant_eval\04_scripts\plot_all_final_v3.py
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ========== 1. 全局设置 ==========
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

root = r"C:\Users\cjy13\quant_eval"
res_dir = os.path.join(root, "06_results")
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


# ========== 2. 安全读取 ==========
def safe_read(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            if "Quant" in df.columns:
                df["Quant"] = df["Quant"].astype(str).str.upper()
            return df
        except pd.errors.EmptyDataError:
            print(f"⚠️ 文件为空 {os.path.basename(path)}，相关图表将跳过。")
            return pd.DataFrame()
    print(f"⚠️ 找不到 {os.path.basename(path)}，相关图表将跳过。")
    return pd.DataFrame()


# 数据源
df_eng = safe_read(os.path.join(res_dir, "table2_engineering.csv"))
df_kld = safe_read(os.path.join(res_dir, "table_kld_summary.csv"))   # 新增：KLD 汇总
df_arc = safe_read(os.path.join(res_dir, "table4_downstream_arc.csv"))
df_ca = safe_read(os.path.join(res_dir, "table7_ca_kappa_final.csv"))
df_power = safe_read(os.path.join(res_dir, "tableA2_power.csv"))
df_kv = safe_read(os.path.join(res_dir, "tableA3_kv_cache.csv"))
df_reason = safe_read(os.path.join(res_dir, "table5_reasoning_final_summary.csv"))


# ========== 3. fig01：量化档位 vs pp512/tg128 ==========
if not df_eng.empty:
    df1 = df_eng[df_eng["Model"] == "qwen1.5b"].copy()
    df1["Quant"] = pd.Categorical(df1["Quant"], categories=QUANT_ORDER, ordered=True)
    df1 = df1.sort_values("Quant").dropna(subset=["Quant"])
    x_axis = df1["Quant"].astype(str).tolist()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Quantization Level")
    ax1.set_ylabel("Prefill pp512 (t/s)", color="tab:blue")
    ax1.plot(x_axis, df1["pp512 (t/s)"], "s-", color="tab:blue", label="pp512")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Decode tg128 (t/s)", color="tab:red")
    ax2.plot(x_axis, df1["tg128 (t/s)"], "o-", color="tab:red", label="tg128")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title("Qwen1.5B: Quantization vs Performance")
    plt.savefig(os.path.join(fig_dir, "fig01_performance.pdf"))
    plt.close()
    print("✅ fig01_performance.pdf")


# ========== 4. fig02：KLD vs ARC-Easy（双轴，带误差棒） ==========
if not df_kld.empty and not df_arc.empty:
    df_k = df_kld[df_kld["Model"] == "qwen1.5b"].copy()
    df_a = df_arc[df_arc["Model"] == "qwen1.5b"].copy()
    merged = pd.merge(df_a, df_k, on=["Model", "Quant"], how="inner")
    merged["Quant"] = pd.Categorical(merged["Quant"], categories=QUANT_ORDER, ordered=True)
    merged = merged.sort_values("Quant").dropna(subset=["Quant"])
    x_axis = merged["Quant"].astype(str).tolist()

    if len(merged) > 0:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel("Quantization Level")
        ax1.set_ylabel("KLD (log scale)", color="tab:red")
        ax1.errorbar(x_axis, merged["Mean_KLD"], yerr=merged["KLD_SD"],
                     fmt="o-", color="tab:red", capsize=3, label="KLD")
        ax1.set_yscale("log")
        ax1.tick_params(axis='y', labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("ARC-Easy Accuracy", color="tab:blue")
        ax2.plot(x_axis, merged["Accuracy"], "s-", color="tab:blue", label="ARC")
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        plt.title("Qwen1.5B: KLD vs ARC-Easy")
        plt.savefig(os.path.join(fig_dir, "fig02_kld_vs_arc.pdf"))
        plt.close()
        print("✅ fig02_kld_vs_arc.pdf")


# ========== 5. fig03：1.5B vs 3B PPL/KLD ==========
if not df_kld.empty:
    df_15b = df_kld[df_kld["Model"] == "qwen1.5b"].copy()
    df_3b = df_kld[df_kld["Model"] == "qwen3b"].copy()

    for df in [df_15b, df_3b]:
        df["Quant"] = pd.Categorical(df["Quant"], categories=QUANT_ORDER, ordered=True)
        df.sort_values("Quant", inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(df_15b["Quant"].astype(str), df_15b["Mean_PPL"],
                 yerr=df_15b["PPL_SD"], fmt="o-", capsize=3, label="Qwen1.5B")
    ax1.errorbar(df_3b["Quant"].astype(str), df_3b["Mean_PPL"],
                 yerr=df_3b["PPL_SD"], fmt="s--", capsize=3, label="Qwen3B")
    ax1.set_yscale("log")
    ax1.set_ylabel("PPL")
    ax1.set_title("PPL vs Quantization")
    ax1.legend()

    ax2.errorbar(df_15b["Quant"].astype(str), df_15b["Mean_KLD"],
                 yerr=df_15b["KLD_SD"], fmt="o-", capsize=3, label="Qwen1.5B")
    ax2.errorbar(df_3b["Quant"].astype(str), df_3b["Mean_KLD"],
                 yerr=df_3b["KLD_SD"], fmt="s--", capsize=3, label="Qwen3B")
    ax2.set_yscale("log")
    ax2.set_ylabel("KLD")
    ax2.set_title("KLD vs Quantization")
    ax2.legend()

    plt.savefig(os.path.join(fig_dir, "fig03_ppl_kld_scale.pdf"))
    plt.close()
    print("✅ fig03_ppl_kld_scale.pdf")


# ========== 6. fig04：Qwen1.5B 关键指标热力图（含 F16） ==========
if not df_eng.empty and not df_kld.empty and not df_arc.empty:
    df_e = df_eng[df_eng["Model"] == "qwen1.5b"].copy()
    df_k = df_kld[df_kld["Model"] == "qwen1.5b"].copy()
    df_a = df_arc[df_arc["Model"] == "qwen1.5b"].copy()

    df_h = pd.merge(df_e, df_k, on=["Model", "Quant"], how="inner")
    df_h = pd.merge(df_h, df_a, on=["Model", "Quant"], how="inner")

    # 手动补 F16（保真度指标为基准，ARC 用 table4 里 F16 行）
    f16_arc = df_a[df_a["Quant"] == "F16"]
    if not f16_arc.empty:
        f16_row = {
            "Model": "qwen1.5b", "Quant": "F16",
            "Mean_PPL": df_h["Mean_PPL"].iloc[0] if "Mean_PPL" in df_h else np.nan,
            "Mean_KLD": 0.0,
            "Accuracy": f16_arc["Accuracy"].values[0],
            "tg128 (t/s)": df_e[df_e["Quant"] == "F16"]["tg128 (t/s)"].values[0] if not df_e[df_e["Quant"] == "F16"].empty else np.nan,
            "Size (MiB)": df_e[df_e["Quant"] == "F16"]["Size (MiB)"].values[0] if not df_e[df_e["Quant"] == "F16"].empty else np.nan,
        }
        df_h = pd.concat([pd.DataFrame([f16_row]), df_h], ignore_index=True)

    df_h["Quant"] = pd.Categorical(df_h["Quant"], categories=QUANT_ORDER, ordered=True)
    df_h = df_h.sort_values("Quant").dropna(subset=["Quant"])

    cols = ["Mean_PPL", "Mean_KLD", "Accuracy", "tg128 (t/s)", "Size (MiB)"]
    available_cols = [c for c in cols if c in df_h.columns]

    if len(available_cols) >= 2:
        data_norm = df_h[available_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
        data_norm.index = df_h["Quant"].astype(str)

        plt.figure(figsize=(8, 5))
        sns.heatmap(data_norm.T, annot=df_h[available_cols].T, fmt=".2f", cmap="RdYlGn_r")
        plt.title("Qwen1.5B: Key Metrics Heatmap")
        plt.savefig(os.path.join(fig_dir, "fig04_heatmap.pdf"))
        plt.close()
        print(f"✅ fig04_heatmap.pdf（列：{available_cols}）")


# ========== 7. fig05：跨架构 ARC 对比 ==========
if not df_arc.empty:
    df_cross = df_arc[df_arc["Quant"].isin(["F16", "Q4_K_M", "Q2_K"])].copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_cross, x="Model", y="Accuracy", hue="Quant")
    plt.title("Cross-Architecture ARC-Easy Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(fig_dir, "fig05_cross_arch.pdf"))
    plt.close()
    print("✅ fig05_cross_arch.pdf")


# ========== 8. fig06：长上下文 ==========
ctx_csv = os.path.join(root, "05_raw", "ctx", "qwen1.5b-Q4_K_M.csv")
if os.path.exists(ctx_csv):
    df_ctx = pd.read_csv(ctx_csv)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_ctx["Ctx"].astype(str), df_ctx["Peak_RSS_MiB"], "o-", color="tab:red")
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Peak RSS (MiB)", color="tab:red")
    plt.title("Qwen1.5B Q4_K_M: Context Length vs Peak RSS")
    plt.savefig(os.path.join(fig_dir, "fig06_context.pdf"))
    plt.close()
    print("✅ fig06_context.pdf")


# ========== 9. fig07：电源模式 ==========
if not df_power.empty:
    plt.figure(figsize=(6, 4))
    df_power_melt = df_power.melt(id_vars=["Power_Mode"],
                                  value_vars=["pp512 (t/s)", "tg128 (t/s)"],
                                  var_name="Metric", value_name="Value")
    sns.barplot(data=df_power_melt, x="Metric", y="Value", hue="Power_Mode")
    plt.title("Power Mode Impact")
    plt.savefig(os.path.join(fig_dir, "fig07_power.pdf"))
    plt.close()
    print("✅ fig07_power.pdf")


# ========== 10. fig08：KV Cache 消融 ==========
if not df_kv.empty:
    kv_types = df_kv["KV_Cache_Type"].tolist()
    pp = df_kv["pp4096 (t/s)"].tolist()
    tg = df_kv["tg128 (t/s)"].tolist()

    x = np.arange(len(kv_types))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - width/2, pp, width, label="pp4096", color="tab:blue")
    ax1.set_ylabel("pp4096 (t/s)", color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(kv_types)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, tg, width, label="tg128", color="tab:red")
    ax2.set_ylabel("tg128 (t/s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("KV Cache Ablation (Context 4096)")
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig08_kv_cache.pdf"))
    plt.close()
    print("✅ fig08_kv_cache.pdf")


# ========== 11. fig09：并发吞吐 ==========
conc_csv = os.path.join(root, "05_raw", "concurrency", "qwen1.5b-Q4_K_M.csv")
if os.path.exists(conc_csv):
    df_conc = pd.read_csv(conc_csv)
    plt.figure(figsize=(6, 4))
    plt.plot(df_conc["Concurrency"], df_conc["Throughput_tok_s"], "o-", label="Aggregate Throughput")
    ideal = df_conc["Concurrency"] * df_conc["Throughput_tok_s"].iloc[0]
    plt.plot(df_conc["Concurrency"], ideal, "r--", label="Ideal Linear")
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (tok/s)")
    plt.title("Concurrency Throughput (Qwen1.5B Q4_K_M)")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "fig09_concurrency.pdf"))
    plt.close()
    print("✅ fig09_concurrency.pdf")


# ========== 12. fig10：体积 vs 解码速度 ==========
if not df_eng.empty:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_eng, x="Size (MiB)", y="tg128 (t/s)",
                    hue="Model", style="Model", s=100)
    plt.title("Model Size vs Decode Speed")
    plt.xlabel("File Size (MiB)")
    plt.ylabel("tg128 (t/s)")
    plt.savefig(os.path.join(fig_dir, "fig10_size_vs_speed.pdf"))
    plt.close()
    print("✅ fig10_size_vs_speed.pdf")


# ========== 13. fig11：Correctness Agreement 热力图 ==========
if not df_ca.empty and "Correctness_Agreement" in df_ca.columns:
    plt.figure(figsize=(8, 5))
    pivot_df = df_ca.pivot_table(index="Model", columns="Quant",
                                 values="Correctness_Agreement", aggfunc="mean")
    sns.heatmap(pivot_df, annot=True, cmap="Blues", fmt=".3f")
    plt.title("Correctness Agreement (FP16 vs Quantized)")
    plt.savefig(os.path.join(fig_dir, "fig11_ca_heatmap.pdf"))
    plt.close()
    print("✅ fig11_ca_heatmap.pdf")


# ========== 14. fig12：多步推理得分矩阵 ==========
if not df_reason.empty:
    for model_tag in ["qwen1.5b", "qwen3b"]:
        sub = df_reason[df_reason["Model"] == model_tag].copy()
        if sub.empty:
            continue

        sub["Quant"] = pd.Categorical(sub["Quant"], categories=QUANT_ORDER, ordered=True)
        sub = sub.sort_values("Quant")

        levels = ["2step", "3step", "4step", "5step+"]
        mat = []
        for _, r in sub.iterrows():
            row = {}
            for lv in levels:
                s = r.get(f"{lv}_Score", np.nan)
                t = r.get(f"{lv}_Total", np.nan)
                if pd.isna(s) or pd.isna(t) or t == 0:
                    row[lv] = np.nan
                else:
                    row[lv] = s / t
            mat.append(row)
        mat_df = pd.DataFrame(mat, index=sub["Quant"].astype(str))

        plt.figure(figsize=(7, 4))
        sns.heatmap(mat_df, annot=True, cmap="RdYlGn", fmt=".2f", vmin=0, vmax=1)
        plt.title(f"{model_tag}: Reasoning Score Matrix (per-level accuracy)")
        plt.xlabel("Difficulty Level")
        plt.ylabel("Quantization")
        plt.savefig(os.path.join(fig_dir, f"fig12_reason_matrix_{model_tag}.pdf"))
        plt.close()
        print(f"✅ fig12_reason_matrix_{model_tag}.pdf")


# ========== 15. fig13：多步推理总分条形图 ==========
if not df_reason.empty:
    df_r = df_reason[df_reason["Total_Score"] > 0].copy()
    df_r["Label"] = df_r["Model"] + "-" + df_r["Quant"]
    df_r = df_r.sort_values(["Model", "Quant"])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_r["Label"], df_r["Total_Score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total Score (max 120)")
    plt.title("Multi-step Reasoning: Total Score (Human Scoring)")
    for b, v in zip(bars, df_r["Total_Score"]):
        plt.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig13_reason_total_score.pdf"))
    plt.close()
    print("✅ fig13_reason_total_score.pdf")


# ========== 16. fig14：KLD / PPL / Same_top_p 三联图 ==========
if not df_kld.empty:
    df_k = df_kld[df_kld["Model"].isin(["qwen1.5b", "qwen3b"])].copy()
    df_k["Quant"] = pd.Categorical(df_k["Quant"], categories=QUANT_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for model, marker in [("qwen1.5b", "o-"), ("qwen3b", "s--")]:
        sub = df_k[df_k["Model"] == model].sort_values("Quant")
        x = sub["Quant"].astype(str).tolist()
        axes[0].errorbar(x, sub["Mean_PPL"], yerr=sub["PPL_SD"],
                         fmt=marker, capsize=3, label=model)
        axes[1].errorbar(x, sub["Mean_KLD"], yerr=sub["KLD_SD"],
                         fmt=marker, capsize=3, label=model)
        axes[2].plot(x, sub["Same_top_p_%"], marker, label=model)

    axes[0].set_yscale("log"); axes[0].set_ylabel("PPL"); axes[0].set_title("PPL")
    axes[1].set_yscale("log"); axes[1].set_ylabel("KLD"); axes[1].set_title("KLD")
    axes[2].set_ylabel("Same top p (%)"); axes[2].set_title("Same top p")
    for ax in axes:
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig14_kld_triple.pdf"))
    plt.close()
    print("✅ fig14_kld_triple.pdf")


print(f"\n✅ 所有图表绘制完毕，已保存至: {fig_dir}")