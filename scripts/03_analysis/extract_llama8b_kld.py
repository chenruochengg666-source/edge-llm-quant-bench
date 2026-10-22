# C:\Users\cjy13\quant_eval\04_scripts\extract_llama8b_kld.py
import os
import re
import pandas as pd

# 配置日志目录和输出路径
LOG_DIR = r"C:\Users\cjy13\quant_eval\05_raw\kld_logs"
OUT_CSV = r"C:\Users\cjy13\quant_eval\06_results\table_llama8b_kld_summary.csv"

# 目标文件及对应的模型、量化档位
target_files = {
    "llama8b-F16-ppl.log": ("Llama8B", "F16"),
    "llama8b-Q4_K_M-kld.log": ("Llama8B", "Q4_K_M"),
    "llama8b-Q2_K-kld.log": ("Llama8B", "Q2_K")
}

results = []


# 强力安全读取（解决 PowerShell 产生的 UTF-16 空字节和乱码）
def read_file_safely(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    return raw.decode('utf-8', errors='ignore').replace('\x00', '').replace('\ufeff', '')


for filename, (model, quant) in target_files.items():
    filepath = os.path.join(LOG_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ 找不到文件: {filepath}")
        continue

    content = read_file_safely(filepath)

    row = {
        "Model": model, "Quant": quant,
        "PPL_Mean": None, "PPL_SD": None,
        "KLD_Mean": 0.0, "KLD_SD": 0.0,
        "Same_top_p_%": 100.0
    }

    if quant == "F16":
        # 匹配特殊格式：[5]1.7996
        match = re.search(r"\[5\]\s*([\d\.]+)", content)
        if match:
            row["PPL_Mean"] = float(match.group(1))
            # 标准差在 F16 的 log 里被吞了，这里从 Q4_K_M 的日志里提取到的基准值手动填入
            row["PPL_SD"] = 0.064816
    else:
        # 匹配 Mean PPL(Q)：1.914221 卤 0.073979
        ppl_match = re.search(r"Mean\s+PPL\(Q\)\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if ppl_match:
            row["PPL_Mean"] = float(ppl_match.group(1))
            row["PPL_SD"] = float(ppl_match.group(2))

        # 匹配 Mean KLD：0.124878 卤 0.008144
        kld_match = re.search(r"Mean\s+KLD\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if kld_match:
            row["KLD_Mean"] = float(kld_match.group(1))
            row["KLD_SD"] = float(kld_match.group(2))

        # 匹配 Same top p：90.118 卤 0.836 %
        top_p_match = re.search(r"Same\s+top\s+p\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if top_p_match:
            row["Same_top_p_%"] = float(top_p_match.group(1))

    results.append(row)

# 保存为 CSV
if results:
    df = pd.DataFrame(results)
    # 保证排序
    quant_order = pd.CategoricalDtype(categories=["F16", "Q4_K_M", "Q2_K"], ordered=True)
    df["Quant"] = df["Quant"].astype(quant_order)
    df = df.sort_values("Quant")

    # 保留 4 位小数，方便阅读
    df = df.round(4)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 提取成功！已保存至: {OUT_CSV}")
    print("\n--- Llama-3.1-8B 规模外推数据预览 ---")
    print(df.to_string(index=False))
else:
    print("❌ 没有提取到数据，请检查日志文件。")