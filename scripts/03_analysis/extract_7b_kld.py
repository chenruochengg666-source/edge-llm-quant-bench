# C:\Users\cjy13\quant_eval\04_scripts\extract_7b_kld.py
import os
import re
import pandas as pd

LOG_DIR = r"C:\Users\cjy13\quant_eval\05_raw\kld_logs"
OUT_CSV = r"C:\Users\cjy13\quant_eval\06_results\table7b_kld_summary.csv"

target_files = {
    "7b-F16-ppl.log": ("Qwen7B", "F16"),
    "7b-Q4_K_M-kld.log": ("Qwen7B", "Q4_K_M"),
    "7b-Q2_K-kld.log": ("Qwen7B", "Q2_K")
}

results = []


# 强力安全读取：兼容 PowerShell 的 UTF-16 和大文件
def read_file_safely(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    # 按 utf-8 忽略错误读取，并强行移除所有 UTF-16 产生的空字节 \x00 和 BOM 头
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
        # 匹配你特有的简化格式：[5]1.7359
        match = re.search(r"\[5\]\s*([\d\.]+)", content)
        if match:
            row["PPL_Mean"] = float(match.group(1))
            # 因为你终端里没重定向到 F16 的标准差，这里用你截图终端里的实际数据 0.06509
            row["PPL_SD"] = 0.06509
    else:
        # 匹配：Mean PPL(Q)  :  1.914221 卤  0.073979
        ppl_match = re.search(r"Mean\s+PPL\(Q\)\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if ppl_match:
            row["PPL_Mean"] = float(ppl_match.group(1))
            row["PPL_SD"] = float(ppl_match.group(2))

        # 匹配：Mean KLD:  0.124878 卤  0.008144
        kld_match = re.search(r"Mean\s+KLD\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if kld_match:
            row["KLD_Mean"] = float(kld_match.group(1))
            row["KLD_SD"] = float(kld_match.group(2))

        # 匹配：Same top p: 90.118 卤 0.836 %
        top_p_match = re.search(r"Same\s+top\s+p\s*:\s*([\d\.]+)[^\d]+([\d\.]+)", content)
        if top_p_match:
            row["Same_top_p_%"] = float(top_p_match.group(1))

    results.append(row)

# 保存
if results:
    df = pd.DataFrame(results)
    quant_order = pd.CategoricalDtype(categories=["F16", "Q4_K_M", "Q2_K"], ordered=True)
    df["Quant"] = df["Quant"].astype(quant_order)
    df = df.sort_values("Quant")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 提取成功！已保存至: {OUT_CSV}")
    print("\n--- 7B 规模外推数据预览 ---")
    print(df.to_string(index=False))