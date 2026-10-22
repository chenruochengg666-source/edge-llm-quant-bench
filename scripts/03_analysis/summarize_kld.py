# C:\Users\cjy13\quant_eval\04_scripts\summarize_kld.py
import os
import re
import pandas as pd

# 绝对路径配置
KLD_LOG_DIR = r"C:\Users\cjy13\quant_eval\05_raw\kld_logs"
OUT_CSV = r"C:\Users\cjy13\quant_eval\06_results\table_kld_summary.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 解析结果存储
rows = []

print(f"开始扫描目录: {KLD_LOG_DIR}")

if not os.path.exists(KLD_LOG_DIR):
    print(f"❌ 找不到日志目录: {KLD_LOG_DIR}，请先运行 run_kld_logits.ps1 生成日志。")
    exit(1)

# 遍历所有日志文件
for filename in os.listdir(KLD_LOG_DIR):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(KLD_LOG_DIR, filename)

    # 从文件名解析 Model 和 Quant
    # 文件名格式预期为: "qwen1.5b-Q4_K_M-kld.txt"
    match = re.match(r"^(.+?)-(Q\d.*|F16)-kld\.txt$", filename)
    if not match:
        print(f"⚠️ 文件名格式不符，跳过: {filename}")
        continue

    model_name = match.group(1)
    quant_name = match.group(2)

    # 读取日志内容（注意捕获潜在的编码问题）
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件 {filename}: {e}")
        continue

    # 1. 提取 Mean KLD 均值及其标准差
    # 匹配格式: Mean KLD: 1.117978 ± 0.019330
    kld_match = re.search(r"Mean\s+KLD:\s*([\d\.]+)\s*±\s*([\d\.]+)", content)
    if kld_match:
        mean_kld = float(kld_match.group(1))
        std_kld = float(kld_match.group(2))
    else:
        print(f"⚠️ 在 {filename} 中找不到 'Mean KLD'，跳过该指标")
        mean_kld = None
        std_kld = None

    # 2. 提取 Same top p 及其标准差
    # 匹配格式: Same top p: 59.078 ± 0.689 %
    top_p_match = re.search(r"Same\s+top\s+p:\s*([\d\.]+)\s*±\s*([\d\.]+)", content)
    if top_p_match:
        same_top_p = float(top_p_match.group(1))
        std_top_p = float(top_p_match.group(2))
    else:
        print(f"⚠️ 在 {filename} 中找不到 'Same top p'，跳过该指标")
        same_top_p = None
        std_top_p = None

    # 3. 提取 PPL 均值（可选，用于验证）
    ppl_match = re.search(r"Mean\s+PPL\(Q\)\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)", content)
    if ppl_match:
        mean_ppl = float(ppl_match.group(1))
        std_ppl = float(ppl_match.group(2))
    else:
        mean_ppl = None
        std_ppl = None

    rows.append({
        "Model": model_name,
        "Quant": quant_name,
        "Mean_PPL": mean_ppl,
        "PPL_SD": std_ppl,
        "Mean_KLD": mean_kld,
        "KLD_SD": std_kld,
        "Same_top_p_%": same_top_p,
        "Top_p_SD": std_top_p
    })

# 生成 DataFrame 并排序
if rows:
    df = pd.DataFrame(rows)
    # 按照 Model 和 Quant 进行排序，方便阅读
    df = df.sort_values(by=["Model", "Quant"]).reset_index(drop=True)

    # 保存为 CSV
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 汇总完成！已保存到: {OUT_CSV}")
    print("\n--- 汇总结果预览 ---")
    print(df.to_string(index=False))
else:
    print("❌ 没有提取到任何有效数据，请检查日志文件内容。")