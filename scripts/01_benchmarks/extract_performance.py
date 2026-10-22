# C:\Users\cjy13\quant_eval\04_scripts\extract_performance.py
import os, json, re, pandas as pd

root = r"C:\Users\cjy13\quant_eval"
perf_dir = os.path.join(root, "05_raw", "perf")
rss_dir = os.path.join(root, "05_raw", "rss")
ctx_dir = os.path.join(root, "05_raw", "ctx")
out_csv = os.path.join(root, "06_results", "table2_engineering.csv")

rows = []

# 1. 读取 llama-bench 的 JSON 结果
for filename in os.listdir(perf_dir):
    if not filename.endswith(".json") or "power" in filename or "thread" in filename:
        continue

    # 文件名格式：qwen1.5b-Q4_K_M.json
    model_name = filename.replace(".json", "")
    filepath = os.path.join(perf_dir, filename)

    with open(filepath, 'r', encoding='utf-16') as f:
        content = f.read()

    # 解析 llama-bench JSON（它是一个列表）
    try:
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            # 找 pp512 和 tg128
            pp512 = next((item["avg_ts"] for item in data if item.get("n_prompt") == 512), None)
            tg128 = next((item["avg_ts"] for item in data if item.get("n_gen") == 128), None)

            rows.append({
                "Model": "-".join(model_name.split("-")[:-1]),
                "Quant": model_name.split("-")[-1],
                "pp512 (t/s)": round(pp512, 2) if pp512 else None,
                "tg128 (t/s)": round(tg128, 2) if tg128 else None,
                "RSS (MiB)": None,
                "Size (MiB)": None,
                "KV_Cache": "f16",
                "Power": "AC"
            })
    except Exception as e:
        print(f"解析 {filename} 失败: {e}")

# 2. 读取 RSS 的 CSV 结果
if os.path.exists(rss_dir):
    for filename in os.listdir(rss_dir):
        if filename.endswith(".csv"):
            model_name = filename.replace(".csv", "")
            filepath = os.path.join(rss_dir, filename)
            try:
                df = pd.read_csv(filepath)
                # 假设 CSV 里有一列叫 Peak_RSS_MiB
                rss_val = df["Peak_RSS_MiB"].max() if "Peak_RSS_MiB" in df.columns else df.iloc[:, -1].max()
                # 更新 rows 里对应的模型
                for r in rows:
                    if f"{r['Model']}-{r['Quant']}" == model_name:
                        r["RSS (MiB)"] = round(rss_val, 2)
            except:
                pass

# 3. 计算模型磁盘体积
gguf_dir = os.path.join(root, "02_gguf", "quantized")
for r in rows:
    filename = f"{r['Model']}-{r['Quant']}.gguf"
    filepath = os.path.join(gguf_dir, filename)
    if os.path.exists(filepath):
        r["Size (MiB)"] = round(os.path.getsize(filepath) / (1024 * 1024), 2)

# 4. 导出
if not rows:
    print("❌ 警告：没有找到工程轨数据，请先跑 llama-bench 和 measure_rss.ps1")
else:
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Model", "Quant"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 表2 数据提取完毕，共 {len(df)} 条记录，已保存至: {out_csv}")
    print(df.to_string())