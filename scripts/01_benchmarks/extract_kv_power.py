# C:\Users\cjy13\quant_eval\04_scripts\extract_kv_power.py
import os, json, pandas as pd

root = r"C:\Users\cjy13\quant_eval"
ctx_dir = os.path.join(root, "05_raw", "ctx")
perf_dir = os.path.join(root, "05_raw", "perf")

# ============ 通用读取函数：兼容 PowerShell 输出的 UTF-16 ============
def load_json(filepath):
    """自动兼容 UTF-16 和 UTF-8 编码"""
    with open(filepath, 'rb') as f:
        raw = f.read()
    # 尝试 UTF-16 (PowerShell 默认重定向编码)，失败则用 UTF-8
    try:
        content = raw.decode('utf-16')
    except:
        content = raw.decode('utf-8', errors='ignore')
    # 清理可能残留的 BOM 头
    content = content.replace('\ufeff', '')
    return json.loads(content)

# ============ 1. 解析 KV Cache 消融 ============
kv_rows = []
for kv_type in ["f16", "q8_0"]:
    filepath = os.path.join(ctx_dir, f"kv_{kv_type}.json")
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到 {filepath}，跳过")
        continue
    try:
        data = load_json(filepath)
        pp_val = next((x["avg_ts"] for x in data if x.get("n_prompt") == 4096), None)
        tg_val = next((x["avg_ts"] for x in data if x.get("n_gen") == 128), None)
        if pp_val:
            kv_rows.append({
                "Model": "qwen1.5b",
                "Quant": "Q4_K_M",
                "KV_Cache_Type": kv_type,
                "Context_Length": 4096,
                "pp4096 (t/s)": round(pp_val, 2),
                "tg128 (t/s)": round(tg_val, 2) if tg_val else None
            })
    except Exception as e:
        print(f"❌ 解析 {filepath} 失败: {e}")

if kv_rows:
    df_kv = pd.DataFrame(kv_rows)
    out_kv = os.path.join(root, "06_results", "tableA3_kv_cache.csv")
    os.makedirs(os.path.dirname(out_kv), exist_ok=True)
    df_kv.to_csv(out_kv, index=False, encoding='utf-8-sig')
    print(f"\n✅ KV Cache 表已保存: {out_kv}")
    print(df_kv.to_string(index=False))

# ============ 2. 解析电源模式 ============
power_rows = []
for mode, filename in [("Battery_Saver", "power_battery.json"), ("AC_High_Perf", "power_ac.json")]:
    filepath = os.path.join(perf_dir, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到 {filepath}，跳过")
        continue
    try:
        data = load_json(filepath)
        pp_val = next((x["avg_ts"] for x in data if x.get("n_prompt") == 512), None)
        tg_val = next((x["avg_ts"] for x in data if x.get("n_gen") == 128), None)
        if pp_val:
            power_rows.append({
                "Model": "qwen1.5b",
                "Quant": "Q4_K_M",
                "Power_Mode": mode,
                "Context_Length": 512,
                "pp512 (t/s)": round(pp_val, 2),
                "tg128 (t/s)": round(tg_val, 2) if tg_val else None
            })
    except Exception as e:
        print(f"❌ 解析 {filepath} 失败: {e}")

if power_rows:
    df_power = pd.DataFrame(power_rows)
    out_power = os.path.join(root, "06_results", "tableA2_power.csv")
    os.makedirs(os.path.dirname(out_power), exist_ok=True)
    df_power.to_csv(out_power, index=False, encoding='utf-8-sig')
    print(f"\n✅ 电源模式表已保存: {out_power}")
    print(df_power.to_string(index=False))