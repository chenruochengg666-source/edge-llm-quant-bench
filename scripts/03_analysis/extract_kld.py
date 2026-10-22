# C:\Users\cjy13\quant_eval\04_scripts\extract_kld.py
import os, re, pandas as pd

log_dir = r"C:\Users\cjy13\quant_eval\05_raw\kld"
out_csv = r"C:\Users\cjy13\quant_eval\06_results\table3_fidelity.csv"

data = []

for filename in os.listdir(log_dir):
    if not filename.endswith("-v2.log") or "FP16" in filename:
        continue

    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    content = raw_bytes.decode('utf-8', errors='ignore').replace('\x00', '')

    # ====== 用正则匹配，容忍任意数量的空格和乱码分隔符 ======
    def grab(pattern, content, group=0):
        m = re.search(pattern, content, re.IGNORECASE)
        return float(m.group(1)) if m else None

    ppl_base = grab(r'Mean\s+PPL\(base\)[^\d\-]*([\d\.]+)', content)
    ppl_q    = grab(r'Mean\s+PPL\(Q\)[^\d\-]*([\d\.]+)', content)
    kld_mean = grab(r'Mean\s+KLD\s*:[^\d\-]*([\d\.]+)', content)
    kld_sd   = grab(r'Mean\s+KLD\s*:[^\d\-]*[\d\.]+\s*[^\d\-]*([\d\.]+)', content)
    rms_dp   = grab(r'RMS\s+[^\n:]*:[^\d\-]*([\d\.]+)', content)
    top_p_match = re.search(r'Same\s+top\s+p[\s\S]*?:\s*([\d\.]+)\s*%', content)
    top_p = float(top_p_match.group(1)) if top_p_match else None

    # 合理性检查
    if ppl_q is None or ppl_base is None or ppl_q < 1.0:
        continue

    data.append({
        "Model": filename.split("-")[0],
        "Quant": "-".join(filename.split("-")[1:-1]).replace("-v2", ""),
        "PPL_FP16": round(ppl_base, 4),
        "PPL_Quant": round(ppl_q, 4),
        "KLD_mean": round(kld_mean, 4) if kld_mean else None,
        "KLD_sd":   round(kld_sd, 4) if kld_sd else None,
        "RMS_dp":   round(rms_dp, 4) if rms_dp else None,
        "Same_top_p": round(top_p, 4) if top_p else None
    })

df = pd.DataFrame(data).sort_values(by=["Model", "Quant"]).reset_index(drop=True)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"✅ 提取完成，共 {len(df)} 条")
print(df.to_string())