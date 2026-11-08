# -*- coding: utf-8 -*-
"""
calc_zone_judge.py
依据正文 3.2 节的双轨交叉判定定义，逐档计算四类区间归属，并输出表5A 与附录表A10 的数据。
口径：
  保真度轨主判据 = sigma_pool 规则（|ΔPPL| >= 2*sqrt(sd_q^2 + sd_base^2)）
  替代口径      = ΔPPL 相对增幅 >= 5%；KLD 绝对阈值 >= 0.05
  能力轨        = 直接读取 table3_ARC_final.csv 的 BH-FDR 校正判定
用法：
  python calc_zone_judge.py --kld-log-dir ../../05_raw/kld_logs \
                            --arc-table ../../results/summary/table3_ARC_final.csv \
                            --out ../../results/summary/table5A_zone_judgement.csv
"""
import argparse, csv, math, os, re

MODEL_ALIAS = {"ll32": "Llama-1B", "phi3": "Phi-3", "qwen0.5b": "Qwen0.5B",
               "qwen1.5b": "Qwen1.5B", "qwen3b": "Qwen3B"}
RE_PPL_Q = re.compile(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*[±\+-]\s*([\d.]+)")
RE_PPL_B = re.compile(r"Mean PPL\(base\)\s*:\s*([\d.]+)\s*[±\+-]\s*([\d.]+)")
RE_KLD   = re.compile(r"Mean\s+KLD\s*:\s*([\d.]+)\s*[±\+-]\s*([\d.]+)")

def parse_log(path):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    q, b, k = RE_PPL_Q.search(txt), RE_PPL_B.search(txt), RE_KLD.search(txt)
    if not q or not b:
        return None
    return {"ppl_q": float(q.group(1)), "sd_q": float(q.group(2)),
            "ppl_b": float(b.group(1)), "sd_b": float(b.group(2)),
            "kld": float(k.group(1)) if k else None}

def load_arc(path):
    table = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["模型"].strip(), row["档位"].strip().upper())
            table[key] = row["判定"].strip()
    return table

def judge_arc(arc_judge, d_pp):
    if arc_judge is None:
        return "未获取"
    if "崩溃" in arc_judge or "退化" in arc_judge:
        return "显著下降"
    if "提升" in arc_judge:
        return "显著上升"
    if arc_judge.startswith("基准"):
        return "基准"
    return "无显著变化"

def zone_of(fid, cap):
    if cap == "显著下降":
        return "同步退化区" if fid == "显著" else "沉默区"
    if cap == "显著上升":
        return "能力反升区"
    if cap == "无显著变化":
        return "保真先行区" if fid == "显著" else "统计等价区"
    return "基准"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kld-log-dir", required=True)
    ap.add_argument("--arc-table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arc = load_arc(a.arc_table)
    base = {}
    rows = []
    files = sorted(f for f in os.listdir(a.kld_log_dir) if f.endswith("-kld.txt"))
    for fn in files:
        m = re.match(r"^(.+?)-(Q\d.*|F16)-kld\.txt$", fn)
        if not m:
            continue
        model_raw, quant = m.group(1), m.group(2).upper()
        rec = parse_log(os.path.join(a.kld_log_dir, fn))
        if rec is None:
            continue
        if model_raw not in base:
            base[model_raw] = (rec["ppl_b"], rec["sd_b"])
        bp, bs = base[model_raw]
        dppl = rec["ppl_q"] - bp
        sigma = 2 * math.sqrt(rec["sd_q"] ** 2 + bs ** 2)
        fid = "显著" if abs(dppl) >= sigma else "未显著"
        rel = dppl / bp * 100.0
        fid5 = "显著" if rel >= 5.0 else "未显著"
        fidk = "显著" if (rec["kld"] or 0) >= 0.05 else "未显著"
        model = MODEL_ALIAS.get(model_raw, model_raw)
        aj = arc.get((model, quant))
        cap = judge_arc(aj, dppl)
        rows.append([model, quant, round(bp, 3), round(rec["ppl_q"], 3), round(dppl, 3),
                     round(sigma, 3), fid, round(rel, 1), fid5,
                     "" if rec["kld"] is None else round(rec["kld"], 4), fidk,
                     (aj or ""), cap, zone_of(fid, cap),
                     "是" if len({fid, fid5, fidk}) > 1 else "否"])
    rows.sort(key=lambda r: (r[0], r[1]))
    header = ["Model", "Quant", "PPL_FP16", "PPL_Quant", "dPPL", "sigma_pool",
              "Judge_2sigma", "RelIncrease_pct", "Judge_5pct", "KLD", "Judge_KLD",
              "ARC_judge_raw", "Capability_branch", "Zone", "Judge_Conflict"]
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("已输出 %d 档判定 -> %s" % (len(rows), a.out))
    conflicts = [r for r in rows if r[14] == "是"]
    print("三口径分歧档位 %d 个：%s" % (len(conflicts), [(r[0], r[1]) for r in conflicts]))

if __name__ == "__main__":
    main()
