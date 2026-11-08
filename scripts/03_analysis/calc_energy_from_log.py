# -*- coding: utf-8 -*-
"""
calc_energy_from_log.py —— 由手工记录的功耗日志计算能效指标

输入：power_manual_log_template.csv（或你填好的同名文件）
输出：table_power_energy.csv，含净功耗与 J/token，可直接进论文表

说明：
    净功耗 = 负载稳态功耗 − 同时刻待机功耗（扣除屏幕/系统基础开销）
    J/token = 净功耗(W) / 解码速度(tg128, t/s)

用法：
    python calc_energy_from_log.py --csv ../../06_results\power_manual_log_template.csv --out ../../06_results\table_power_energy.csv
"""
import argparse, csv, os, sys


def to_float(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="手工记录表路径")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print("[错误] 找不到输入文件:", a.csv)
        return

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    out = []
    missing = []
    for r in rows:
        tg = to_float(r.get("tg128_tps"))
        load = to_float(r.get("Load_W"))
        idle = to_float(r.get("Idle_W"))
        net = (load - idle) if (load is not None and idle is not None) else None
        jpt = (net / tg) if (net is not None and tg) else None
        if tg is None or load is None or idle is None:
            missing.append(r.get("Group"))
        out.append({
            "Group": r.get("Group"),
            "Supply": r.get("Supply"),
            "PerfMode": r.get("PerfMode"),
            "tg128_tps": tg if tg is not None else "",
            "pp512_tps": r.get("pp512_tps"),
            "Idle_W": idle if idle is not None else "",
            "Load_W": load if load is not None else "",
            "Net_W": round(net, 2) if net is not None else "",
            "J_per_token": round(jpt, 3) if jpt is not None else "",
            "Measure_Tool": r.get("Measure_Tool"),
            "Note": r.get("Note"),
        })

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print("%-12s %-5s %-9s %8s %8s %8s %10s" % ("Group", "Supp", "Perf", "tg128", "Net_W", "J/token", "工具"))
    for r in out:
        print("%-12s %-5s %-9s %8s %8s %8s %10s" % (
            r["Group"], r["Supply"], r["PerfMode"], r["tg128_tps"] or "-",
            r["Net_W"] or "-", r["J_per_token"] or "-", r["Measure_Tool"] or "-"))

    if missing:
        print("\n[提示] 以下组还有空项未填：%s" % "、".join([m for m in missing if m]))
    print("\n已输出:", a.out)

    # 横向对比（只对填全的组）
    valid = [r for r in out if r["J_per_token"] != ""]
    if len(valid) >= 2:
        print("\n=== 相对能效对比（以最优组为 1.00）===")
        best = min(v["J_per_token"] for v in valid)
        for v in sorted(valid, key=lambda x: x["J_per_token"]):
            print("  %-12s %.3f J/token  → %.2fx" % (v["Group"], v["J_per_token"], v["J_per_token"] / best))


if __name__ == "__main__":
    main()
