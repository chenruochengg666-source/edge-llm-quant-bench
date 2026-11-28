# -*- coding: utf-8 -*-
"""
功耗补测分析：HWiNFO64 日志 × 阶段时间标记 × llama-bench 吞吐 → 每 token 能耗与频率汇总
========================================================================================
输入（由 scripts/01_benchmarks/run_supplement_tests.ps1 生成）：
  results/raw/power_hwinfo/<组>_hwinfo.CSV     HWiNFO64 传感器日志（1 s 间隔，UTF-8）
  results/raw/power_hwinfo/<组>_markers.csv    idle / decode / prefill 各阶段起止时间
  results/raw/power_hwinfo/<组>_decode.csv     llama-bench -p 0 -n 128 -r 20 结果
  results/raw/power_hwinfo/<组>_prefill.csv    llama-bench -p 512 -n 0 -r 20 结果
组：AC_HighPerf / AC_BestEff / BAT_HighPerf / BAT_BestEff（Windows“平衡”电源计划 + 最佳性能/最佳能效，未开节电模式）

口径：
  · 每阶段剔除开头 TRIM_LOAD 秒（模型加载）与结尾 1 s；待机阶段剔除开头 10 s
  · 封装功率 = HWiNFO「CPU 封装功率」（RAPL）；系统总功率 = HWiNFO「系统总功率」（PSys）
  · 净能耗 =（阶段平均功率 − 同组待机平均功率）/ 吞吐；毛能耗 = 阶段平均功率 / 吞吐
  · 交互对比 =（插电能效 − 插电性能）−（电池能效 − 电池性能），Welch 合成标准误，n = 20
用法（仓库根目录下）：python scripts/03_analysis/analyze_power_hwinfo.py
输出：results/summary/power_hwinfo_summary.csv
"""
import csv, io, math, os, statistics as st
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "raw" / "power_hwinfo"
OUT = REPO / "results" / "summary" / "power_hwinfo_summary.csv"
GROUPS = ["AC_HighPerf", "AC_BestEff", "BAT_HighPerf", "BAT_BestEff"]
TRIM_LOAD, TRIM_IDLE = 6, 10
COLS = {"pkg": "CPU 封装功率 [W]", "sys": "系统总功率 [W]", "ia": "IA 核心功率 [W]", "chg": "充电速度 [W]",
        "freq": "核心频率 (avg) [MHz]", "efreq": "核心有效频率 (avg) [MHz]", "ring": "Ring/LLC 频率 [MHz]",
        "temp": "CPU 封装 [℃]"}


def parse_time(s):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S.%f"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def load_hwinfo(path):
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    hdr = rows[0]; idx = {k: hdr.index(v) for k, v in COLS.items() if v in hdr}
    data = []
    for r in rows[1:]:
        if len(r) < len(hdr) - 5 or not r[0] or r[0] == "Date":
            continue
        ts = parse_time(r[0] + " " + r[1])
        if ts is None:
            continue
        rec = {"ts": ts}
        for k, i in idx.items():
            try:
                rec[k] = float(r[i])
            except (ValueError, IndexError):
                pass
        data.append(rec)
    return data


def bench(path):
    with open(path, encoding="utf-8-sig") as f:
        r = list(csv.DictReader(f))[0]
    return float(r["avg_ts"]), float(r["stddev_ts"])


def phase_mean(data, start, end, trim):
    seg = [x for x in data if start + timedelta(seconds=trim) <= x["ts"] <= end - timedelta(seconds=1)]
    out = {"n": len(seg)}
    for k in COLS:
        v = [x[k] for x in seg if k in x]
        if v:
            out[k] = st.mean(v)
    return out


def main():
    rows = {}
    for g in GROUPS:
        data = load_hwinfo(SRC / f"{g}_hwinfo.CSV")
        with open(SRC / f"{g}_markers.csv", encoding="utf-8-sig") as f:
            marks = {m["phase"]: (parse_time(m["start"]), parse_time(m["end"])) for m in csv.DictReader(f)}
        ph = {p: phase_mean(data, *marks[p], TRIM_IDLE if p == "idle" else TRIM_LOAD) for p in ("idle", "decode", "prefill")}
        tg, tg_sd = bench(SRC / f"{g}_decode.csv")
        pp, pp_sd = bench(SRC / f"{g}_prefill.csv")
        i, d, p = ph["idle"], ph["decode"], ph["prefill"]
        rows[g] = {
            "group": g, "pp512": pp, "pp512_sd": pp_sd, "tg128": tg, "tg128_sd": tg_sd,
            "idle_pkg_W": i["pkg"], "decode_pkg_W": d["pkg"], "prefill_pkg_W": p["pkg"],
            "idle_sys_W": i["sys"], "decode_sys_W": d["sys"], "prefill_sys_W": p["sys"],
            "decode_freq_MHz": d["freq"], "decode_eff_freq_MHz": d["efreq"], "decode_ring_MHz": d["ring"],
            "prefill_eff_freq_MHz": p["efreq"], "prefill_ring_MHz": p["ring"],
            "decode_temp_C": d["temp"], "prefill_temp_C": p["temp"],
            "decode_J_per_tok_pkg": d["pkg"] / tg, "decode_netJ_per_tok_pkg": (d["pkg"] - i["pkg"]) / tg,
            "decode_netJ_per_tok_sys": (d["sys"] - i["sys"]) / tg,
            "prefill_netmJ_per_tok_pkg": (p["pkg"] - i["pkg"]) / pp * 1000,
            "prefill_netmJ_per_tok_sys": (p["sys"] - i["sys"]) / pp * 1000,
            "n_samples_idle/decode/prefill": f'{i["n"]}/{d["n"]}/{p["n"]}',
        }
    a, b, c, d = (rows[g] for g in GROUPS)
    pct = lambda x, y: (x / y - 1) * 100
    print("组 | pp512 | tg128 | 解码有效频率 | 封装功率 待机/解码/预填充 | 解码净 J/token | 预填充净 mJ/token")
    for g in GROUPS:
        r = rows[g]
        print(f'{g:13s} | {r["pp512"]:.1f}±{r["pp512_sd"]:.1f} | {r["tg128"]:.2f}±{r["tg128_sd"]:.2f} | {r["decode_eff_freq_MHz"]:.0f} | '
              f'{r["idle_pkg_W"]:.1f}/{r["decode_pkg_W"]:.1f}/{r["prefill_pkg_W"]:.1f} | {r["decode_netJ_per_tok_pkg"]:.3f} | {r["prefill_netmJ_per_tok_pkg"]:.0f}')
    print(f'\n插电 能效 vs 性能：pp512 {pct(b["pp512"], a["pp512"]):+.1f}%  tg128 {pct(b["tg128"], a["tg128"]):+.1f}%')
    print(f'电池 能效 vs 性能：pp512 {pct(d["pp512"], c["pp512"]):+.1f}%  tg128 {pct(d["tg128"], c["tg128"]):+.1f}%  '
          f'解码有效频率 {pct(d["decode_eff_freq_MHz"], c["decode_eff_freq_MHz"]):+.1f}%  预填充有效频率 {pct(d["prefill_eff_freq_MHz"], c["prefill_eff_freq_MHz"]):+.1f}%  '
          f'Ring {pct(d["decode_ring_MHz"], c["decode_ring_MHz"]):+.1f}%  解码净能耗 {pct(d["decode_netJ_per_tok_pkg"], c["decode_netJ_per_tok_pkg"]):+.1f}%')
    print(f'电池性能 vs 插电性能：tg128 {pct(c["tg128"], a["tg128"]):+.1f}%  解码核心频率 {pct(c["decode_freq_MHz"], a["decode_freq_MHz"]):+.1f}%  '
          f'封装功率 {pct(c["decode_pkg_W"], a["decode_pkg_W"]):+.1f}%  解码净能耗 {pct(c["decode_netJ_per_tok_pkg"], a["decode_netJ_per_tok_pkg"]):+.1f}%')
    n = 20
    for key in ("pp512", "tg128"):
        contrast = (b[key] - a[key]) - (d[key] - c[key])
        vs = [r[key + "_sd"] ** 2 / n for r in (a, b, c, d)]
        se = math.sqrt(sum(vs)); df = sum(vs) ** 2 / sum(v * v / (n - 1) for v in vs)
        print(f"交互对比 {key}：{contrast:.2f} t/s，Welch t = {contrast / se:.2f}，df ≈ {df:.1f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(a)); w.writeheader()
        for g in GROUPS:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[g].items()})
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
