# -*- coding: utf-8 -*-
"""
power_mode_test.py —— 电源模式 2×2 析因实验（可选接入功率计）

原实验只对比了「电池+节电」与「插电+高性能」，同时变动了供电方式与性能模式两个因素，
无法分离二者的贡献。本脚本改为 2×2 析因设计：

    供电：插电(AC) / 电池(BAT)      性能：节电(Saver) / 高性能(HighPerf)

四组各跑一次 llama-bench，并可选录入功率计读数，用于计算每 token 能耗。

用法：
    python power_mode_test.py --bench "C:\\path\\to\\llama-bench.exe" --model "C:\\path\\to\\qwen1.5b-Q4KM.gguf" --out power_2x2.csv

说明：
    - 每组开始前脚本会暂停，请你手动把笔记本切到对应状态（插拔电源 / 切换 Windows 电源模式），回车继续。
    - 若手边有功率计，在提示处输入该组「解码稳态」的整机功耗（W），留空则该项记为 NaN。
    - 输出 J/token = 功耗(W) / 解码速度(tg128, t/s)，即生成单个 token 的能耗。
"""
import argparse, csv, math, os, re, subprocess, sys

GROUPS = [
    ("AC",  "HighPerf"),
    ("AC",  "Saver"),
    ("BAT", "HighPerf"),
    ("BAT", "Saver"),
]

def run_bench(bench, model, runs):
    cmd = [bench, "-m", model, "-p", "512", "-n", "128", "-r", str(runs),
           "-t", "4", "-b", "2048", "-ub", "512", "-fa", "1", "-ngl", "0"]
    print("  执行:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return out.stdout + out.stderr


def parse_bench(text):
    """解析 llama-bench 输出中的 pp512 / tg128（均值±标准差）

    不同参数下 llama-bench 的列数会变（带不带 threads、ngl 等），
    因此不按固定列号取值，而是统一取「倒数第二列 = 测试项，最后一列 = 数值」。
    """
    res = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        test, val = cells[-2], cells[-1]
        if not re.search(r"pp|tg", test, re.I):
            continue
        m = re.match(r"([\d.]+)\s*(?:±|\+/-|\+-)\s*([\d.]+)", val)
        if not m:
            continue
        avg, sd = float(m.group(1)), float(m.group(2))
        if re.search(r"pp\s*512", test, re.I):
            res["pp512"] = (avg, sd)
        elif re.search(r"tg\s*128", test, re.I):
            res["tg128"] = (avg, sd)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="llama-bench 可执行文件路径")
    ap.add_argument("--model", required=True, help="待测 gguf 模型路径")
    ap.add_argument("--out", default="power_2x2.csv")
    ap.add_argument("--runs", type=int, default=5, help="重复次数，建议 >=5")
    a = ap.parse_args()

    rows = []
    for supply, perf in GROUPS:
        label = "%s_%s" % (supply, perf)
        print("\n" + "=" * 60)
        print("即将测试：供电 = %s，性能模式 = %s" % (supply, perf))
        print("请先手动切换笔记本状态，然后按回车开始……")
        input()

        text = run_bench(a.bench, a.model, a.runs)
        raw_log = a.out.replace(".csv", "") + "_raw_%s.log" % label
        with open(raw_log, "w", encoding="utf-8") as lf:
            lf.write(text)
        print("  原始输出已保存:", raw_log)
        m = parse_bench(text)
        pp, tg = m.get("pp512"), m.get("tg128")
        if not pp or not tg:
            print("  [警告] 未能解析出 pp512/tg128。输出末尾 20 行：")
            print("\n".join(text.splitlines()[-20:]))
            continue

        watt_s = input("  若有功率计，请输入本组解码稳态整机功耗(W)，留空跳过: ").strip()
        watt = float(watt_s) if re.fullmatch(r"\d+(\.\d+)?", watt_s or "") else None
        j_per_tok = (watt / tg[0]) if watt else None

        rows.append({
            "Supply": supply, "PerfMode": perf, "Config": label,
            "pp512_mean": round(pp[0], 2), "pp512_sd": round(pp[1], 2),
            "tg128_mean": round(tg[0], 2), "tg128_sd": round(tg[1], 2),
            "Power_W": watt if watt else "",
            "J_per_token": round(j_per_tok, 3) if j_per_tok else "",
        })
        print("  ✅ %s: pp512=%.2f, tg128=%.2f" % (label, pp[0], tg[0]) +
              ("，%.2f W，%.3f J/token" % (watt, j_per_tok) if watt else ""))

    if not rows:
        print("\n没有采集到有效数据，请检查 llama-bench 路径与输出格式。")
        return

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n已输出: %s（%d 组）" % (a.out, len(rows)))
    print("提示：论文中可据此检验「供电」与「性能模式」各自的主效应及交互效应。")

if __name__ == "__main__":
    main()
