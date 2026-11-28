# -*- coding: utf-8 -*-
"""
CPU 权重在线重排（repack）内存复测分析
======================================
输入（由 scripts/01_benchmarks/run_supplement_tests.ps1 -Test Repack 生成）：
  results/raw/rss_repack/rss_repack.csv     5 个模型 × 默认 / --no-repack × 3 次的峰值 RSS（ctx=4096，PeakWorkingSet64）
  results/raw/rss_repack/speed_repack_interleaved.csv  qwen1.5b-Q4_K_M 默认 / --no-repack 交替 6 轮（ABBA）的预填充与生成速度
  results/raw/rss_repack/speed_repack.csv   首批顺序测量（先默认后关闭，各 3 次），受顺序影响，仅供追溯
GGUF 张量类型统计：
  若设置了 EDGE_LLM_MODEL_DIR 且其中有对应 .gguf，则直接解析文件头统计各量化类型张量体积，
  并写出 results/summary/gguf_tensor_types.csv；否则读取仓库中已有的 gguf_tensor_types.csv。
判断：RSS(默认) − RSS(--no-repack) 应等于该模型 Q4_K 张量体积（重排副本），对照模型 Q5_K_M 应为 0。
用法（仓库根目录下）：python scripts/03_analysis/analyze_rss_repack.py
输出：results/summary/rss_repack_summary.csv、results/summary/speed_repack_summary.csv
"""
import csv, math, os, statistics as st, struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw" / "rss_repack"
SUM = REPO / "results" / "summary"
MODEL_DIRS = [p for p in os.environ.get("EDGE_LLM_MODEL_DIR", str(REPO / "models")).split(os.pathsep) if p]
MIB = 1024 * 1024

# ggml 类型编号 → (名称, 每块字节数, 每块元素数)
GGML = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32),
        7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32), 9: ("Q8_1", 36, 32), 10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256),
        12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 15: ("Q8_K", 292, 256), 30: ("BF16", 2, 1)}
VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_tensor_types(path):
    """只读 GGUF 文件头，返回 {类型名: MiB}。"""
    with open(path, "rb") as f:
        def u32(): return struct.unpack("<I", f.read(4))[0]
        def u64(): return struct.unpack("<Q", f.read(8))[0]
        def string(): return f.read(u64())
        def skip_value(t):
            if t == 8: string()
            elif t == 9:
                et, n = u32(), u64()
                if et in VT_SIZE: f.seek(VT_SIZE[et] * n, 1)
                else:
                    for _ in range(n): skip_value(et)
            else: f.seek(VT_SIZE[t], 1)
        assert f.read(4) == b"GGUF", "不是 GGUF 文件"
        u32(); n_tensors, n_kv = u64(), u64()
        for _ in range(n_kv):
            string(); skip_value(u32())
        out = {}
        for _ in range(n_tensors):
            string(); nd = u32(); n = math.prod(u64() for _ in range(nd)); t = u32(); u64()
            name, bsz, bel = GGML.get(t, (f"type{t}", 0, 1))
            out[name] = out.get(name, 0) + n // bel * bsz / MIB
        return out


def find_model(name):
    for d in MODEL_DIRS:
        p = os.path.join(d, name + ".gguf")
        if os.path.exists(p):
            return p
    return None


def main():
    with open(RAW / "rss_repack.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    models = list(dict.fromkeys(r["Model"] for r in rows))

    types_csv = SUM / "gguf_tensor_types.csv"
    types = {}
    if all(find_model(m) for m in models):
        for m in models:
            types[m] = gguf_tensor_types(find_model(m))
        allt = sorted({t for v in types.values() for t in v})
        with open(types_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["Model"] + [f"{t}_MiB" for t in allt])
            for m in models:
                w.writerow([m] + [round(types[m].get(t, 0), 2) for t in allt])
        print(f"已从 GGUF 文件头解析张量类型 → {types_csv}")
    else:
        with open(types_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                types[r["Model"]] = {k[:-4]: float(v) for k, v in r.items() if k.endswith("_MiB")}
        print(f"未找到模型文件，使用仓库中的 {types_csv.name}")

    out = []
    print("\n模型 | 文件 MiB | Q4_K 张量 MiB | RSS 默认 | RSS --no-repack | 差值 | 差值 − Q4_K | 默认/文件 | 关闭后/文件 | 固定开销")
    for m in models:
        sel = [r for r in rows if r["Model"] == m]
        a = [float(r["Peak_RSS_MiB"]) for r in sel if r["NoRepack"] == "False"]
        b = [float(r["Peak_RSS_MiB"]) for r in sel if r["NoRepack"] == "True"]
        size = float(sel[0]["File_MiB"]); q4k = types[m].get("Q4_K", 0.0)
        d = st.mean(a) - st.mean(b)
        rec = {"Model": m, "File_MiB": size, "Q4_K_MiB": round(q4k, 2),
               "RSS_default_MiB": round(st.mean(a), 2), "RSS_default_SD": round(st.stdev(a), 3),
               "RSS_norepack_MiB": round(st.mean(b), 2), "RSS_norepack_SD": round(st.stdev(b), 3),
               "Delta_MiB": round(d, 2), "Delta_minus_Q4K_MiB": round(d - q4k, 2),
               "Ratio_default": round(st.mean(a) / size, 3), "Ratio_norepack": round(st.mean(b) / size, 3),
               "Overhead_norepack_MiB": round(st.mean(b) - size, 1), "n": len(a)}
        out.append(rec)
        print(f'{m:16s} | {size:7.1f} | {q4k:7.1f} | {rec["RSS_default_MiB"]:8.2f} | {rec["RSS_norepack_MiB"]:8.2f} | '
              f'{d:7.2f} | {d - q4k:+6.2f} | {rec["Ratio_default"]:.2f} | {rec["Ratio_norepack"]:.2f} | {rec["Overhead_norepack_MiB"]:.1f}')
    with open(SUM / "rss_repack_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # 速度：以交替补测（speed_repack_interleaved.csv，6 轮 ABBA，按轮配对）为准；
    # 首批顺序测量（speed_repack.csv，先默认后关闭，各 3 次）受顺序/升温影响，仅供追溯
    sp = RAW / "speed_repack_interleaved.csv"
    if sp.exists():
        with open(sp, encoding="utf-8-sig") as f:
            s = list(csv.DictReader(f))
        rounds = sorted({r["Round"] for r in s}, key=int)
        res = []
        print(f"\n速度（qwen1.5b-Q4_K_M，llama-server，-t 4，513 token 提示 + 128 token 生成；{len(rounds)} 轮交替，每轮每种设置预热 1 次后测 3 次）")
        for key, label in (("PP_tps", "预填充"), ("TG_tps", "生成")):
            a = [float(r[key]) for r in s if r["NoRepack"] == "False"]
            b = [float(r[key]) for r in s if r["NoRepack"] == "True"]
            pa = [st.mean(float(r[key]) for r in s if r["Round"] == k and r["NoRepack"] == "False") for k in rounds]
            pb = [st.mean(float(r[key]) for r in s if r["Round"] == k and r["NoRepack"] == "True") for k in rounds]
            d = [x - y for x, y in zip(pa, pb)]
            t = st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))
            gain = (st.mean(a) / st.mean(b) - 1) * 100
            print(f"  {label}：默认 {st.mean(a):.2f}±{st.stdev(a):.2f}  --no-repack {st.mean(b):.2f}±{st.stdev(b):.2f}  "
                  f"重排带来 {gain:+.1f}%  按轮配对 t({len(d) - 1}) = {t:.2f}  逐轮差值 {min(d):+.2f}~{max(d):+.2f}")
            res.append({"Metric": key, "Default_mean": round(st.mean(a), 2), "Default_SD": round(st.stdev(a), 2),
                        "NoRepack_mean": round(st.mean(b), 2), "NoRepack_SD": round(st.stdev(b), 2),
                        "Repack_gain_pct": round(gain, 1), "Paired_t": round(t, 2), "df": len(d) - 1,
                        "Round_diff_min": round(min(d), 2), "Round_diff_max": round(max(d), 2),
                        "n_rounds": len(rounds), "n_per_setting": len(a), "Source": sp.name})
        with open(SUM / "speed_repack_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)

    print(f"\n已写入 {SUM / 'rss_repack_summary.csv'}")


if __name__ == "__main__":
    main()
