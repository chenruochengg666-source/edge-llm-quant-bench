# -*- coding: utf-8 -*-
"""
Qwen1.5B Q4_K_M 并发吞吐测试与绘图脚本

用法:
  1. 先启动 llama-server（--parallel >= 最大并发数）:
       llama-server.exe -m qwen1.5b-Q4_K_M.gguf --parallel 4 -c 4096 -t 8 --port 8080
  2. 运行测试:
       python concurrent_test_qwen1.5b.py
     只画图（已有 JSON）:
       python concurrent_test_qwen1.5b.py --plot-only

输出（均在 --outdir 目录下）:
  qwen1.5b-Q4_K_M_concurrency.json    原始 JSON
  qwen1.5b-Q4_K_M_concurrency.csv     每次重复的明细（Excel 可读）
  qwen1.5b-Q4_K_M_summary.csv         每个并发级别的汇总
  qwen1.5b-Q4_K_M_concurrency.png     并发吞吐图
"""

import argparse
import concurrent.futures
import csv
import json
import os
import time
from statistics import mean, stdev

# 强制禁用代理，避免 requests 走系统代理访问 127.0.0.1
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import requests
import numpy as np
import matplotlib.pyplot as plt

# 英文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# 单请求
# ----------------------------------------------------------------------
def send_request(url, prompt, n_predict, timeout=600):
    """发送单个补全请求，返回 (生成token数, 耗时秒)"""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post(
        f"{url}/completion",
        json=payload,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    if resp.status_code != 200:
        print(f"\n[服务器报错] 状态码: {resp.status_code}")
        print(f"[服务器返回] {resp.text}")
    resp.raise_for_status()
    elapsed = time.time() - t0
    data = resp.json()
    n_tokens = data.get("tokens_predicted", n_predict)
    return n_tokens, elapsed


# ----------------------------------------------------------------------
# 指定并发级别测试
# ----------------------------------------------------------------------
def test_concurrency(url, concurrency, prompt, n_predict, repeats=3, timeout=600):
    """测试指定并发级别，返回每次 repeat 的聚合吞吐列表"""
    throughputs = []
    for r in range(repeats):
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(send_request, url, prompt, n_predict, timeout)
                       for _ in range(concurrency)]
            results = [f.result() for f in futures]
        total_time = time.time() - t0
        total_tokens = sum(nt for nt, _ in results)
        agg_ts = total_tokens / total_time
        throughputs.append(agg_ts)
        print(f"    [repeat {r+1}/{repeats}] {concurrency} 并发 -> "
              f"{total_tokens} tokens / {total_time:.2f}s = {agg_ts:.2f} tok/s")
    return throughputs


# ----------------------------------------------------------------------
# 保存 CSV
# ----------------------------------------------------------------------
def save_csv(results, model_name, args, outdir):
    """
    保存两个 CSV：
    1. <model>_concurrency.csv : 每次重复的明细
    2. <model>_summary.csv     : 每个并发级别的汇总
    """
    detail_file = os.path.join(outdir, f"{model_name}_concurrency.csv")
    summary_file = os.path.join(outdir, f"{model_name}_summary.csv")

    # ---- 1. 明细 CSV ----
    with open(detail_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "concurrency", "repeat", "aggregate_throughput_tok_s",
            "n_predict", "n_prompt", "url", "timestamp"
        ])
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for c_str, info in results.items():
            for i, ts in enumerate(info["samples"], start=1):
                writer.writerow([
                    model_name, c_str, i, f"{ts:.4f}",
                    args.n_predict, args.n_prompt, args.url, timestamp
                ])
    print(f"[OK] 明细 CSV 已保存 -> {detail_file}")

    # ---- 2. 汇总 CSV ----
    concurrencies = sorted(int(k) for k in results.keys())
    baseline = results[str(concurrencies[0])]["mean"]

    with open(summary_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "concurrency", "mean_throughput_tok_s", "stdev_tok_s",
            "theoretical_linear_tok_s", "efficiency_percent",
            "repeats", "n_predict", "n_prompt"
        ])
        for c in concurrencies:
            m = results[str(c)]["mean"]
            s = results[str(c)]["stdev"]
            theory = baseline * c
            eff = 100.0 * m / theory if theory > 0 else 0.0
            writer.writerow([
                model_name, c, f"{m:.4f}", f"{s:.4f}",
                f"{theory:.4f}", f"{eff:.2f}",
                len(results[str(c)]["samples"]),
                args.n_predict, args.n_prompt
            ])
    print(f"[OK] 汇总 CSV 已保存 -> {summary_file}")

    return detail_file, summary_file


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_results(results, model_name, out_file):
    """绘制并发吞吐图，含理论线性参考线"""
    concurrencies = sorted(int(k) for k in results.keys())
    means = [results[str(c)]["mean"] for c in concurrencies]
    stds = [results[str(c)]["stdev"] for c in concurrencies]

    baseline = means[0]
    linear = [baseline * c for c in concurrencies]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(concurrencies, linear, linestyle="--", color="gray", linewidth=1.8,
            label="Theoretical Linear (y = baseline × concurrency)")

    ax.errorbar(concurrencies, means, yerr=stds, marker="o", markersize=10,
                linewidth=2.5, color="#3B7DD8", capsize=6, label="Measured Aggregate Throughput")

    for c, m in zip(concurrencies, means):
        ax.annotate(f"{m:.1f}", xy=(c, m), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight='bold', color="#3B7DD8")

    for c, l in zip(concurrencies, linear):
        ax.annotate(f"{l:.1f}", xy=(c, l), xytext=(0, -20),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="gray")

    for c, m, l in zip(concurrencies, means, linear):
        eff = 100.0 * m / l
        ax.annotate(f"Efficiency {eff:.0f}%", xy=(c, m), xytext=(0, -32),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#B23A48")

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Aggregate Throughput (tok/s)", fontsize=12)
    ax.set_title(f"{model_name} Concurrency Throughput", fontsize=14, fontweight='bold')
    ax.set_xticks(concurrencies)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"[OK] 图已保存 -> {out_file}")
    plt.show()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080",
                   help="llama-server 地址")
    p.add_argument("--model", default="qwen1.5b-Q4_K_M",
                   help="模型名，用于命名输出文件")
    p.add_argument("--n_predict", type=int, default=128,
                   help="每次生成 token 数（建议 128）")
    p.add_argument("--n_prompt", type=int, default=128,
                   help="提示词近似 token 数")
    p.add_argument("--concurrency", default="1,2,4",
                   help="并发数列表，逗号分隔")
    p.add_argument("--repeats", type=int, default=3,
                   help="每个并发级别重复次数")
    p.add_argument("--outdir", default="./concurrency_results",
                   help="结果保存目录")
    p.add_argument("--plot-only", action="store_true",
                   help="只绘图，不测试（读取已有 JSON）")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    json_file = os.path.join(args.outdir, f"{args.model}_concurrency.json")
    png_file = os.path.join(args.outdir, f"{args.model}_concurrency.png")

    # 只绘图模式
    if args.plot_only:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        plot_results(results, args.model, png_file)
        save_csv(results, args.model, args, args.outdir)
        return

    # 构造固定长度提示词
    prompt = "The quick brown fox jumps over the lazy dog. " * max(1, args.n_prompt // 10)

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    results = {}

    for c in concurrencies:
        print(f"\n>>> 测试 {c} 并发 ...")
        ts_list = test_concurrency(args.url, c, prompt, args.n_predict,
                                    args.repeats)
        results[str(c)] = {
            "mean": mean(ts_list),
            "stdev": stdev(ts_list) if len(ts_list) > 1 else 0.0,
            "samples": ts_list,
        }

    # 保存 JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 测试结果已保存 -> {json_file}")

    # 保存 CSV
    save_csv(results, args.model, args, args.outdir)

    # 绘图
    plot_results(results, args.model, png_file)


if __name__ == "__main__":
    main()