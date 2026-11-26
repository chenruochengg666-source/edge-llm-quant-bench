# -*- coding: utf-8 -*-
"""
噪声对照实验：验证"准确率反升"是否为量化特有，还是噪声的统计效应。

设计思路
--------
论文在 ARC-Easy 上观察到 Qwen0.5B 的 Q4_K_M/Q2_K 档准确率高于 FP16（能力反升区）。
可能的解释有两种：
  (A) 量化特有：量化噪声以某种方式"正则化"了模型，使其决策更好；
  (B) 统计效应：任何形式的扰动（包括纯高斯噪声）都可能改变预测分布，
      在标签分布不均匀的题集上表现为准确率的偶然上升。

本实验用最简的扰动——采样温度（temperature）——来近似 (B)。
temperature>0 时，softmax 前的 logits 被温度缩放并引入采样随机性，
效果上等价于对 logits 施加幅度与温度相关的扰动。

判定标准：
  - 若 T>0 时准确率同样上升（相对 T=0 基线），则"反升"更可能来自分布扰动效应；
  - 若 T>0 时准确率单调下降，则"反升"更可能是量化特有的。

用法
----
    python run_noise_control.py

输出
----
    05_raw/noise_control/{model}_T{temp}_S{seed}.jsonl
    06_results/noise_control_summary.csv
"""
import json, os, re, time, subprocess, requests, random, argparse

# ================= 避免 Clash / 系统代理干扰 =================
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
os.environ["no_proxy"] = "127.0.0.1,localhost,0.0.0.0"
NO_PROXY_ARG = {"http": None, "https": None}
# ============================================================

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR    = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE    = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Easy-test-full.jsonl"
OUT_DIR      = r"C:\Users\cjy13\quant_eval\05_raw\noise_control"
RESULT_CSV   = r"C:\Users\cjy13\quant_eval\06_results\noise_control_summary.csv"

# 被测模型：FP16 基线（反升现象出现在 Qwen0.5B 上）
MODEL = "qwen0.5b-base.gguf"

# 温度梯度：0 = 确定性基线，其余为扰动强度
TEMPERATURES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]   # 密梯度：覆盖弱扰动区间，才能定位峰值

# 每个温度重复的随机种子（T>0 时有随机性，需多次取均值）
SEEDS = [42, 100042, 200042]   # 大间隔：llama.cpp 在相邻种子值上可能碰撞（实测 43 与 44 输出完全一致）

# 题量：None = 全量 2376；设为 500 可把总耗时压到约 25 分钟
SUBSET = None   # None = 全量 2376，与主实验 FP16 基线可直接对齐

CTX_SIZE = "2048"      # FP16 模型用 2048 省内存
MAX_TOKENS = 64
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)


def kill_all_servers():
    os.system('taskkill /F /IM llama-server.exe > nul 2>&1')
    time.sleep(5)


def start_server(model_path):
    kill_all_servers()
    port = random.randint(8100, 9000)
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", CTX_SIZE, "-t", "8", "--port", str(port)]
    print(f"  [启动服务] 端口 {port} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies=NO_PROXY_ARG)
            return proc, port
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"服务启动超时: {model_path}")


def warmup(port, max_wait=300):
    """反复探测直到模型真正就绪（health 通过时模型可能仍在加载）。"""
    print("  [预热] 探测中 ...")
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < max_wait:
        attempt += 1
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                              json={"prompt": "Hello", "max_tokens": 1, "temperature": 0},
                              timeout=30, proxies=NO_PROXY_ARG)
            if r.status_code == 200:
                print(f"  [预热] 就绪，耗时 {time.time()-t0:.1f} 秒（第 {attempt} 次）")
                return True
            if r.status_code == 503:
                print(f"  [预热] 加载中（503），第 {attempt} 次，等待 5 秒 ...")
                time.sleep(5); continue
            print(f"  [预热] 返回码 {r.status_code}，等待 5 秒 ...")
            time.sleep(5)
        except Exception as e:
            print(f"  [预热] 异常（第 {attempt} 次）：{e}")
            time.sleep(5)
    print(f"  [预热] 等待 {max_wait} 秒仍未就绪")
    return False


def extract_letter(ans: str) -> str:
    """从模型输出中提取选项字母（与主实验口径一致）。"""
    if not ans:
        return "?"
    a = ans.strip().upper()
    if a in ("A", "B", "C", "D", "E"):
        return a
    m = re.match(r"^[\s\.\:\,\u3001\uff0c\uff1a\-\*\uff08\(]*([A-E])(?![A-Z])", a)
    if m:
        return m.group(1)
    patterns = [
        r"(?:ANSWER\s*(?:IS|:))\s*\(?([A-E])\)?",
        r"(?:答案\s*(?:是|为)?\s*[：:])\s*\(?([A-E])\)?",
        r"(?:答案\s*(?:是|为)\s*)\(?([A-E])\)?(?![\u4e00-\u9fff])",
        r"^\(([A-E])\)",
        r"^([A-E])[\.\:、，,]",
    ]
    for pat in patterns:
        m = re.search(pat, a)
        if m:
            return m.group(1)
    if len(a) <= 3:
        m = re.search(r"\b([A-E])\b", a)
        if m:
            return m.group(1)
    return "?"


def run_one(questions, port, temp, seed, out_file):
    """在给定温度与种子下跑一批题，支持续跑。"""
    done = set()
    results = []
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                try:
                    it = json.loads(line)
                except Exception:
                    continue
                results.append(it)
                if str(it.get("pred", "")).strip():
                    done.add(it["id"])
        print(f"    [续跑] 已完成 {len(done)} 题")

    correct = sum(1 for r in results if r.get("ok"))
    for idx, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        q_text = q["question"] + "\n"
        labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
        for j, opt in enumerate(q["choices"]["text"]):
            q_text += f"{labels[j]}. {opt}\n"
        q_text += "只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容：\n"

        name = MODEL.lower()
        if "phi3" in name:
            prompt = f"<|user|>\n{q_text}答案: <|end|>\n<|assistant|>\n"
        elif "ll32" in name or "llama" in name:
            prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                      f"{q_text}答案: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        elif "qwen" in name:
            prompt = f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = q_text + "答案："

        ans = ""
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": prompt, "max_tokens": MAX_TOKENS,
                      "temperature": temp, "top_p": 1, "seed": seed},
                timeout=60, proxies=NO_PROXY_ARG)
            js = r.json()
            if "choices" in js and js["choices"]:
                ans = js["choices"][0]["text"].strip()
        except Exception:
            ans = ""

        pred = extract_letter(ans)
        gold = q["answerKey"]
        if isinstance(gold, int):
            gold = labels[gold]
        gold = str(gold).strip().upper()
        ok = (pred == gold)
        if ok:
            correct += 1

        rec = {"id": q["id"], "pred": pred, "gold": gold, "ok": ok,
               "raw": ans, "temperature": temp, "seed": seed}
        results.append(rec)
        done.add(q["id"])
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if idx % 50 == 0:
            print(f"      进度 {idx}/{len(questions)} | 准确率 {correct/len(done):.4f}")

    return correct / len(questions) if questions else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", default=",".join(str(t) for t in TEMPERATURES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--subset", type=int, default=SUBSET)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return

    # 种子间隔自检：提醒小间隔可能碰撞
    if len(seeds) > 1:
        gaps = [abs(seeds[i+1]-seeds[i]) for i in range(len(seeds)-1)]
        if min(gaps) < 100:
            print(f"⚠️ 种子间隔过小（最小 {min(gaps)}）：llama.cpp 在相邻种子值上可能产生相同采样序列，")
            print(f"   建议改用大间隔（如 [42, 100042, 200042]）。当前种子：{seeds}")

    with open(DATA_FILE, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f]
    if args.subset and args.subset < len(questions):
        random.seed(42)
        questions = random.sample(questions, args.subset)
    print(f"题量 {len(questions)} | 温度 {temps} | 种子 {seeds}")

    by_temp = {}
    for temp in temps:
        accs = []
        for seed in seeds:
            # T=0 时种子无意义，只跑一次
            if temp == 0.0 and seed != seeds[0]:
                continue
            out_file = os.path.join(OUT_DIR, f"{args.model}_T{temp}_S{seed}.jsonl")
            print(f"\n=== T={temp} S={seed} ===")
            try:
                proc, port = start_server(model_path)
            except Exception as e:
                print(f"  ❌ 启动失败: {e}")
                continue
            if not warmup(port):
                print("  ⚠️ 预热失败，跳过")
                proc.terminate(); kill_all_servers(); continue
            try:
                acc = run_one(questions, port, temp, seed, out_file)
                print(f"  ✅ 完成，准确率 {acc*100:.2f}%")
                accs.append(acc)
            except Exception as e:
                print(f"  ❌ 运行异常: {e}")
            finally:
                proc.terminate(); kill_all_servers(); time.sleep(3)
        if accs:
            by_temp[temp] = (sum(accs)/len(accs), accs)

    # ===== 汇总（含标准差）=====
    import statistics
    print("\n" + "=" * 78)
    print(f"{'温度':<8}{'平均准确率(%)':>14}{'标准差':>10}{'相对基线':>12}   各次明细")
    print("-" * 78)
    rows = []
    base_mean = by_temp.get(0.0, (None, []))[0]
    for t in sorted(by_temp):
        mean, accs = by_temp[t]
        sd = statistics.pstdev(accs) * 100 if len(accs) > 1 else 0.0
        delta = (mean - base_mean) * 100 if base_mean is not None else 0.0
        flag = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
        print(f"{t:<8}{mean*100:>14.2f}{sd:>10.2f}{delta:>+11.2f}{flag}  "
              f"{', '.join(f'{a*100:.2f}' for a in accs)}")
        rows.append([t, round(mean*100, 2), round(sd, 2), round(delta, 2),
                     " | ".join(f"{a*100:.2f}" for a in accs), len(accs)])

    if rows:
        os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
        import csv
        with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["温度", "平均准确率(%)", "标准差", "相对基线(pp)", "各次准确率", "重复次数"])
            w.writerows(rows)
        print(f"\n✅ 汇总已保存: {RESULT_CSV}")

        # ===== 绘图 =====
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            sds = [r[2] for r in rows]
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
            ax.errorbar(xs, ys, yerr=sds, marker="o", markersize=5,
                        linewidth=1.6, capsize=3, color="#1F4E79")
            ax.axhline(y=base_mean * 100, ls="--", lw=1.1, color="#C00000",
                       label=f"T=0 基线 ({base_mean*100:.2f}%)")
            # 标注峰值
            if len(ys) > 1:
                imax = max(range(len(ys)), key=lambda i: ys[i])
                if ys[imax] > base_mean * 100 + 0.5:
                    ax.annotate(f"峰值 {ys[imax]:.2f}%",
                                xy=(xs[imax], ys[imax]),
                                xytext=(0, 12), textcoords="offset points",
                                ha="center", fontsize=9, color="#C00000")
            ax.set_xlabel("采样温度 T")
            ax.set_ylabel("ARC-Easy 准确率 (%)")
            ax.set_title("噪声对照：FP16 模型在不同温度扰动下的准确率")
            ax.grid(alpha=0.3, ls=":")
            ax.legend(loc="lower left", fontsize=9)
            fig.tight_layout()
            fig_path = RESULT_CSV.replace(".csv", ".png")
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"✅ 曲线图已保存: {fig_path}")
        except Exception as e:
            print(f"⚠️ 绘图失败（不影响数据）: {e}")

    # ===== 自动判读 =====
    if base_mean is not None:
        print("\n" + "=" * 78)
        print("=== 自动判读 ===")
        print(f"  T=0 基线: {base_mean*100:.2f}%")
        ups = [(t, by_temp[t][0]) for t in by_temp if t > 0 and by_temp[t][0] > base_mean]
        if ups:
            peak_t, peak_acc = max(ups, key=lambda x: x[1])
            print(f"  ✅ {len(ups)} 个温度点高于基线，峰值出现在 T={peak_t}（{peak_acc*100:.2f}%，"
                  f"+{(peak_acc-base_mean)*100:.2f}pp）")
            print("     → 弱扰动可复现'反升'，支持【统计效应】解释；")
            print("       论文应把反升从'量化正则化'改写为'分布扰动的统计表现'。")
        else:
            print("  ❌ 所有 T>0 均不高于基线 → 反升更可能是【量化特有】")
            print("     论文可写：单纯决策随机化不能复现反升，成因在权重层。")
        # 单调性检查
        vals = [by_temp[t][0] for t in sorted(by_temp)]
        mono_down = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        print(f"  单调递减: {'是' if mono_down else '否'}（共 {len(vals)} 个温度点）")


if __name__ == "__main__":
    main()
