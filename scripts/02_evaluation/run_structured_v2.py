# run_structured_v2.py  (v2.1 支持续跑)
# 特性：
#   - 断点续跑：读取已有 jsonl，跳过 status=ok 的题；empty/error 会重跑
#   - 模型级跳过：若某模型全部完成，直接跳过（不启动 llama-server，省时间）
#   - 干净写盘：每 N 题按题库顺序重写文件（不产生重复行），崩溃最多丢 N 题
#   - 可选筛选：--models / --only-types / --force
#
# 用法：
#   python run_structured_v2.py                       # 全部模型，自动续跑
#   python run_structured_v2.py --models qwen1.5b     # 只跑名字含 qwen1.5b 的模型
#   python run_structured_v2.py --only-types json_extraction
#   python run_structured_v2.py --force               # 忽略已有结果，全部重跑
import argparse, json, os, re, sys, time, subprocess, requests

LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR    = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE    = r"C:\Users\cjy13\quant_eval\03_data\structured\struct_80_v2.jsonl"
OUT_DIR      = r"C:\Users\cjy13\quant_eval\05_raw\structured_v2"
PORT         = 8080
QUANT_RE     = re.compile(r"[._-](f16|base|q2_k|q3_k_m|q4_k_m|q5_k_m|q6_k|q8_0)$", re.I)


def parse_name(fname):
    base = re.sub(r"\.gguf$", "", fname, flags=re.I)
    m = QUANT_RE.search(base)
    if m:
        return base[:m.start()], m.group(1).upper()
    return base, "UNKNOWN"


def kill_servers():
    os.system("taskkill /F /IM llama-server.exe > nul 2>&1")
    time.sleep(3)


def wait_ready(port, timeout=120):
    for _ in range(timeout):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2,
                             proxies={"http": None, "https": None})
            if r.status_code == 200:
                time.sleep(2)
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(model_path, retries=2):
    for attempt in range(1, retries + 1):
        kill_servers()
        cmd = [LLAMA_SERVER, "-m", model_path, "-c", "4096", "-t", "8", "--port", str(PORT)]
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
        if wait_ready(PORT):
            print(f"     [就绪] 第 {attempt} 次启动成功")
            return proc
        print(f"     [失败] 第 {attempt} 次启动未就绪，清理重试")
        try:
            proc.terminate()
        except Exception:
            pass
    return None


def ask(port, prompt, retries=1):
    err = "UNKNOWN"
    for k in range(retries + 1):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": prompt + "\n请直接输出结果：\n",
                      "max_tokens": 256, "temperature": 0, "top_p": 1, "seed": 42},
                timeout=180, proxies={"http": None, "https": None})
            if r.status_code == 200:
                j = r.json()
                txt = (j.get("choices") or [{}])[0].get("text", "") or ""
                return txt.strip(), "ok"
            err = f"HTTP_{r.status_code}"
        except Exception as e:
            err = f"EXC_{type(e).__name__}"
        if k < retries:
            time.sleep(2)
    return "", err


def load_existing(path):
    """读取已有记录：按 id 去重，后写覆盖先写；丢弃旧的 server_failed 汇总行"""
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("id") == "__ALL__":
                continue
            if r.get("id"):
                recs[r["id"]] = r
    return recs


def dump_clean(path, recs, questions):
    """按题库顺序重写文件，保证无重复行、顺序稳定"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q in questions:
            r = recs.get(q["id"])
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="逗号分隔的模型名关键词，空=全部")
    ap.add_argument("--only-types", default="", help="逗号分隔的题型，空=全部")
    ap.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    ap.add_argument("--flush-every", type=int, default=5, help="每 N 题落盘一次")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    questions = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()]
    if args.only_types:
        keep = {t.strip() for t in args.only_types.split(",") if t.strip()}
        questions = [q for q in questions if q["type"] in keep]
        print(f"[筛选] 只跑题型: {sorted(keep)}，共 {len(questions)} 题")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".gguf")]
    if args.models:
        keys = [k.strip().lower() for k in args.models.split(",") if k.strip()]
        model_files = [f for f in model_files if any(k in f.lower() for k in keys)]
    if not model_files:
        print("没有匹配的模型文件"); return
    print(f"题库 {len(questions)} 题 | 目标模型 {len(model_files)} 个 | force={args.force}\n")

    for mf in model_files:
        model, quant = parse_name(mf)
        out_file = os.path.join(OUT_DIR, f"{mf}.jsonl")

        if args.force and os.path.exists(out_file):
            os.remove(out_file)

        recs = load_existing(out_file)
        done_ids = {i for i, r in recs.items() if r.get("status") == "ok"}
        todo = [q for q in questions if q["id"] not in done_ids]

        print(f"=== {mf} | 已完成 {len(done_ids)}/{len(questions)}，待跑 {len(todo)}")
        if not todo:
            print("     全部完成，跳过（不启动服务）\n")
            continue

        proc = start_server(os.path.join(MODEL_DIR, mf))
        if proc is None:
            print(f"     ❌ 启动失败，本模型跳过（进度已保留，下次可续跑）\n")
            continue

        n_ok = n_empty = n_err = 0
        for k, q in enumerate(todo, 1):
            txt, st = ask(PORT, q["prompt"])
            if st == "ok" and not txt:
                st = "empty"
            if st == "ok": n_ok += 1
            elif st == "empty": n_empty += 1
            else: n_err += 1

            recs[q["id"]] = {"model": model, "quant": quant, "status": st,
                             "id": q["id"], "type": q["type"], "raw_output": txt}

            if k % max(1, args.flush_every) == 0 or k == len(todo):
                dump_clean(out_file, recs, questions)

            print(f"     [{k}/{len(todo)}] ok={n_ok} empty={n_empty} err={n_err}", end="\r")

        dump_clean(out_file, recs, questions)
        print(f"\n     ✅ 本模型完成：ok={n_ok} empty={n_empty} err={n_err}"
              f"（累计 {len([1 for r in recs.values() if r.get('status')=='ok'])}/{len(questions)}）\n")
        try:
            proc.terminate()
        except Exception:
            pass
        kill_servers()

    print("全部目标模型处理完毕。若中途中断，直接重跑本脚本即可从断点继续。")


if __name__ == "__main__":
    main()
